import os
import torch
import argparse
import json
from glob import glob
import tqdm
import numpy as np
from torch.nn import functional as F
import commons
from scipy.io.wavfile import write
import torchaudio
import utils
from data_loader import MelSpectrogramFixed

import amfm_decompy.basic_tools as basic
import amfm_decompy.pYAAPT as pYAAPT

from model.vc_sdt import (
    Wav2vec2, 
    SDTVoiceConversion,
    VPSDESchedule
)
from model.styleencoder import StyleEncoder
from hifigan.vocoder import HiFi
from torch import nn

h = None
device = None
seed = 1234
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)  

def load_audio(path):
    audio, sr = torchaudio.load(path) 
    audio = audio[:1]
    if sr != 16000:
        audio = torchaudio.functional.resample(audio, sr, 16000, resampling_method="kaiser_window")
    
    p = (audio.shape[-1] // 1280 + 1) * 1280 - audio.shape[-1] 
    audio = torch.nn.functional.pad(audio, (0, p)) 
     
    return audio 


def save_audio(wav, out_file, syn_sr=16000):
    wav = (wav.squeeze() / wav.abs().max() * 0.999 * 32767.0).cpu().numpy().astype('int16')
    write(out_file, syn_sr, wav) 
        

def get_yaapt_f0(audio, sr=16000, interp=False):
    """Extract F0 using YAAPT algorithm"""
    to_pad = int(20.0 / 1000 * sr) // 2
    f0s = []
    for y in audio.astype(np.float64):
        y_pad = np.pad(y.squeeze(), (to_pad, to_pad), "constant", constant_values=0) 
        pitch = pYAAPT.yaapt(
            basic.SignalObj(y_pad, sr), 
            **{
                'frame_length': 20.0, 
                'frame_space': 5.0, 
                'nccf_thresh1': 0.25, 
                'tda_frame_length': 25.0
            }
        )
        f0s.append(pitch.samp_interp[None, None, :] if interp else pitch.samp_values[None, None, :])

    return np.vstack(f0s)


def normalize_f0(f0, src_mean, src_std, trg_mean, trg_std):
    """Normalize F0 from source to target speaker"""
    # Avoid division by zero
    src_std = max(src_std, 1e-8)
    
    # Normalize and denormalize
    f0_norm = np.where(f0 != 0, (f0 - src_mean) / src_std * trg_std + trg_mean, 0)
    
    return f0_norm


def inference(a): 
    os.makedirs(a.output_dir, exist_ok=True) 
    
    # Initialize mel-spectrogram function
    mel_fn = MelSpectrogramFixed(
        sample_rate=hps.data.sampling_rate,
        n_fft=hps.data.filter_length,
        win_length=hps.data.win_length,
        hop_length=hps.data.hop_length,
        f_min=hps.data.mel_fmin,
        f_max=hps.data.mel_fmax,
        n_mels=hps.data.n_mel_channels,
        window_fn=torch.hann_window
    ).to(device)

    # Initialize content encoder (MMS/Wav2vec2)
    content_encoder = Wav2vec2().to(device)
    
    # Initialize style encoder
    style_encoder = nn.Module()
    style_encoder.emb_g = StyleEncoder(in_dim=80, hidden_dim=256, out_dim=256).to(device)
    
    # Initialize SDT Voice Conversion model
    model = SDTVoiceConversion(
        content_encoder=content_encoder,
        style_encoder=style_encoder,
        hidden_channels=256,
        gin_channels=256,
        mel_size=(80, 128),
        patch_size=(16, 8),
        embed_dim=768,
        depth=12,
        n_heads=12,
        mlp_ratio=4.0,
        beta_min=0.05,
        beta_max=20.0,
        cfg_drop=0.0  # No dropout during inference
    ).to(device)
    
    # Load checkpoint
    utils.load_checkpoint(a.ckpt_model, model, None)
    model.eval()
 
    # Load vocoder
    net_v = HiFi(
        hps.data.n_mel_channels, 
        hps.train.segment_size // hps.data.hop_length, 
        **hps.model
    ).to(device)
    utils.load_checkpoint(a.ckpt_voc, net_v, None)
    net_v.eval()
    net_v.dec.remove_weight_norm()   
 
    # Process source audio
    print('>> Loading source audio...')
    src_name = os.path.splitext(os.path.basename(a.src_path))[0]
    src_audio = load_audio(a.src_path)   
    
    # Extract source F0
    try:
        src_f0 = get_yaapt_f0(src_audio.numpy())
    except:
        print("Warning: F0 extraction failed, using zeros")
        src_f0 = np.zeros((1, 1, src_audio.shape[-1] // 80), dtype=np.float32)
    
    # Process target audio  
    print('>> Loading target audio...')
    trg_name = os.path.splitext(os.path.basename(a.trg_path))[0] 
    trg_audio = load_audio(a.trg_path)    
    
    # Extract target F0 for normalization
    try:
        trg_f0 = get_yaapt_f0(trg_audio.numpy())
    except:
        trg_f0 = np.zeros((1, 1, trg_audio.shape[-1] // 80), dtype=np.float32)
    
    # F0 normalization (source to target)
    src_f0_nonzero = src_f0[src_f0 != 0]
    trg_f0_nonzero = trg_f0[trg_f0 != 0]
    
    if len(src_f0_nonzero) > 0 and len(trg_f0_nonzero) > 0:
        src_mean = src_f0_nonzero.mean()
        src_std = src_f0_nonzero.std()
        trg_mean = trg_f0_nonzero.mean()
        trg_std = trg_f0_nonzero.std()
        
        # Normalize F0
        f0_normalized = normalize_f0(src_f0, src_mean, src_std, trg_mean, trg_std)
    else:
        f0_normalized = src_f0
    
    # Convert to log scale as in the paper
    f0_tensor = torch.FloatTensor(f0_normalized).to(device)
    
    # Compute mel-spectrograms
    src_mel = mel_fn(src_audio.to(device))
    trg_mel = mel_fn(trg_audio.to(device))
    
    # Prepare lengths
    src_length = torch.LongTensor([src_mel.size(-1)]).to(device)
    trg_length = torch.LongTensor([trg_mel.size(-1)]).to(device)
    
    # Pad source audio for wav2vec2
    src_audio_padded = F.pad(src_audio, (40, 40), "reflect")
    
    print('>> Performing voice conversion...')
    with torch.no_grad():
        # Perform voice conversion
        converted_mel = model.voice_conversion(
            source_wav=src_audio_padded.to(device),
            source_f0=f0_tensor,
            target_mel=trg_mel,
            source_lengths=src_length,
            target_lengths=trg_length,
            num_steps=a.time_step,
            guidance_scale=a.guidance_scale,
            guidance_mode=a.guidance_mode
        )
        
        # Crop to match source length
        converted_mel = converted_mel[:, :, :src_mel.size(-1)]
        
        # Generate audio with vocoder
        converted_audio = net_v(converted_mel)
    
    # Save output
    f_name = f'{src_name}_to_{trg_name}_{a.guidance_mode}_scale{a.guidance_scale}.wav' 
    out_path = os.path.join(a.output_dir, f_name)
    save_audio(converted_audio, out_path)   
    print(f">> Saved: {out_path}")
    print(">> Done.")
     

def main():
    print('>> Initializing SDT Inference...')
     
    parser = argparse.ArgumentParser()
    parser.add_argument('--src_path', type=str, required=True,
                        help='Path to source audio file')  
    parser.add_argument('--trg_path', type=str, required=True,
                        help='Path to target audio file for style')  
    parser.add_argument('--ckpt_model', type=str, default='./logs/sdt/G_latest.pth',
                        help='Path to SDT model checkpoint')
    parser.add_argument('--ckpt_voc', type=str, default='./hifigan/G_2930000.pth',
                        help='Path to vocoder checkpoint')   
    parser.add_argument('--output_dir', '-o', type=str, default='./converted',
                        help='Output directory')  
    parser.add_argument('--time_step', '-t', type=int, default=6,
                        help='Number of diffusion steps (default: 6)')
    parser.add_argument('--guidance_mode', '-gm', type=str, default='source',
                        choices=['source', 'null', 'dual'],
                        help='Pitch guidance mode')
    parser.add_argument('--guidance_scale', '-gs', type=float, default=1.0,
                        help='Guidance scale for dual mode (0.0-2.0)')
    
    global hps, device, a
    
    a = parser.parse_args()
    
    # Load config
    config_path = os.path.join(os.path.split(a.ckpt_model)[0], 'config.json')
    if os.path.exists(config_path):
        hps = utils.get_hparams_from_file(config_path)
    else:
        # Use default config
        hps = utils.get_hparams()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'>> Using device: {device}')
    
    inference(a)


if __name__ == '__main__':
    main()