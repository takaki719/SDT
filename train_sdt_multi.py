import os
import torch
from torch.nn import functional as F
from torch.nn.parallel import DistributedDataParallel as DDP

import torch.distributed as dist
import torch.multiprocessing as mp
from torch.cuda.amp import autocast, GradScaler

from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data.distributed import DistributedSampler

import random
import commons
import utils

from augmentation.aug import Augment
from model_f0_vqvae import Quantizer
from model.vc_sdt import (
    Wav2vec2, 
    SpeechDiffusionTransformer,
    VPSDESchedule,
    SpeechDisentanglementEncoder,
    SDTVoiceConversion
)
from data_loader import AudioDataset, MelSpectrogramFixed
from hifigan.vocoder import HiFi
from torch.utils.data import DataLoader
from torch import nn, Tensor

torch.backends.cudnn.benchmark = True
global_step = 0

def get_param_num(model):
    """モデルのパラメータ数を計算するヘルパー関数"""
    num_param = sum(param.numel() for param in model.parameters())
    return num_param

@torch.no_grad()
def sample(model: SDTVoiceConversion,
           source_wav: Tensor,
           source_f0: Tensor,
           target_mel: Tensor,
           source_lengths: Tensor,
           target_lengths: Tensor,
           steps: int = 6,
           guidance_mode: str = 'source'):
    """Voice conversion sampling"""
    model.eval()
    
    # Use the voice_conversion method from the model
    mel_converted = model.voice_conversion(
        source_wav=source_wav,
        source_f0=source_f0,
        target_mel=target_mel,
        source_lengths=source_lengths,
        target_lengths=target_lengths,
        num_steps=steps,
        guidance_scale=1.0,
        guidance_mode=guidance_mode
    )
    
    return mel_converted

def main():
    """Assume Single Node Multi GPUs Training Only"""
    assert torch.cuda.is_available(), "CPU training is not allowed."

    n_gpus = torch.cuda.device_count()
    port = 50000 + random.randint(0, 100)
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = str(port)

    hps = utils.get_hparams()
    mp.spawn(run, nprocs=n_gpus, args=(n_gpus, hps,))

def run(rank, n_gpus, hps):
    global global_step
    if rank == 0:
        logger = utils.get_logger(hps.model_dir)
        logger.info(hps)
        utils.check_git_hash(hps.model_dir)
        writer = SummaryWriter(log_dir=hps.model_dir)
        writer_eval = SummaryWriter(log_dir=os.path.join(hps.model_dir, "eval"))

    dist.init_process_group(backend='nccl', init_method='env://', world_size=n_gpus, rank=rank)
    torch.manual_seed(hps.train.seed)
    torch.cuda.set_device(rank)

    # 音声波形をメルスペクトログラムに変換するクラスを初期化
    mel_fn = MelSpectrogramFixed(
        sample_rate=hps.data.sampling_rate,
        win_length=hps.data.win_length,
        n_fft=hps.data.filter_length,
        hop_length=hps.data.hop_length,
        f_min=hps.data.mel_fmin,
        f_max=hps.data.mel_fmax,
        n_mels=hps.data.n_mel_channels,
        window_fn=torch.hann_window
    ).cuda(rank)

    # 音声データセットを初期化
    train_dataset = AudioDataset(hps, training=True)
    train_sampler = DistributedSampler(train_dataset) if n_gpus > 1 else None
    train_loader = DataLoader(
        train_dataset, 
        batch_size=hps.train.batch_size, 
        num_workers=32,
        sampler=train_sampler, 
        drop_last=True, 
        persistent_workers=True, 
        pin_memory=True
    )

    if rank == 0:
        test_dataset = AudioDataset(hps, training=False)
        eval_loader = DataLoader(test_dataset, batch_size=1)

        # Vocoder for evaluation
        net_v = HiFi(
            hps.data.n_mel_channels,
            hps.train.segment_size // hps.data.hop_length,
            **hps.model
        ).cuda()
        
        path_ckpt = './hifigan/G_2930000.pth'
        utils.load_checkpoint(path_ckpt, net_v, None)
        net_v.eval()
        net_v.dec.remove_weight_norm()
    else:
        net_v = None
    
    # Initialize content encoder (Wav2vec2/MMS)
    content_encoder = Wav2vec2().cuda(rank)
    
    # Initialize style encoder (from SynthesizerTrn)
    from model.styleencoder import StyleEncoder
    style_encoder = nn.Module()  # Placeholder - use your existing SynthesizerTrn
    style_encoder.emb_g = StyleEncoder(in_dim=80, hidden_dim=256, out_dim=256).cuda(rank)
    
    # Data augmentation
    aug = Augment(hps).cuda(rank)
    
    # Initialize complete SDT Voice Conversion model
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
        cfg_drop=0.2
    ).cuda(rank)
    
    # Print model parameters
    if rank == 0:
        logger.info(f"Model parameters: {get_param_num(model):,}")
    
    # Initialize optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        hps.train.learning_rate,
        betas=hps.train.betas,
        eps=hps.train.eps
    )
    
    # Wrap model with DDP
    model = DDP(model, device_ids=[rank])
    
    # Load checkpoint if exists
    try:
        _, _, _, epoch_str = utils.load_checkpoint(
            utils.latest_checkpoint_path(hps.model_dir, "G_*.pth"), 
            model, 
            optimizer
        )
        global_step = (epoch_str - 1) * len(train_loader)
    except:
        epoch_str = 1
        global_step = 0
    
    # Mixed precision scaler
    scaler = GradScaler(enabled=hps.train.fp16_run)
    
    # Training loop
    for epoch in range(epoch_str, hps.train.epochs + 1):
        if rank == 0:
            train_and_evaluate(
                rank, epoch, hps, 
                [model, mel_fn, aug, net_v], 
                optimizer, scaler, 
                [train_loader, eval_loader], 
                logger, 
                [writer, writer_eval], 
                n_gpus
            )
        else:
            train_and_evaluate(
                rank, epoch, hps, 
                [model, mel_fn, aug, net_v], 
                optimizer, scaler, 
                [train_loader, None], 
                None, None, 
                n_gpus
            )

def train_and_evaluate(rank, epoch, hps, nets, optimizer, scaler, loaders, logger, writers, n_gpus):
    model, mel_fn, aug, net_v = nets
    train_loader, eval_loader = loaders
    
    if writers is not None:
        writer, writer_eval = writers
    
    global global_step
    
    if n_gpus > 1:
        train_loader.sampler.set_epoch(epoch)
    
    model.train()
    
    for batch_idx, (x, x_f0, length) in enumerate(train_loader):
        # Move data to GPU
        x = x.cuda(rank, non_blocking=True)
        x_f0 = x_f0.cuda(rank, non_blocking=True)
        length = length.cuda(rank, non_blocking=True).squeeze()
        
        # Compute mel-spectrogram
        mel_x = mel_fn(x)
        
        # Apply augmentation
        aug_x = aug(x)
        nan_x = torch.isnan(aug_x).any()
        x_aug = x if nan_x else aug_x
        
        # Pad audio for wav2vec2
        x_pad = F.pad(x_aug, (40, 40), "reflect")
        
        # Zero gradients
        optimizer.zero_grad()
        
        # Compute loss
        with autocast(enabled=hps.train.fp16_run):
            loss_dict = model.module.compute_loss(
                mel_gt=mel_x,
                wav_input=x_pad,
                f0=x_f0,
                lengths=length
            )
            loss_total = loss_dict['loss']
        
        # Backward pass
        if hps.train.fp16_run:
            scaler.scale(loss_total).backward()
            scaler.unscale_(optimizer)
            grad_norm = commons.clip_grad_value_(model.parameters(), None)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss_total.backward()
            grad_norm = commons.clip_grad_value_(model.parameters(), None)
            optimizer.step()
        
        # Logging
        if rank == 0:
            if global_step % hps.train.log_interval == 0:
                lr = optimizer.param_groups[0]['lr']
                logger.info(
                    f'Train Epoch: {epoch} [{100. * batch_idx / len(train_loader):.0f}%] '
                    f'Loss: {loss_total.item():.4f}'
                )
                
                scalar_dict = {
                    "loss/total": loss_total.item(),
                    "loss/diff": loss_dict['loss_diff'].item(),
                    "loss/recon": loss_dict['loss_recon'].item(),
                    "learning_rate": lr,
                    "grad_norm": grad_norm
                }
                
                utils.summarize(
                    writer=writer,
                    global_step=global_step,
                    scalars=scalar_dict
                )
            
            if global_step % hps.train.eval_interval == 0:
                torch.cuda.empty_cache()
                evaluate(hps, model, mel_fn, net_v, eval_loader, writer_eval)
                
                if global_step % hps.train.save_interval == 0:
                    utils.save_checkpoint(
                        model, optimizer, hps.train.learning_rate, epoch,
                        os.path.join(hps.model_dir, f"G_{global_step}.pth")
                    )
        
        global_step += 1
    
    if rank == 0:
        logger.info(f'====> Epoch: {epoch}')


def evaluate(hps, model, mel_fn, net_v, eval_loader, writer_eval):
    model.eval()
    image_dict = {}
    audio_dict = {}
    mel_loss_total = 0
    
    with torch.no_grad():
        for batch_idx, (y, y_f0, y_length) in enumerate(eval_loader):
            if batch_idx > 10:  # Evaluate only first 10 samples
                break
                
            # Move to GPU
            y = y.cuda(0)
            y_f0 = y_f0.cuda(0)
            y_length = y_length.cuda(0)
            
            # Compute mel-spectrogram
            mel_y = mel_fn(y)
            
            # Pad audio for wav2vec2
            y_pad = F.pad(y, (40, 40), "reflect")
            
            # Perform voice conversion (resynthesis for evaluation)
            mel_converted = sample(
                model.module,
                source_wav=y_pad,
                source_f0=y_f0,
                target_mel=mel_y,
                source_lengths=y_length,
                target_lengths=y_length,
                steps=6,
                guidance_mode='source'
            )
            
            # Crop to match target length
            mel_converted = mel_converted[:, :, :mel_y.size(2)]
            
            # Calculate loss
            mel_loss = F.l1_loss(mel_y, mel_converted).item()
            mel_loss_total += mel_loss
            
            # Generate audio for first 4 samples
            if batch_idx < 4:
                y_hat = net_v(mel_converted)
                
                # Plot mel-spectrograms
                plot_mel = torch.cat([mel_y, mel_converted], dim=1)
                plot_mel = plot_mel.clip(min=-10, max=10)
                
                image_dict.update({
                    f"gen/mel_{batch_idx}": utils.plot_spectrogram_to_numpy(
                        plot_mel.squeeze().cpu().numpy()
                    )
                })
                
                audio_dict.update({
                    f"gen/audio_{batch_idx}": y_hat.squeeze(),
                })
                
                if global_step == 0:
                    audio_dict.update({
                        f"gt/audio_{batch_idx}": y.squeeze()
                    })
    
    mel_loss_avg = mel_loss_total / min(10, len(eval_loader))
    
    scalar_dict = {"val/mel_loss": mel_loss_avg}
    
    utils.summarize(
        writer=writer_eval,
        global_step=global_step,
        images=image_dict,
        audios=audio_dict,
        audio_sampling_rate=hps.data.sampling_rate,
        scalars=scalar_dict
    )
    
    model.train()


if __name__ == "__main__":
    main()