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
from model.vc_sdt import Wav2vec2, SpeechDiffusionTransformer,VPSDESchedule
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
def sample(model: nn.Module,
           scheduler: VPSDESchedule,
           Zsrc: Tensor,
           Zftr: Tensor,
           espk: Tensor,
           steps: int = 15):
    model.eval()
    B = Zsrc.size(0)
    device = Zsrc.device

    # x_T ~ N(0, I)
    x = torch.randn(B, 80, 128, device=device)
    dt = -1.0 / steps  # 時間を 1→0 へ均等ステップ

    t = torch.ones(B, device=device)  # 初期時刻 1
    for _ in range(steps):
        eps_pred = model(x, t, Zsrc, Zftr, espk)
        x = scheduler.step_euler(x, t, eps_pred, dt)
        t = t + dt
        t = t.clamp(min=0.0)
    return x 

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

    # 音声波形をメルスペクトログラムに変換するクラスを初期化。
    mel_fn = MelSpectrogramFixed(
        sample_rate=hps.data.sampling_rate,
        n_fft=hps.data.filter_length,
        win_length=hps.data.win_length,
        hop_length=hps.data.hop_length,
        f_min=hps.data.mel_fmin,
        f_max=hps.data.mel_fmax,
        n_mels=hps.data.n_mel_channels,
        window_fn=torch.hann_window
    ).cuda(rank)

    # 音声データセットを初期化。
    train_dataset = AudioDataset(hps, training=True)
    train_sampler = DistributedSampler(train_dataset) if n_gpus > 1 else None
    train_loader = DataLoader(
        train_dataset, batch_size=hps.train.batch_size, num_workers=32,
        sampler=train_sampler, drop_last=True, persistent_workers=True, pin_memory=True
    )

    if rank == 0:
        test_dataset = AudioDataset(hps, training=False)
        eval_loader = DataLoader(test_dataset, batch_size=1)

        net_v = HiFi(
            hps.data.n_mel_channels,
            hps.train.segment_size // hps.data.hop_length,
            **hps.model).cuda()
        path_ckpt = './hifigan/G_2930000.pth'

        utils.load_checkpoint(path_ckpt, net_v, None)
        net_v.eval()
        net_v.dec.remove_weight_norm()
    else:
        net_v = None
    
    #Wav2vec2モデルを初期化
    w2v = Wav2vec2().cuda(rank)
    aug = Augment(hps).cuda(rank)

    #SDTモデルを初期化
    model = SpeechDiffusionTransformer(cond_dim=1024).cuda(rank)

    # F0（基本周波数）を量子化するモデルを初期化
    f0_quantizer = Quantizer(hps).cuda(rank)
    utils.load_checkpoint('./f0_vqvae/f0_vqvae.pth', f0_quantizer)
    f0_quantizer.eval() 
     
    # オプティマイザ（最適化アルゴリズム）を設定。AdamWを使用。
    optimizer = torch.optim.AdamW(
        model.parameters(),
        hps.train.learning_rate,
        betas=hps.train.betas,
        eps=hps.train.eps)

    # モデルをDDPでラップする。これにより、各GPUで計算された勾配が自動的に集約・同期される。
    model = DDP(model, device_ids=[rank])

    try:
        # 最新のチェックポイント（学習途中保存したモデル）を探してロードする。
        _, _, _, epoch_str = utils.load_checkpoint(utils.latest_checkpoint_path(hps.model_dir, "G_*.pth"), model, optimizer) 
        global_step = (epoch_str - 1) * len(train_loader)
    except:
        # チェックポイントがない場合は、最初から学習を開始する。
        epoch_str = 1
        global_step = 0


    # 混合精度学習（fp16）が有効な場合、勾配スケーラーを初期化する。
    scaler = GradScaler(enabled=hps.train.fp16_run)
    # 学習率スケジューラを設定。
    scheduler = VPSDESchedule(beta_min=hps.diffusion.beta_min,
                          beta_max=hps.diffusion.beta_max)

    # 学習ループを開始。
    for epoch in range(epoch_str, hps.train.epochs + 1):
        if rank == 0:#シングルGPUの場合を考えなければならない。
            # rank 0のプロセスは学習と評価の両方を行う。
            train_and_evaluate(rank, epoch, hps, [model, mel_fn, w2v, f0_quantizer, aug, net_v], optimizer,
                               scheduler, scaler, [train_loader, eval_loader], logger, [writer, writer_eval], n_gpus)
        else:
            # rank 0のプロセスは学習と評価の両方を行う。
            train_and_evaluate(rank, epoch, hps, [model, mel_fn, w2v, f0_quantizer, aug, net_v], optimizer,
                               scheduler, scaler, [train_loader, None], None, None, n_gpus)
        # 1エポック終了後に学習率を更新する。
        scheduler.step()

def train_and_evaluate(rank, epoch, hps, nets, optims, scheduler, scaler, loaders, logger, writers, n_gpus):
    # 引数から各オブジェクトを取り出す。
    model, mel_fn, w2v, f0_quantizer, aug, net_v = nets
    optimizer = optims
    train_loader, eval_loader = loaders

    if writers is not None:
        writer, writer_eval = writers

    global global_step
    if n_gpus > 1:
        train_loader.sampler.set_epoch(epoch)
    
    # モデルを学習モードに切り替える。
    model.train()
    # データローダーからバッチ単位でデータを取り出し、ループ処理。
    for batch_idx, (x, x_f0, length) in enumerate(train_loader):
        # データを現在のGPUに移動。non_blocking=Trueはデータ転送を非同期で行い、高速化する。
        x = x.cuda(rank, non_blocking=True)
        x_f0 = x_f0.cuda(rank, non_blocking=True)
        length = length.cuda(rank, non_blocking=True).squeeze()

        # 波形xからメルスペクトログラムを計算。
        mel_x = mel_fn(x)
        # 波形xにデータ拡張を適用。
        aug_x = aug(x)
        # データ拡張の結果がNaN（非数）になっていないかチェック。
        nan_x = torch.isnan(aug_x).any()
        # NaNでなければ拡張後の音声、NaNなら元の音声を使う。
        x = x if nan_x else aug_x
        # Wav2vec2モデルに入力するために、音声の両端にパディングを追加。
        x_pad = F.pad(x, (40, 40), "reflect")
        
        # Wav2vec2モデルを使って音声から特徴量を抽出。
        w2v_x = w2v(x_pad)
        # F0（基本周波数）を量子化するモデルを使って、音声のF0を抽出。
        f0_x = f0_quantizer.code_extraction(x_f0)

        # オプティマイザの勾配を初期化。
        optimizer.zero_grad()
        # モデルの損失を計算。
        loss_gen_all = model.module.sdt_compute_loss(model.module, scheduler,
                                 mel_x, w2v_x, f0_x, length)
        

        if hps.train.fp16_run:
            scaler.scale(loss_gen_all).backward()
            scaler.unscale_(optimizer)
            grad_norm_g = commons.clip_grad_value_(model.parameters(), None)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss_gen_all.backward()
            grad_norm_g = commons.clip_grad_value_(model.parameters(), None)
            optimizer.step()

        if rank == 0:
            if global_step % hps.train.log_interval == 0:
                lr = optimizer.param_groups[0]['lr']
                logger.info('Train Epoch: {} [{:.0f}%]'.format(
                    epoch,
                    100. * batch_idx / len(train_loader)))

                scalar_dict = {"loss/g/total": loss_gen_all, "learning_rate": lr, "grad_norm_g": grad_norm_g}

                utils.summarize(
                    writer=writer,
                    global_step=global_step,
                    scalars=scalar_dict)

            if global_step % hps.train.eval_interval == 0:
                torch.cuda.empty_cache()
                evaluate(hps, model, mel_fn, w2v, f0_quantizer, net_v, eval_loader, writer_eval)

                if global_step % hps.train.save_interval == 0:
                    utils.save_checkpoint(model, optimizer, hps.train.learning_rate, epoch,
                                          os.path.join(hps.model_dir, "G_{}.pth".format(global_step)))

        global_step += 1

    if rank == 0:
        logger.info('====> Epoch: {}'.format(epoch))



def evaluate(hps, model, mel_fn, w2v, f0_quantizer, net_v, eval_loader, writer_eval):
    model.eval()
    image_dict = {}
    audio_dict = {}
    mel_loss = 0
    enc_loss = 0
    with torch.no_grad():
        for batch_idx, (y, y_f0) in enumerate(eval_loader):
            # データをGPUに転送。
            y = y.cuda(0)
            y_f0 = y_f0.cuda(0)

            # 正解のメルスペクトログラムを計算。
            mel_y = mel_fn(y)
            # 正解のF0コードを抽出。
            f0_y = f0_quantizer.code_extraction(y_f0)
            length = torch.LongTensor([mel_y.size(2)]).cuda(0)

            y_pad = F.pad(y, (40, 40), "reflect")
            w2v_y = w2v(y_pad)
            

            scheduler = VPSDESchedule()

            mel_rec = sample(model.module, scheduler, Zsrc=f0_y,
                              Zftr=w2v_y,
                              espk=torch.randn(1,256).cuda(0),
                              steps=15)
            #mel_rec = mel_rec[:,:,:mel_y.size(2)]
            enc_output = mel_rec   # encoder 出力を流用するなら適宜

            mel_loss += F.l1_loss(mel_y, mel_rec).item()
            enc_loss += F.l1_loss(mel_y, enc_output).item()

            if batch_idx > 100:
                break
            if batch_idx <= 4:
                y_hat = net_v(mel_rec)
                enc_hat = net_v(enc_output)

                plot_mel = torch.cat([mel_y, mel_rec, enc_output], dim=1)
                plot_mel = plot_mel.clip(min=-10, max=10)

                image_dict.update({
                    "gen/mel_{}".format(batch_idx): utils.plot_spectrogram_to_numpy(plot_mel.squeeze().cpu().numpy())
                })
                audio_dict.update({
                    "gen/audio_{}".format(batch_idx): y_hat.squeeze(),
                    "gen/enc_audio_{}".format(batch_idx): enc_hat.squeeze(),
                })
                if global_step == 0:
                    audio_dict.update({"gt/audio_{}".format(batch_idx): y.squeeze()})

        mel_loss /= 100
        enc_loss /= 100

    scalar_dict = {"val/mel": mel_loss, "val/enc_mel": enc_loss}
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
