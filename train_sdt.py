import os
import torch
from torch.nn import functional as F
from torch.cuda.amp import autocast, GradScaler

from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import random
import commons
import utils

from augmentation.aug import Augment
from model.vc_sdt import (
    Wav2vec2,
    SpeechDiffusionTransformer,
    VPSDESchedule,
    SpeechDisentanglementEncoder,
    SDTVoiceConversion
)
from data_loader import AudioDataset, MelSpectrogramFixed
from hifigan.vocoder import HiFi
from torch import nn, Tensor

# CUDNNのベンチマークモードを有効にし、最適な畳み込みアルゴリズムを自動的に選択させる
torch.backends.cudnn.benchmark = True
# グローバルステップカウンターを初期化
global_step = 0

def get_param_num(model):
    """モデルのパラメータ数を計算するヘルパー関数"""
    # モデルの全パラメータの要素数を合計して返す
    num_param = sum(param.numel() for param in model.parameters())
    return num_param

@torch.no_grad() # この関数内では勾配計算を行わない
def sample(model: SDTVoiceConversion,
           source_wav: Tensor,
           source_f0: Tensor,
           target_mel: Tensor,
           source_lengths: Tensor,
           target_lengths: Tensor,
           steps: int = 6,
           guidance_mode: str = 'source'):
    """
    音声変換のサンプリング（推論）を実行する関数

    Args:
        model (SDTVoiceConversion): 音声変換モデル
        source_wav (Tensor): 変換元の音声波形
        source_f0 (Tensor): 変換元の基本周波数(F0)
        target_mel (Tensor): 変換先のメルスペクトログラム（スタイル情報として使用）
        source_lengths (Tensor): 変換元の長さ
        target_lengths (Tensor): 変換先の長さ
        steps (int): 拡散モデルのサンプリングステップ数
        guidance_mode (str): ガイダンスモード ('source' または 'target')
    
    Returns:
        Tensor: 変換されたメルスペクトログラム
    """
    # モデルを評価モードに設定
    model.eval()

    # モデルのvoice_conversionメソッドを使用して、メルスペクトログラムを生成
    mel_converted = model.voice_conversion(
        source_wav=source_wav,
        source_f0=source_f0,
        target_mel=target_mel,
        source_lengths=source_lengths,
        target_lengths=target_lengths,
        num_steps=steps,
        guidance_scale=1.0, # ガイダンスの強さ
        guidance_mode=guidance_mode
    )

    return mel_converted

def main():
    """単一GPUでの学習を実行するメイン関数"""
    # CUDAが利用可能であることを確認。利用できない場合はエラーを発生させる。
    assert torch.cuda.is_available(), "CPU training is not allowed."

    # デバイスをCUDAに設定
    device = torch.device("cuda:0")

    # 設定ファイルからハイパーパラメータを読み込む
    hps = utils.get_hparams()

    # ログ設定
    logger = utils.get_logger(hps.model_dir) # ログ出力用のロガーを取得
    logger.info(hps) # ハイパーパラメータをログに出力
    utils.check_git_hash(hps.model_dir) # Gitのハッシュ値を確認・保存
    writer = SummaryWriter(log_dir=hps.model_dir) # TensorBoard用のライター（学習ログ用）
    writer_eval = SummaryWriter(log_dir=os.path.join(hps.model_dir, "eval")) # TensorBoard用のライター（評価ログ用）

    # 乱数シードを設定して再現性を確保
    torch.manual_seed(hps.train.seed)
    torch.cuda.manual_seed(hps.train.seed)

    # メルスペクトログラム変換用のクラスを初期化
    mel_fn = MelSpectrogramFixed(
        sample_rate=hps.data.sampling_rate,
        win_length=hps.data.win_length,
        n_fft=hps.data.filter_length,
        hop_length=hps.data.hop_length,
        f_min=hps.data.mel_fmin,
        f_max=hps.data.mel_fmax,
        n_mels=hps.data.n_mel_channels,
        window_fn=torch.hann_window # 窓関数としてHann窓を使用
    ).to(device)

    # データセットとデータローダーの初期化
    train_dataset = AudioDataset(hps, training=True)
    train_loader = DataLoader(
        train_dataset,
        batch_size=hps.train.batch_size,
        num_workers=8,  # 単一GPU用にワーカー数を調整
        shuffle=True,   # データをシャッフル
        drop_last=True, # バッチサイズに満たない最後のデータを捨てる
        persistent_workers=True, # ワーカープロセスを維持して高速化
        pin_memory=True # GPU転送を高速化
    )

    test_dataset = AudioDataset(hps, training=False)
    eval_loader = DataLoader(
        test_dataset,
        batch_size=1, # 評価時は1サンプルずつ処理
        shuffle=False
    )

    # 評価用のボコーダー(HiFi-GAN)を初期化
    net_v = HiFi(
        hps.data.n_mel_channels,
        hps.train.segment_size // hps.data.hop_length,
        **hps.model
    ).to(device)

    # 学習済みHiFi-GANの重みを読み込む
    path_ckpt = './hifigan/G_2930000.pth'
    utils.load_checkpoint(path_ckpt, net_v, None)
    net_v.eval() # 評価モードに設定
    net_v.dec.remove_weight_norm() # Weight Normalizationを削除（推論時）

    # コンテンツエンコーダー (Wav2vec2/MMS) を初期化
    content_encoder = Wav2vec2().to(device)

    # スタイルエンコーダーを初期化
    from model.styleencoder import StyleEncoder
    style_encoder = nn.Module()
    # 話者埋め込みを生成するスタイルエンコーダー
    style_encoder.emb_g = StyleEncoder(in_dim=80, hidden_dim=256, out_dim=256).to(device)

    # データ拡張用のクラスを初期化
    aug = Augment(hps).to(device)

    # メインの音声変換モデル (SDTVoiceConversion) を初期化
    model = SDTVoiceConversion(
        content_encoder=content_encoder,
        style_encoder=style_encoder,
        hidden_channels=256, # 隠れ層のチャンネル数
        gin_channels=256,    # 話者埋め込みのチャンネル数
        mel_size=(80, 128),  # メルスペクトログラムのサイズ (n_mels, frames)
        patch_size=(16, 8),  # Transformerのパッチサイズ
        embed_dim=768,       # 埋め込み次元数
        depth=12,            # Transformerの層数
        n_heads=12,          # アテンションヘッド数
        mlp_ratio=4.0,       # MLP層の拡大率
        beta_min=0.05,       # 拡散プロセスのβの最小値
        beta_max=20.0,       # 拡散プロセスのβの最大値
        cfg_drop=0.2         # Classifier-Free Guidanceのドロップアウト率
    ).to(device)

    # モデルのパラメータ数をログに出力
    logger.info(f"Model parameters: {get_param_num(model):,}")

    # 最適化アルゴリズム (AdamW) を初期化
    optimizer = torch.optim.AdamW(
        model.parameters(),
        hps.train.learning_rate,
        betas=hps.train.betas,
        eps=hps.train.eps
    )

    # チェックポイントが存在すれば読み込み、学習を再開する
    try:
        _, _, _, epoch_str = utils.load_checkpoint(
            utils.latest_checkpoint_path(hps.model_dir, "G_*.pth"),
            model,
            optimizer
        )
        # グローバルステップを復元
        global_step = (epoch_str - 1) * len(train_loader)
    except:
        # チェックポイントがない場合は最初から学習を開始
        epoch_str = 1
        global_step = 0

    # 混合精度学習(FP16)用のGradScalerを初期化
    scaler = GradScaler(enabled=hps.train.fp16_run)

    # 学習ループを開始
    for epoch in range(epoch_str, hps.train.epochs + 1):
        train_and_evaluate(
            epoch, hps,
            [model, mel_fn, aug, net_v], # モデル群
            optimizer, scaler,
            [train_loader, eval_loader], # データローダー群
            logger,
            [writer, writer_eval], # TensorBoardライター群
            device
        )

def train_and_evaluate(epoch, hps, nets, optimizer, scaler, loaders, logger, writers, device):
    """1エポック分の学習と評価を実行する関数"""
    model, mel_fn, aug, net_v = nets
    train_loader, eval_loader = loaders
    writer, writer_eval = writers

    global global_step # グローバル変数のglobal_stepを使用

    model.train() # モデルを学習モードに設定

    # 学習データローダーからバッチ単位でデータを取得
    for batch_idx, (x, x_f0, length) in enumerate(train_loader):
        # データをGPUに転送
        x = x.to(device, non_blocking=True)
        x_f0 = x_f0.to(device, non_blocking=True)
        length = length.to(device, non_blocking=True).squeeze()

        # 音声波形からメルスペクトログラムを計算
        mel_x = mel_fn(x)

        # データ拡張を適用
        aug_x = aug(x)
        # 拡張後にNaNが発生した場合は元のデータを使用
        nan_x = torch.isnan(aug_x).any()
        x_aug = x if nan_x else aug_x

        # wav2vec2の入力のためにオーディオをパディング
        x_pad = F.pad(x_aug, (40, 40), "reflect")

        # 勾配をリセット
        optimizer.zero_grad()

        # 混合精度学習(FP16)を有効化
        with autocast(enabled=hps.train.fp16_run):
            # モデルの損失を計算
            loss_dict = model.compute_loss(
                mel_gt=mel_x,     # 正解メルスペクトログラム
                wav_input=x_pad,  # パディングされた入力波形
                f0=x_f0,          # 基本周波数
                lengths=length    # 長さ
            )
            loss_total = loss_dict['loss'] # 合計損失

        # 勾配計算とパラメータ更新
        if hps.train.fp16_run:
            # 混合精度の場合
            scaler.scale(loss_total).backward() # 損失をスケーリングして逆伝播
            scaler.unscale_(optimizer) # 勾配のスケールを元に戻す
            grad_norm = commons.clip_grad_value_(model.parameters(), None) # 勾配クリッピング
            scaler.step(optimizer) # オプティマイザでパラメータを更新
            scaler.update() # スケーラーを更新
        else:
            # FP32の場合
            loss_total.backward() # 逆伝播
            grad_norm = commons.clip_grad_value_(model.parameters(), None) # 勾配クリッピング
            optimizer.step() # パラメータを更新

        # ログ出力
        if global_step % hps.train.log_interval == 0:
            lr = optimizer.param_groups[0]['lr'] # 現在の学習率を取得
            logger.info(
                f'Train Epoch: {epoch} [{100. * batch_idx / len(train_loader):.0f}%] '
                f'Loss: {loss_total.item():.4f}'
            )

            # TensorBoardに書き込むスカラ値を辞書にまとめる
            scalar_dict = {
                "loss/total": loss_total.item(),
                "loss/diff": loss_dict['loss_diff'].item(), # 拡散損失
                "loss/recon": loss_dict['loss_recon'].item(), # 再構成損失
                "learning_rate": lr,
                "grad_norm": grad_norm # 勾配のノルム
            }

            # TensorBoardにログを書き込む
            utils.summarize(
                writer=writer,
                global_step=global_step,
                scalars=scalar_dict
            )

        # 評価インターバルごとに評価を実行
        if global_step % hps.train.eval_interval == 0:
            torch.cuda.empty_cache() # GPUキャッシュをクリア
            evaluate(hps, model, mel_fn, net_v, eval_loader, writer_eval, device)

            # 保存インターバルごとにチェックポイントを保存
            if global_step % hps.train.save_interval == 0:
                utils.save_checkpoint(
                    model, optimizer, hps.train.learning_rate, epoch,
                    os.path.join(hps.model_dir, f"G_{global_step}.pth")
                )
        
        # グローバルステップをインクリメント
        global_step += 1

    logger.info(f'====> Epoch: {epoch}')


def evaluate(hps, model, mel_fn, net_v, eval_loader, writer_eval, device):
    """評価データセットでモデルを評価する関数"""
    model.eval() # モデルを評価モードに設定
    image_dict = {} # TensorBoardに表示する画像（スペクトログラム）を格納
    audio_dict = {} # TensorBoardで再生する音声を格納
    mel_loss_total = 0 # メルスペクトログラムのL1損失の合計

    with torch.no_grad(): # 勾配計算を無効化
        # 評価データローダーからデータを取得
        for batch_idx, (y, y_f0, y_length) in enumerate(eval_loader):
            if batch_idx > 10:  # 最初の10サンプルのみ評価して時間を節約
                break

            # データをGPUに転送
            y = y.to(device)
            y_f0 = y_f0.to(device)
            y_length = y_length.to(device)

            # 正解のメルスペクトログラムを計算
            mel_y = mel_fn(y)

            # wav2vec2のためにオーディオをパディング
            y_pad = F.pad(y, (40, 40), "reflect")

            # 音声変換を実行（評価時は自己再合成）
            mel_converted = sample(
                model,
                source_wav=y_pad,   # 変換元音声
                source_f0=y_f0,     # 変換元F0
                target_mel=mel_y,   # 変換先メル（スタイル情報として同じものを入力）
                source_lengths=y_length,
                target_lengths=y_length,
                steps=6,            # サンプリングステップ数
                guidance_mode='source'
            )

            # 生成されたメルスペクトログラムをターゲットの長さにクロッピング
            mel_converted = mel_converted[:, :, :mel_y.size(2)]

            # L1損失を計算
            mel_loss = F.l1_loss(mel_y, mel_converted).item()
            mel_loss_total += mel_loss

            # 最初の4サンプルについて、音声と画像を生成してログに記録
            if batch_idx < 4:
                # HiFi-GANボコーダーでメルスペクトログラムから音声を生成
                y_hat = net_v(mel_converted)

                # 正解と生成されたメルスペクトログラムを結合してプロット
                plot_mel = torch.cat([mel_y, mel_converted], dim=1)
                plot_mel = plot_mel.clip(min=-10, max=10) # 表示範囲をクリップ

                image_dict.update({
                    f"gen/mel_{batch_idx}": utils.plot_spectrogram_to_numpy(
                        plot_mel.squeeze().cpu().numpy()
                    )
                })

                # 生成された音声を辞書に追加
                audio_dict.update({
                    f"gen/audio_{batch_idx}": y_hat.squeeze(),
                })
                
                # 最初の評価時のみ、正解音声も辞書に追加
                if global_step == 0:
                    audio_dict.update({
                        f"gt/audio_{batch_idx}": y.squeeze()
                    })

    # メル損失の平均を計算
    mel_loss_avg = mel_loss_total / min(10, len(eval_loader))

    # TensorBoard用のスカラ値を辞書にまとめる
    scalar_dict = {"val/mel_loss": mel_loss_avg}

    # TensorBoardに評価結果（スカラ、画像、音声）を書き込む
    utils.summarize(
        writer=writer_eval,
        global_step=global_step,
        images=image_dict,
        audios=audio_dict,
        audio_sampling_rate=hps.data.sampling_rate,
        scalars=scalar_dict
    )

    model.train() # モデルを学習モードに戻す


# このスクリプトが直接実行された場合にmain()関数を呼び出す
if __name__ == "__main__":
    main()