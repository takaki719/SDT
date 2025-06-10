import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import transformers 
# このプロジェクトで定義されたスタイルエンコーダーやカスタムモジュール
from model.styleencoder import StyleEncoder
from modules_sf.modules import *


# -------------------------------------------------------------------------
# Speech Disentanglement Encoder Components (音声分離エンコーダーのコンポーネント群)
# -------------------------------------------------------------------------
class Wav2vec2(torch.nn.Module):
    """MMSモデルを使って、音声から言語的な特徴表現を抽出するクラス"""
    def __init__(self, layer=7):  # 論文では7層目を使用
        super().__init__() 
        # MMS（大規模多言語音声）モデルを使用
        self.wav2vec2 = transformers.Wav2Vec2ForPreTraining.from_pretrained("facebook/mms-300m")
        # MMSモデルの重みは学習中に更新しない（特徴抽出器として固定）
        for param in self.wav2vec2.parameters():
            param.requires_grad = False
            param.grad = None
        # モデルを評価モードに設定
        self.wav2vec2.eval()
        # 特徴量として抽出する層を指定
        self.feature_layer = layer
        
    @torch.no_grad() # このメソッド内では勾配計算を行わない
    def forward(self, x): 
        # 入力波形xをMMSモデルに通し、すべての中間層の出力を得る
        outputs = self.wav2vec2(x.squeeze(1), output_hidden_states=True)
        # 指定した層の隠れ状態（特徴量）を抽出
        y = outputs.hidden_states[self.feature_layer]    
        # (バッチ, 次元, 系列長) の形式に並び替えて返す
        return y.permute((0, 2, 1))    


class PitchEncoder(nn.Module):
    """F0（基本周波数）を処理するためのピッチエンコーダー"""
    def __init__(self, in_channels=1, out_channels=256):
        super().__init__()
        # 1x1の畳み込み層でチャンネル数を変換
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=1)
        
    def forward(self, f0):
        # f0: [B, 1, T] ここでのTはメルスペクトログラムの時間長の4倍
        # 対数変換を適用。知覚的なスケールに近づける
        f0_log = torch.log(f0 + 1.0)
        return self.conv(f0_log)


class SourceEncoder(nn.Module):
    """WaveNetアーキテクチャを使用した音源(Source)エンコーダー"""
    def __init__(self, 
                 pitch_channels=256,
                 hidden_channels=256,
                 kernel_size=3,
                 dilation_rate=1,
                 n_layers=8,
                 gin_channels=256): # gin_channelsは話者埋め込み用のチャネル数
        super().__init__()
        self.pitch_encoder = PitchEncoder(1, pitch_channels)
        self.wavenet = WN(hidden_channels, kernel_size, dilation_rate, n_layers, 
                         gin_channels=gin_channels)
        self.proj = nn.Conv1d(hidden_channels, hidden_channels, 1)
        
    def forward(self, f0, espk, x_mask=None):
        # f0: F0系列, espk: 話者埋め込み, x_mask: マスク
        
        # ピッチ(F0)をエンコード
        e_f0 = self.pitch_encoder(f0)
        
        # ピッチ特徴量の時間解像度をコンテンツ特徴量に合わせるためにダウンサンプリング
        e_f0 = F.avg_pool1d(e_f0, kernel_size=4, stride=4)
        
        # WaveNetに話者情報(espk)を条件として入力し、音源特徴量z_srcを生成
        if x_mask is not None:
            z_src = self.wavenet(e_f0 * x_mask, x_mask, g=espk)
            z_src = self.proj(z_src) * x_mask
        else:
            z_src = self.wavenet(e_f0, x_mask, g=espk)
            z_src = self.proj(z_src)
            
        return z_src


class FilterEncoder(nn.Module):
    """WaveNetアーキテクチャを使用したフィルター(Filter)エンコーダー"""
    def __init__(self,
                 content_channels=1024, # Wav2vec2から出力される特徴量の次元数
                 hidden_channels=256,
                 kernel_size=3,
                 dilation_rate=1,
                 n_layers=8,
                 gin_channels=256):
        super().__init__()
        self.content_proj = nn.Conv1d(content_channels, hidden_channels, 1)
        self.wavenet = WN(hidden_channels, kernel_size, dilation_rate, n_layers,
                         gin_channels=gin_channels)
        self.proj = nn.Conv1d(hidden_channels, hidden_channels, 1)
        
    def forward(self, w2v, espk, x_mask=None):
        # w2v: コンテンツ特徴量, espk: 話者埋め込み, x_mask: マスク
        
        # コンテンツ特徴量を射影（次元削減）
        e_w2v = self.content_proj(w2v)
        
        # WaveNetに話者情報(espk)を条件として入力し、フィルター特徴量z_ftrを生成
        if x_mask is not None:
            z_ftr = self.wavenet(e_w2v * x_mask, x_mask, g=espk)
            z_ftr = self.proj(z_ftr) * x_mask
        else:
            z_ftr = self.wavenet(e_w2v, x_mask, g=espk)
            z_ftr = self.proj(z_ftr)
            
        return z_ftr


class SpeechDisentanglementEncoder(nn.Module):
    """音声分離エンコーダー全体をまとめるクラス"""
    def __init__(self,
                 content_encoder,  # Wav2vec2のインスタンス
                 style_encoder,    # 話者情報を抽出するエンコーダー
                 hidden_channels=256,
                 gin_channels=256):
        super().__init__()
        self.content_encoder = content_encoder
        self.style_encoder = style_encoder
        
        # 音源エンコーダーを初期化
        self.source_encoder = SourceEncoder(
            pitch_channels=256,
            hidden_channels=hidden_channels,
            gin_channels=gin_channels
        )
        # フィルターエンコーダーを初期化
        self.filter_encoder = FilterEncoder(
            content_channels=1024,
            hidden_channels=hidden_channels,
            gin_channels=gin_channels
        )
        
    def forward(self, wav_input, f0, mel_input, lengths):
        # 音声からコンテンツ（言語）表現を抽出
        w2v = self.content_encoder(wav_input)
        
        # メルスペクトログラムから話者表現を抽出
        x_mask = torch.unsqueeze(commons.sequence_mask(lengths, mel_input.size(2)), 1).to(mel_input.dtype)
        espk = self.style_encoder.emb_g(mel_input, x_mask).unsqueeze(-1)
        
        # 音源(source)とフィルター(filter)の特徴量をエンコード
        z_src = self.source_encoder(f0, espk, x_mask)
        z_ftr = self.filter_encoder(w2v, espk, x_mask)
        
        # 各分離された特徴量を返す
        return z_src, z_ftr, espk, w2v


# -------------------------------------------------------------------------
# Speech Diffusion Transformer (音声拡散トランスフォーマー)
# -------------------------------------------------------------------------
class TimestepEmbedder(nn.Module):
    """拡散のタイムステップ（スカラー値）をベクトルに埋め込むクラス"""
    def __init__(self, hidden_size: int, frequency_embedding_size: int = 256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(), # SiLU (Swish) 活性化関数
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t: Tensor, dim: int, max_period: int = 10_000) -> Tensor:
        """Sinusoidal Positional Encodingと同様の方法でタイムステップ埋め込みを生成"""
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(half, dtype=torch.float32, device=t.device) / half
        )
        args = t[:, None].float() * freqs[None]
        emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2: # 次元が奇数の場合
            emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
        return emb

    def forward(self, t: Tensor) -> Tensor:
        # タイムステップtをSinusoidal embeddingに変換
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        # MLPを通して最終的な埋め込みベクトルを生成
        return self.mlp(t_freq)


class Attention(nn.Module):
    """標準的なマルチヘッド・セルフアテンション"""
    def __init__(self, dim: int, num_heads: int = 8, qkv_bias: bool = True):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5 # スケーリングファクター

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias) # Q, K, Vを一度に計算
        self.proj = nn.Linear(dim, dim) # 出力射影層

    def forward(self, x: Tensor) -> Tensor:
        B, N, C = x.shape
        # Q, K, Vを計算し、マルチヘッド用に分割
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Attentionスコアを計算 (scaled dot-product)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        
        # Attentionを適用し、次元を元に戻す
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        return self.proj(x)


class Mlp(nn.Module):
    """フィードフォワードネットワーク (MLP)"""
    def __init__(self, in_features: int, hidden_features: int, act_layer=nn.GELU, drop: float = 0.0):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, in_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x: Tensor) -> Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


def modulate(h: Tensor, shift: Tensor, scale: Tensor) -> Tensor:
    """AdaLN (Adaptive Layer Normalization) の変調を適用する関数"""
    # 外部からの条件(shift, scale)を使って、入力hをアフィン変換する
    return h * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class SDTBlock(nn.Module):
    """AdaLN-Zeroによる条件付けを組み込んだTransformerブロック"""
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        cond_dim: int = 1024, # 条件ベクトルの次元数
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(hidden_size, num_heads=num_heads, qkv_bias=True)
        
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden = int(hidden_size * mlp_ratio)
        self.mlp = Mlp(
            in_features=hidden_size,
            hidden_features=mlp_hidden,
            act_layer=lambda: nn.GELU(approximate="tanh"),
        )
        
        # AdaLN-Zero変調層: 条件ベクトルからshift, scale, gateを生成する
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(cond_dim, 6 * hidden_size, bias=True),
        )
        # 出力層の重みとバイアスをゼロで初期化（AdaLN-Zeroの重要な部分）
        nn.init.zeros_(self.adaLN_modulation[-1].weight)
        nn.init.zeros_(self.adaLN_modulation[-1].bias)
        
        # ゲートの強さを調整する学習可能なパラメータ
        self.alpha = nn.Parameter(torch.zeros(1))

    def forward(self, x: Tensor, cond: Tensor) -> Tensor:
        # 条件ベクトルcondから、MSAとMLPのためのshift, scale, gateを生成
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.adaLN_modulation(cond).chunk(6, dim=1)
        )
        
        # Self-Attention + AdaLN-Zero + 残差接続
        x = x + self.alpha * gate_msa.unsqueeze(1) * self.attn(
            modulate(self.norm1(x), shift_msa, scale_msa)
        )
        
        # MLP + AdaLN-Zero + 残差接続
        x = x + self.alpha * gate_mlp.unsqueeze(1) * self.mlp(
            modulate(self.norm2(x), shift_mlp, scale_mlp)
        )
        
        return x


class SpeechDiffusionTransformer(nn.Module):
    """音声変換のためのSpeech Diffusion Transformer本体"""
    def __init__(
        self,
        mel_size: tuple[int, int] = (80, 128),  # (周波数次元, 時間次元)
        patch_size: tuple[int, int] = (16, 8),  # パッチサイズ
        embed_dim: int = 768,
        depth: int = 12,
        n_heads: int = 12,
        mlp_ratio: float = 4.0,
        cond_dim: int = 1024,
    ):
        super().__init__()
        
        # タイムステップ埋め込み層
        self.t_embedder = TimestepEmbedder(hidden_size=embed_dim)
        
        # パッチ関連の次元を計算
        F, T = mel_size
        Pf, Pt = patch_size
        assert F % Pf == 0 and T % Pt == 0, "パッチサイズはメル次元を割り切れる必要があります"
        self.num_patches = (F // Pf) * (T // Pt)
        
        # パッチ埋め込み層
        self.patch_proj = nn.Linear(Pf * Pt, embed_dim)
        # 学習可能な位置埋め込み
        self.pos_emb = nn.Parameter(torch.randn(1, self.num_patches, embed_dim) * 0.02)
        
        # Transformerブロックを重ねる
        self.blocks = nn.ModuleList([
            SDTBlock(embed_dim, n_heads, mlp_ratio, cond_dim) 
            for _ in range(depth)
        ])
        
        # 出力層
        self.out_norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, Pf * Pt)
        
        # 次元情報を保存
        self.Pf, self.Pt, self.F, self.T = Pf, Pt, F, T
        
    def _to_patches(self, x: Tensor) -> Tensor:
        """メルスペクトログラムをパッチに変換: [B, F, T] → [B, N, Pf*Pt]"""
        B, F, T = x.shape
        x = x.unfold(1, self.Pf, self.Pf).unfold(2, self.Pt, self.Pt)
        return x.contiguous().view(B, -1, self.Pf * self.Pt)
    
    def _from_patches(self, tokens: Tensor) -> Tensor:
        """パッチをメルスペクトログラムに戻す: [B, N, Pf*Pt] → [B, F, T]"""
        B, N, _ = tokens.shape
        patches = tokens.view(B, self.F // self.Pf, self.T // self.Pt, self.Pf, self.Pt)
        patches = patches.permute(0, 1, 3, 2, 4).contiguous()
        return patches.view(B, self.F, self.T)
    
    def forward(
        self,
        mel_noisy: Tensor, # ノイズが付加されたメル
        t: Tensor,         # タイムステップ
        z_src: Tensor,     # 音源埋め込み
        z_ftr: Tensor,     # フィルター埋め込み
        espk: Tensor,      # 話者埋め込み
    ) -> Tensor:
        """
        SDTのフォワードパス
        Returns:
            予測されたノイズ [B, 80, 128]
        """
        # タイムステップを埋め込む
        t_emb = self.t_embedder(t)
        
        # 全ての条件ベクトルを結合する
        cond_vec = torch.cat([t_emb, z_src, z_ftr, espk], dim=-1)  # [B, cond_dim]
        
        # メルをパッチに変換し、位置埋め込みを足す
        x = self._to_patches(mel_noisy)
        x = self.patch_proj(x) + self.pos_emb
        
        # Transformerブロックを順に通す
        for blk in self.blocks:
            x = blk(x, cond_vec)
        
        # 出力層を通して、パッチ単位のノイズを予測
        x = self.out_norm(x)
        x = self.head(x)
        
        # パッチをメルスペクトログラム形式に戻して返す
        return self._from_patches(x)


# -------------------------------------------------------------------------
# Diffusion Scheduler (拡散スケジューラー)
# -------------------------------------------------------------------------
class VPSDESchedule:
    """VP-SDEに基づく連続的なβ(t)スケジュール"""
    def __init__(self, beta_min: float = 0.05, beta_max: float = 20.0):
        self.beta_min = beta_min
        self.beta_max = beta_max
    
    def beta(self, t: Tensor) -> Tensor:
        """βスケジュール: β(t) = β_min + (β_max - β_min) * t"""
        return self.beta_min + (self.beta_max - self.beta_min) * t
    
    def alpha_bar(self, t: Tensor) -> Tensor:
        """累積積: ᾱ(t) = exp(-∫β(s)ds)"""
        return torch.exp(-0.5 * (self.beta_max - self.beta_min) * t**2 - self.beta_min * t)
    
    def q_sample(self, x0: Tensor, t: Tensor, noise: Tensor) -> Tensor:
        """順方向の拡散プロセス: q(xt|x0) = N(xt; √ᾱt x0, (1-ᾱt)I)"""
        a_bar = self.alpha_bar(t).view(-1, 1, 1)
        return a_bar.sqrt() * x0 + (1.0 - a_bar).sqrt() * noise
    
    def predict_x0_from_eps(self, xt: Tensor, t: Tensor, eps: Tensor) -> Tensor:
        """xtと予測ノイズepsから元のデータx0を予測"""
        a_bar = self.alpha_bar(t).view(-1, 1, 1)
        return (xt - (1.0 - a_bar).sqrt() * eps) / a_bar.sqrt()
    
    def step_euler(self, x: Tensor, t: Tensor, eps_pred: Tensor, dt: float) -> Tensor:
        """逆方向の拡散プロセスのためのオイラー・丸山法による1ステップ更新"""
        beta_t = self.beta(t)
        drift = -0.5 * beta_t.view(-1, 1, 1) * x - beta_t.view(-1, 1, 1) * eps_pred
        diffusion = beta_t.sqrt().view(-1, 1, 1)
        z = torch.randn_like(x) if t.min() > 0 else torch.zeros_like(x)
        return x + drift * dt + diffusion * math.sqrt(abs(dt)) * z


# -------------------------------------------------------------------------
# Complete Model with Loss (損失計算を含む完全なモデル)
# -------------------------------------------------------------------------
class SDTVoiceConversion(nn.Module):
    """音声変換のための完全なSDTモデル"""
    def __init__(
        self,
        content_encoder,
        style_encoder,
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
        cfg_drop=0.2, # Classifier-Free Guidanceのためのドロップアウト率
    ):
        super().__init__()
        
        # 音声分離エンコーダー
        self.encoder = SpeechDisentanglementEncoder(
            content_encoder=content_encoder,
            style_encoder=style_encoder,
            hidden_channels=hidden_channels,
            gin_channels=gin_channels
        )
        
        # 条件ベクトルの合計次元数を計算
        cond_dim = embed_dim + hidden_channels * 2 + gin_channels
        
        # 音声拡散トランスフォーマー
        self.diffusion = SpeechDiffusionTransformer(
            mel_size=mel_size,
            patch_size=patch_size,
            embed_dim=embed_dim,
            depth=depth,
            n_heads=n_heads,
            mlp_ratio=mlp_ratio,
            cond_dim=cond_dim
        )
        
        # 拡散スケジューラー
        self.schedule = VPSDESchedule(beta_min=beta_min, beta_max=beta_max)
        self.cfg_drop = cfg_drop
        
    def compute_loss(self, mel_gt, wav_input, f0, lengths):
        """
        拡散損失を計算する（オプションでCFGドロップアウト付き）
        Args:
            mel_gt: 正解のメルスペクトログラム
            wav_input: コンテンツ抽出用の入力波形
            f0: 基本周波数
            lengths: 系列長
        """
        B = mel_gt.size(0)
        device = mel_gt.device
        
        # 音声コンポーネントをエンコード
        z_src, z_ftr, espk, w2v = self.encoder(wav_input, f0, mel_gt, lengths)
        
        # 時間次元で平均をとり、条件ベクトルを作成
        z_src_cond = z_src.mean(dim=2)  # [B, 256]
        z_ftr_cond = z_ftr.mean(dim=2)  # [B, 256]
        espk_cond = espk.squeeze(-1)    # [B, 256]
        
        # Classifier-Free Guidanceのためのドロップアウト
        # 学習中に一定確率で音源（ピッチ）の条件をゼロにする
        if self.training and self.cfg_drop > 0:
            mask = (torch.rand(B, device=device) < self.cfg_drop).float().unsqueeze(1)
            z_src_cond = z_src_cond * (1 - mask)
        
        # タイムステップとノイズをランダムにサンプリング
        t = torch.rand(B, device=device)
        noise = torch.randn_like(mel_gt)
        
        # 順方向の拡散プロセスでノイズ付きメルを生成
        mel_noisy = self.schedule.q_sample(mel_gt, t, noise)
        
        # 拡散モデルでノイズを予測
        eps_pred = self.diffusion(mel_noisy, t, z_src_cond, z_ftr_cond, espk_cond)
        
        # ノイズ予測の誤差（MSE損失）を計算
        loss_diff = F.mse_loss(eps_pred, noise)
        
        # エンコーダーの再構成損失
        # 分離した音源とフィルターを足し合わせ、元のメルに近づくように学習
        mel_recon = z_src + z_ftr
        loss_recon = F.l1_loss(mel_recon, mel_gt)
        
        return {
            'loss': loss_diff + loss_recon, # 合計損失
            'loss_diff': loss_diff,
            'loss_recon': loss_recon
        }
    
    @torch.no_grad()
    def voice_conversion(
        self,
        source_wav,
        source_f0,
        target_mel,
        source_lengths,
        target_lengths,
        num_steps=6,
        guidance_scale=1.0,
        guidance_mode='source'  # 'source', 'null', 'dual'のいずれか
    ):
        """
        音声変換を実行する
        Args:
            guidance_scale: ガイダンスの強さ
            guidance_mode: ピッチ制御のためのガイダンスモード
        """
        device = source_wav.device
        B = source_wav.size(0)
        
        # ソース音声から音源(z_src)とフィルター(z_ftr)の特徴量をエンコード
        z_src, z_ftr, _, _ = self.encoder(source_wav, source_f0, target_mel, source_lengths)
        
        # ターゲットのメルから話者埋め込み(espk)を抽出
        x_mask = torch.unsqueeze(commons.sequence_mask(target_lengths, target_mel.size(2)), 1)
        espk = self.encoder.style_encoder.emb_g(target_mel, x_mask).unsqueeze(-1)
        
        # 条件ベクトルを準備
        z_src_cond = z_src.mean(dim=2)
        z_ftr_cond = z_ftr.mean(dim=2)
        espk_cond = espk.squeeze(-1)
        
        # 純粋なノイズから開始
        mel = torch.randn(B, 80, 128, device=device)
        
        # 逆方向の拡散プロセス（サンプリング）
        dt = 1.0 / num_steps
        for i in range(num_steps):
            t = torch.ones(B, device=device) * (1.0 - i * dt)
            
            if guidance_mode == 'source':
                # 通常通り、ソースのピッチ情報を条件としてノイズを予測
                eps_pred = self.diffusion(mel, t, z_src_cond, z_ftr_cond, espk_cond)
            elif guidance_mode == 'null':
                # ピッチ情報をゼロにして、無条件でノイズを予測
                z_src_null = torch.zeros_like(z_src_cond)
                eps_pred = self.diffusion(mel, t, z_src_null, z_ftr_cond, espk_cond)
            elif guidance_mode == 'dual':
                # デュアルガイダンス（Classifier-Free Guidanceと同様）
                # 条件付き予測
                eps_cond = self.diffusion(mel, t, z_src_cond, z_ftr_cond, espk_cond)
                # 無条件予測
                z_src_null = torch.zeros_like(z_src_cond)
                eps_uncond = self.diffusion(mel, t, z_src_null, z_ftr_cond, espk_cond)
                # 2つの予測を線形補間して、ガイダンスを適用
                eps_pred = guidance_scale * eps_cond + (1 - guidance_scale) * eps_uncond
            
            # 1ステップ分のデノイジングを実行
            mel = self.schedule.step_euler(mel, t, eps_pred, dt)
        
        return mel