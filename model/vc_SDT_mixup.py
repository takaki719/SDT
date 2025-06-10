import math
import torch
import torch.nn as nn
from torch import Tensor
import transformers 
from model.styleencoder import StyleEncoder
from modules_sf.modules import *


class Wav2vec2(torch.nn.Module):
    def __init__(self, layer=12): 
        super().__init__() 
        self.wav2vec2 = transformers.Wav2Vec2ForPreTraining.from_pretrained("facebook/wav2vec2-xls-r-300m")
        for param in self.wav2vec2.parameters():
            param.requires_grad = False
            param.grad = None
        self.wav2vec2.eval()
        self.feature_layer = layer
        
    @torch.no_grad()
    def forward(self, x): 
        outputs = self.wav2vec2(x.squeeze(1), output_hidden_states=True)
        y = outputs.hidden_states[self.feature_layer]    
        
        return y.permute((0, 2, 1))    


class Decoder(nn.Module):
    def __init__(self,
                 in_channels,
                 hidden_channels,
                 kernel_size,
                 dilation_rate,
                 n_layers,
                 mel_size=80,
                 gin_channels=0):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.dilation_rate = dilation_rate
        self.n_layers = n_layers
        self.gin_channels = gin_channels

        self.pre = nn.Conv1d(in_channels, hidden_channels, 1)
        self.enc = WN(hidden_channels, kernel_size, dilation_rate, n_layers, gin_channels=gin_channels)
        self.proj = nn.Conv1d(hidden_channels, mel_size, 1)

    def forward(self, x, x_mask, g=None):
        x = self.pre(x * x_mask) * x_mask
        x = self.enc(x, x_mask, g=g)
        x = self.proj(x) * x_mask

        return x


class SynthesizerTrn(nn.Module):
    def __init__(self,
                 spec_channels,
                 segment_size,
                 inter_channels,
                 hidden_channels,
                 filter_channels,
                 n_heads,
                 n_layers,
                 kernel_size,
                 p_dropout,
                 resblock,
                 resblock_kernel_sizes,
                 resblock_dilation_sizes,
                 upsample_rates,
                 upsample_initial_channel,
                 upsample_kernel_sizes,
                 encoder_hidden_size,
                 **kwargs):
        super().__init__()
        
        self.spec_channels = spec_channels
        self.inter_channels = inter_channels
        self.hidden_channels = hidden_channels
        self.filter_channels = filter_channels
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.kernel_size = kernel_size
        self.p_dropout = p_dropout
        self.resblock = resblock
        self.resblock_kernel_sizes = resblock_kernel_sizes
        self.resblock_dilation_sizes = resblock_dilation_sizes
        self.upsample_rates = upsample_rates
        self.upsample_initial_channel = upsample_initial_channel
        self.upsample_kernel_sizes = upsample_kernel_sizes
        self.segment_size = segment_size

        self.emb_c = nn.Conv1d(1024, encoder_hidden_size, 1)
        self.emb_f0 = nn.Embedding(20, encoder_hidden_size)

        self.emb_g = StyleEncoder(in_dim=80, hidden_dim=256, out_dim=256)

        self.dec_f = Decoder(encoder_hidden_size, encoder_hidden_size, 5, 1, 8, mel_size=80, gin_channels=256)
        self.dec_s = Decoder(encoder_hidden_size, encoder_hidden_size, 5, 1, 8, mel_size=80, gin_channels=256) 

    def forward(self, w2v, f0_code, x_mel, length, mixup=False):
        content = self.emb_c(w2v)

        f0 = self.emb_f0(f0_code).transpose(1, 2)
        f0 = F.interpolate(f0, content.shape[-1])

        x_mask = torch.unsqueeze(commons.sequence_mask(length, x_mel.size(2)), 1).to(x_mel.dtype)
        g = self.emb_g(x_mel, x_mask).unsqueeze(-1)

        if mixup is True:
            g_mixup = torch.cat([g, g[torch.randperm(g.size()[0])]], dim=0)
            content = torch.cat([content, content], dim=0)
            f0 = torch.cat([f0, f0], dim=0)
            x_mask = torch.cat([x_mask, x_mask], dim=0)
            y_f = self.dec_f(F.relu(content), x_mask, g=g_mixup)
            y_s = self.dec_s(f0, x_mask, g=g_mixup)
        else:
            y_f = self.dec_f(F.relu(content), x_mask, g=g)
            y_s = self.dec_s(f0, x_mask, g=g)

        return g, y_s, y_f
        
    def voice_conversion(self, w2v, x_length, f0_code, x_mel, length):
        y_mask = torch.unsqueeze(commons.sequence_mask(x_length, w2v.size(2)), 1).to(w2v.dtype)

        content = self.emb_c(w2v)
        f0 = self.emb_f0(f0_code).transpose(1, 2)
        f0 = F.interpolate(f0, content.shape[-1])

        x_mask = torch.unsqueeze(commons.sequence_mask(length, x_mel.size(2)), 1).to(x_mel.dtype)
        g = self.emb_g(x_mel, x_mask).unsqueeze(-1)

        o_f = self.dec_f(F.relu(content), y_mask, g=g)
        o_s = self.dec_s(f0, y_mask, g=g)
        o = o_f + o_s

        return o, g, o_s, o_f

    
# -------------------------------------------------------------------------
# Helper
# -------------------------------------------------------------------------
def modulate(h: Tensor, shift: Tensor, scale: Tensor) -> Tensor:
    """Apply adaptive shift/scale to hidden activations (AdaLN-Zero)."""
    return h * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


# -------------------------------------------------------------------------
# Timestep embedding (sinusoidal + two-layer MLP)
# -------------------------------------------------------------------------
class TimestepEmbedder(nn.Module):
    """Embed scalar diffusion timestep `t` to a vector of size `hidden_size`."""

    def __init__(self, hidden_size: int, frequency_embedding_size: int = 256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t: Tensor, dim: int, max_period: int = 10_000) -> Tensor:
        """Create sinusoidal timestep embeddings (OpenAI GLIDE style)."""
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(half, dtype=torch.float32, device=t.device) / half
        )
        args = t[:, None].float() * freqs[None]
        emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:  # zero-pad if dim is odd
            emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
        return emb

    def forward(self, t: Tensor) -> Tensor:
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        return self.mlp(t_freq)


# -------------------------------------------------------------------------
# ViT-style Self-Attention
# -------------------------------------------------------------------------
class Attention(nn.Module):
    def __init__(self, dim: int, num_heads: int = 8, qkv_bias: bool = True):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5

        self.qkv  = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x: Tensor) -> Tensor:
        B, N, C = x.shape
        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        out  = (attn @ v).transpose(1, 2).reshape(B, N, C)
        return self.proj(out)


# -------------------------------------------------------------------------
# Feed-forward MLP
# -------------------------------------------------------------------------
class Mlp(nn.Module):
    def __init__(self, in_features: int, hidden_features: int, act_layer=nn.GELU, drop: float = 0.0):
        super().__init__()
        self.fc1  = nn.Linear(in_features,  hidden_features)
        self.act  = act_layer()
        self.fc2  = nn.Linear(hidden_features, in_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x: Tensor) -> Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


# -------------------------------------------------------------------------
# SDT Block (AdaLN-Zero)
# -------------------------------------------------------------------------
class SDTBlock(nn.Module):
    """Transformer block with AdaLN-Zero conditioning."""

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        cond_dim:  int   = 1024,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn  = Attention(hidden_size, num_heads=num_heads, qkv_bias=True)

        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden = int(hidden_size * mlp_ratio)
        self.mlp   = Mlp(
            in_features   = hidden_size,
            hidden_features = mlp_hidden,
            act_layer     = lambda: nn.GELU(approximate="tanh"),
        )

        # AdaLN-Zero
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(cond_dim, 6 * hidden_size, bias=True),
        )
        self.alpha = nn.Parameter(torch.zeros(1))  # residual scale (0-init)

    def forward(self, x: Tensor, cond: Tensor) -> Tensor:
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.adaLN_modulation(cond).chunk(6, dim=1)
        )

        # Multi-head self-attention with AdaLN-Zero
        x = x + self.alpha * gate_msa.unsqueeze(1) * self.attn(
            modulate(self.norm1(x), shift_msa, scale_msa)
        )

        # Feed-forward with AdaLN-Zero
        x = x + self.alpha * gate_mlp.unsqueeze(1) * self.mlp(
            modulate(self.norm2(x), shift_mlp, scale_mlp)
        )
        return x


# -------------------------------------------------------------------------
# Speech Diffusion Transformer backbone
# -------------------------------------------------------------------------
class SpeechDiffusionTransformer(nn.Module):
    def __init__(
        self,
        mel_size:   tuple[int, int] = (80, 128),   # (F, T)
        patch_size: tuple[int, int] = (16, 8),     # (Pf, Pt)
        embed_dim:  int   = 128,
        depth:      int   = 12,
        n_heads:    int   = 8,
        mlp_ratio:  float = 4.0,
        cond_dim:   int   = 1024,
    ):
        super().__init__()
        #ここの引数よくわからん。
        self.encoder = SynthesizerTrn(mel_size[0],
                              hps.train.segment_size // hps.data.hop_length,
                              **hps.model)
        # Timestep embedder
        self.t_embedder = TimestepEmbedder(hidden_size=embed_dim, frequency_embedding_size=256)

        F, T  = mel_size
        Pf, Pt = patch_size
        assert F % Pf == 0 and T % Pt == 0, "Patch size must divide mel dimensions"
        self.num_patches = (F // Pf) * (T // Pt)

        # Patch embed & positional encoding
        self.patch_proj = nn.Linear(Pf * Pt, embed_dim)
        self.pos_emb    = nn.Parameter(torch.randn(1, self.num_patches, embed_dim))

        # Transformer blocks
        self.blocks = nn.ModuleList(
            [SDTBlock(embed_dim, n_heads, mlp_ratio, cond_dim) for _ in range(depth)]
        )

        self.out_norm = nn.LayerNorm(embed_dim)
        self.head     = nn.Linear(embed_dim, Pf * Pt)

        # Save dims for reshape
        self.Pf, self.Pt, self.F, self.T = Pf, Pt, F, T

    # -------------------------- patch utilities --------------------------
    def _to_patches(self, x: Tensor) -> Tensor:
        """[B, F, T] → [B, N, Pf*Pt]"""
        B, F, T = x.shape
        x = x.unfold(1, self.Pf, self.Pf).unfold(2, self.Pt, self.Pt)  # (B, F//Pf, T//Pt, Pf, Pt)
        return x.contiguous().view(B, -1, self.Pf * self.Pt)

    def _from_patches(self, tokens: Tensor) -> Tensor:
        """[B, N, Pf*Pt] → [B, F, T]"""
        B, N, _ = tokens.shape
        patches = tokens.view(B, self.F // self.Pf, self.T // self.Pt, self.Pf, self.Pt)
        patches = patches.permute(0, 1, 3, 2, 4).contiguous()          # (B, F//Pf, Pf, T//Pt, Pt)
        return patches.view(B, self.F, self.T)

    # ----------------------------- forward ------------------------------
    def forward(
        self,
        mel_noisy: Tensor,
        t:         Tensor,
        Zsrc:      Tensor,
        Zftr:      Tensor,
        espk:      Tensor,
    ) -> Tensor:
        """Return predicted residual / score map of shape (B, 80, 128)."""
        t_emb   = self.t_embedder(t)
        cond_vec = torch.cat([t_emb, Zsrc, Zftr, espk], dim=-1)  # [B, cond_dim]

        # Patch Embedding
        x = self._to_patches(mel_noisy)
        # patch_embedding + positional_embedding
        x = self.patch_proj(x) + self.pos_emb

        for blk in self.blocks:
            x = blk(x, cond_vec)

        x = self.out_norm(x)
        # Predicted Noise 
        x = self.head(x)
        return self._from_patches(x)
    
    def sdt_compute_loss(model, scheduler, mel_gt, w2v, f0_code,
                     length, cfg_drop=0.2):
        """
        前向き拡散 ε 予測損失 (L_simple) だけを返す。
        mel_gt : (B, 80, T)
        """

        B = mel_gt.size(0)
        device = mel_gt.device

        # ---- 条件ベクトルを作る（自由に調整） ----
        Zsrc = w2v                                   # (B,512) など
        Zftr = torch.randn(B, 512, device=device)    # 例: 特徴プレースホルダ
        espk = torch.randn(B, 256, device=device)    # 例: 話者埋め込みプレースホルダ

        # CFG ドロップアウト
        if cfg_drop > 0:
            mask = (torch.rand(B, device=device) < cfg_drop).float().unsqueeze(1)
            Zsrc = Zsrc * (1 - mask)

        # ---- 前向き拡散 ----
        t = torch.rand(B, device=device)             # ~U(0,1)
        noise = torch.randn_like(mel_gt)
        mel_noisy = scheduler.q_sample(mel_gt, t, noise)

        # ---- ε 予測 ----
        eps_pred = model(mel_noisy, t, Zsrc, Zftr, espk)

        # ---- MSE 損失 ----
        loss_eps = F.mse_loss(eps_pred, noise)
        return loss_eps


class VPSDESchedule:
    """Continuous-time β(t) schedule: β(t) = β_min + (β_max-β_min)*t."""
    def __init__(self, beta_min: float = 0.1, beta_max: float = 20.0):
        self.beta_min = beta_min
        self.beta_max = beta_max

    # 累積 ᾱ_t = exp( - ∫ β(s) ds )
    def alpha_bar(self, t: Tensor) -> Tensor:
        return torch.exp(-0.5 * (self.beta_max - self.beta_min) * t**2
                         - 0.5 * self.beta_min * t)

    def q_sample(self, x0: Tensor, t: Tensor, noise: Tensor) -> Tensor:
        a_bar = self.alpha_bar(t).view(-1, 1, 1)
        return a_bar.sqrt() * x0 + (1.0 - a_bar).sqrt() * noise

    # 逆拡散 (Euler–Maruyama 1st-order)
    def step_euler(self,
                   x: Tensor,
                   t: Tensor,
                   eps_pred: Tensor,
                   dt: float) -> Tensor:
        beta_t = self.beta_min + (self.beta_max - self.beta_min) * t
        drift   = -0.5 * beta_t.view(-1, 1, 1) * x - beta_t.view(-1, 1, 1) * eps_pred
        diffusion = beta_t.sqrt().view(-1, 1, 1)
        z = torch.randn_like(x)
        return x + drift * dt + diffusion * math.sqrt(abs(dt)) * z