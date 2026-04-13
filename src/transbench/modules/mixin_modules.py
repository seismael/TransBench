import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import Tensor
from einops import repeat

# Compatibility shim for some third-party libraries that expect `torch.<backend>.device`.
# Newer PyTorch versions expose `torch.cpu` as a module without a `device` attribute,
# and some libraries call it with `None`.
try:
    if hasattr(torch, "cpu") and not hasattr(torch.cpu, "device"):
        def _cpu_device(device: object | None = None) -> torch.device:  # type: ignore[name-defined]
            if device is None:
                return torch.device("cpu")
            return torch.device(device)  # type: ignore[arg-type]

        setattr(torch.cpu, "device", _cpu_device)
except Exception:
    pass

MultiScaleRetention = None  # type: ignore
Mamba2 = None  # type: ignore
RWKV6Attention = None  # type: ignore


class _CausalDepthwiseConv1d(nn.Module):
    def __init__(self, hidden_size: int, *, kernel_size: int = 3):
        super().__init__()
        if kernel_size <= 0:
            raise ValueError("kernel_size must be > 0")
        self.kernel_size = int(kernel_size)
        self.conv = nn.Conv1d(hidden_size, hidden_size, kernel_size=self.kernel_size, groups=hidden_size, bias=True)

    def forward(self, x: Tensor) -> Tensor:
        # x: (B, S, H) -> (B, H, S)
        x_t = x.transpose(1, 2)
        x_t = F.pad(x_t, (self.kernel_size - 1, 0))
        y = self.conv(x_t)
        return y.transpose(1, 2)


class _FallbackSequenceMixer(nn.Module):
    def __init__(self, hidden_size: int, *, kernel_size: int = 3):
        super().__init__()
        self.in_proj = nn.Linear(hidden_size, 2 * hidden_size)
        self.conv = _CausalDepthwiseConv1d(hidden_size, kernel_size=kernel_size)
        self.out_proj = nn.Linear(hidden_size, hidden_size)

    def forward(self, x: Tensor) -> Tensor:
        u, v = self.in_proj(x).chunk(2, dim=-1)
        v = self.conv(v)
        y = torch.sigmoid(u) * v
        return self.out_proj(y)


class _FallbackRWKVMixer(nn.Module):
    def __init__(self, hidden_size: int, *, kernel_size: int = 3):
        super().__init__()
        self.mix = nn.Linear(hidden_size, hidden_size)
        self.conv = _CausalDepthwiseConv1d(hidden_size, kernel_size=kernel_size)
        self.out_proj = nn.Linear(hidden_size, hidden_size)

    def forward(self, x: Tensor) -> Tensor:
        x_prev = torch.zeros_like(x)
        if x.size(1) > 1:
            x_prev[:, 1:, :] = x[:, :-1, :]
        gate = torch.sigmoid(self.mix(x))
        mixed = gate * x + (1.0 - gate) * x_prev
        y = self.conv(mixed)
        return self.out_proj(y)


def _require_fla() -> tuple[object, object, object]:
    """Import FLA lazily.

    Importing `flash-linear-attention` can emit warnings and/or do backend checks.
    We only want that overhead when actually benchmarking FLA-based architectures.
    """

    global MultiScaleRetention, Mamba2, RWKV6Attention
    if MultiScaleRetention is None or Mamba2 is None or RWKV6Attention is None:
        try:
            from fla.layers import MultiScaleRetention as _MSR, Mamba2 as _M2, RWKV6Attention as _RW  # type: ignore

            MultiScaleRetention = _MSR  # type: ignore
            Mamba2 = _M2  # type: ignore
            RWKV6Attention = _RW  # type: ignore
        except Exception as e:
            message = (
                "This mixin requires the optional dependency 'fla'. "
                "Install it with: uv pip install -e '.[fla]' (or pip install flash-linear-attention). "
                "\n\nNote: flash-linear-attention may fail to import on some setups (e.g. Triton not supported on your GPU, "
                "or Torch < 2.4 falling back to CPU). Underlying error was: "
                f"{type(e).__name__}: {e}"
            )
            raise ImportError(message) from e
    return MultiScaleRetention, Mamba2, RWKV6Attention

from transbench.modules.archi_modules import RMSNorm

class MultiScaleRetentionMixin(nn.Module):
    def __init__(self, hidden_size, num_attention_heads, num_key_value_heads=None, *, use_fla: bool = True):
        super().__init__()

        self._use_fla = bool(use_fla)
        
        if num_key_value_heads is None:
            num_key_value_heads = num_attention_heads

        self.msr = None
        self.fallback = None
        if self._use_fla:
            try:
                (MultiScaleRetention_cls, _, _) = _require_fla()
                self.msr = MultiScaleRetention_cls(
                    hidden_size=hidden_size,
                    num_heads=num_attention_heads,
                    num_kv_heads=num_key_value_heads,
                    expand_k=1.0,
                    expand_v=1.0,
                )
            except Exception:
                self.msr = None

        if self.msr is None:
            # Pure PyTorch fallback; not the same kernel but keeps the benchmark runnable.
            self.fallback = _FallbackSequenceMixer(int(hidden_size), kernel_size=5)
        
    def forward(self, x):
        if self.msr is not None:
            output, _, _ = self.msr(x)
            return output
        return self.fallback(x)
   
class Mamba2Mixin(nn.Module):
    def __init__(self, hidden_size, num_attention_heads, num_key_value_heads=None, *, use_fla: bool = True):
        super().__init__()
        self._use_fla = bool(use_fla)
        self.layer_idx = 0
        head_dim = hidden_size // num_attention_heads

        self.mamba2 = None
        self.fallback = None
        if self._use_fla:
            try:
                (_, Mamba2_cls, _) = _require_fla()
                self.mamba2 = Mamba2_cls(
                    num_heads=num_attention_heads,
                    hidden_size=hidden_size,
                    head_dim=head_dim,
                    state_size=32,
                    expand=1,
                    n_groups=1,
                    chunk_size=128,
                )
            except Exception:
                self.mamba2 = None

        if self.mamba2 is None:
            self.fallback = _FallbackSequenceMixer(int(hidden_size), kernel_size=7)
        
    def forward(self, x):
        if self.mamba2 is not None:
            self.mamba2.layer_idx = self.layer_idx
            output = self.mamba2(hidden_states=x)
            return output
        return self.fallback(x)

class RWKV6Mixin(nn.Module):
    def __init__(self, hidden_size, num_attention_heads, num_key_value_heads=None, *, use_fla: bool = True):
        super().__init__()
        self._use_fla = bool(use_fla)
        self.layer_idx = 0

        self.rwkv6 = None
        self.fallback = None
        if self._use_fla:
            try:
                (_, _, RWKV6Attention_cls) = _require_fla()
                self.rwkv6 = RWKV6Attention_cls(
                    hidden_size=hidden_size,
                    num_heads=num_attention_heads,
                    expand_k=0.5,
                    expand_v=0.5,
                    proj_low_rank_dim=64,
                    gate_low_rank_dim=64,
                )
            except Exception:
                self.rwkv6 = None

        if self.rwkv6 is None:
            self.fallback = _FallbackRWKVMixer(int(hidden_size), kernel_size=3)
        
    def forward(self, x):
        if self.rwkv6 is not None:
            self.rwkv6.layer_idx = self.layer_idx
            output, _, _ = self.rwkv6(x)
            return output
        return self.fallback(x)
 
class GroupedQuerySelfAttentionMixin(nn.Module):
    def __init__(self, hidden_size, num_attention_heads, num_key_value_heads=None, dropout=0.0, bias=True):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads if num_key_value_heads is not None else num_attention_heads
        self.dropout = dropout
        
        assert self.num_attention_heads % self.num_key_value_heads == 0, (
            f"num_attention_heads ({num_attention_heads}) must be divisible by num_key_value_heads ({num_key_value_heads})"
        )
        
        self.head_dim = hidden_size // num_attention_heads
        assert self.head_dim * num_attention_heads == self.hidden_size, (
            f"hidden_size must be divisible by num_attention_heads (got `hidden_size`: {self.hidden_size} "
            f"and `num_attention_heads`: {num_attention_heads})."
        )
        
        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=bias)
        self.k_proj = nn.Linear(hidden_size, self.num_key_value_heads * self.head_dim, bias=bias)
        self.v_proj = nn.Linear(hidden_size, self.num_key_value_heads * self.head_dim, bias=bias)
        self.out_proj = nn.Linear(hidden_size, hidden_size, bias=bias)
        
    def forward(self, x, attn_mask=None, is_causal=True):
        batch_size, seq_len, _ = x.shape
        
        q = self.q_proj(x)  
        k = self.k_proj(x)  
        v = self.v_proj(x)  
        
        q = q.view(batch_size, seq_len, self.num_attention_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        # PyTorch SDPA gained native GQA support (`enable_gqa`) in newer versions.
        # For older versions, expand K/V to match the number of query heads.
        dropout_p = self.dropout if self.training else 0.0
        if self.num_key_value_heads != self.num_attention_heads:
            try:
                attn_output = F.scaled_dot_product_attention(
                    q,
                    k,
                    v,
                    attn_mask=attn_mask,
                    dropout_p=dropout_p,
                    is_causal=is_causal,
                    enable_gqa=True,
                )
            except TypeError:
                repeat_factor = self.num_attention_heads // self.num_key_value_heads
                k_exp = k.repeat_interleave(repeat_factor, dim=1)
                v_exp = v.repeat_interleave(repeat_factor, dim=1)
                attn_output = F.scaled_dot_product_attention(
                    q,
                    k_exp,
                    v_exp,
                    attn_mask=attn_mask,
                    dropout_p=dropout_p,
                    is_causal=is_causal,
                )
        else:
            attn_output = F.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=attn_mask,
                dropout_p=dropout_p,
                is_causal=is_causal,
            )
        
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, seq_len, self.hidden_size)
        output = self.out_proj(attn_output)
        
        return output
    
class MultiHeadLatentAttentionMixin(nn.Module):
    def __init__(
        self, 
        hidden_size,
        num_attention_heads,
        num_key_value_heads = None,
        q_latent_dim = None,
        kv_lora_dim = None,
        dropout=0.0
    ):
        super().__init__()
        
        # assert config.v_head_dim is not None , f"v_head_dim is not defined {config.v_head_dim=}"
        # assert config.q_lora_rank is not None , f"q_lora_rank is not defined {config.q_lora_rank=}"
        # assert config.kv_lora_rank is not None , f"kv_lora_rank is not defined {config.kv_lora_rank=}"
        # assert config.rope_head_dim is not None , f"rope_head_dim is not defined {config.rope_head_dim=}"
        
        # self.config = config
        
        ## Example from deepseek v2 lite config
        # "hidden_size": 2048,
        # "kv_lora_rank": 512, // hidden_size / 4
        # "num_attention_heads" : 16, 
        # "num_key_value_heads": 16,
        # "qk_nope_head_dim": 128, // hidden_size / 16
        # "qk_rope_head_dim": 64, // hidden_size / 32
        
        self.dim = hidden_size
        self.num_heads = num_attention_heads
        self.v_head_dim = hidden_size // num_attention_heads
        
        self.nope_head_dim = max(hidden_size // num_attention_heads, 64) ## avoid getting too small
        self.rope_head_dim = max(hidden_size // (num_attention_heads * 2), 32) ## avoid getting too small
        
        if q_latent_dim is None:
            q_latent_dim = hidden_size // 2
            
        if kv_lora_dim is None:
            kv_lora_dim = hidden_size // 4
        
        self.q_lora_rank = q_latent_dim
        self.kv_lora_rank = kv_lora_dim
        
        self.dropout = dropout
        
        # note: head dim of query and key if different from head dim of value
        
        # (attention_dim == num_head*head_dim) > d_model in deepseekv2
        # this is dim between wV and wQ
        self.value_dim = self.num_heads * self.v_head_dim
        
        # this is dims between wQ and wK
        self.nope_dim = self.num_heads * self.nope_head_dim
        self.rope_dim = self.num_heads * self.rope_head_dim  
        
        # query compression
        self.compress_q_linear = nn.Linear(self.dim, self.q_lora_rank, bias=False)  # W_DQ
        
        self.decompress_q_nope = nn.Linear(self.q_lora_rank, self.nope_dim, bias=False)
        self.decompress_q_rope = nn.Linear(self.q_lora_rank, self.rope_dim, bias=False)
        
        self.q_norm = RMSNorm(self.q_lora_rank)
        
        
        # key and value compression
        self.compress_kv_linear = nn.Linear(self.dim, self.kv_lora_rank, bias=False)  # W_DKV
        self.decompress_k_nope = nn.Linear(self.kv_lora_rank, self.nope_dim, bias=False)
        self.decompress_v_linear = nn.Linear(self.kv_lora_rank, self.value_dim, bias=False)
        self.kv_norm = RMSNorm(self.kv_lora_rank)
        
        
        self.k_rope_linear = nn.Linear(self.dim, self.rope_head_dim  , bias=False)
        # self.rope_norm = RMSNorm(self.rope_dim) # not in deepseekv2

        self.proj = nn.Linear(self.value_dim , self.dim, bias=False)
        self.res_dropout = nn.Dropout(p=dropout)
        
        
    def forward(self, x: Tensor):
        batch_size, seq_len, _ = x.shape

        compressed_q = self.compress_q_linear(x)
        norm_q = self.q_norm(compressed_q)
        query_nope:Tensor = self.decompress_q_nope(norm_q)
        query_rope:Tensor = self.decompress_q_rope(norm_q)

        compressed_kv = self.compress_kv_linear(x)
        norm_kv = self.kv_norm(compressed_kv)
        key_nope: Tensor = self.decompress_k_nope(norm_kv)
        value: Tensor = self.decompress_v_linear(norm_kv)
        
        key_rope:Tensor = self.k_rope_linear(x)
        # norm_rope = self.rope_norm(key_rope)

        query_nope = query_nope.view(batch_size, seq_len, self.num_heads, self.nope_head_dim).transpose(1,2)
        query_rope = query_rope.view(batch_size, seq_len, self.num_heads, self.rope_head_dim).transpose(1,2)
        
        key_rope = key_rope.view(batch_size, seq_len, 1, self.rope_head_dim).transpose(1,2)
        key_nope = key_nope.view(batch_size, seq_len, self.num_heads, self.nope_head_dim).transpose(1,2)
        
        value = value.view(batch_size, seq_len, self.num_heads, self.v_head_dim).transpose(1,2)
        
        # *** the line that fixes MLA :) ***
        # key_rope = key_rope/self.num_heads 

        # q_rope,k_rope = apply_rope(query_rope,key_rope, cis=freqs_cis)

        # IMPORTANT: fully initialize Q/K. Leaving any slice uninitialized (torch.empty)
        # will inject garbage values and can produce NaN losses.
        q_recombined = torch.empty(
            (batch_size, self.num_heads, seq_len, self.rope_head_dim + self.nope_head_dim),
            device=x.device,
            dtype=x.dtype,
        )
        k_recombined = torch.empty(
            (batch_size, self.num_heads, seq_len, self.rope_head_dim + self.nope_head_dim),
            device=x.device,
            dtype=x.dtype,
        )

        q_recombined[:, :, :, : self.nope_head_dim] = query_nope
        q_recombined[:, :, :, self.nope_head_dim :] = query_rope

        k_recombined[:, :, :, : self.nope_head_dim] = key_nope
        # key_rope is shared across heads (shape: B,1,S,D); expand for assignment.
        k_recombined[:, :, :, self.nope_head_dim :] = key_rope.expand(-1, self.num_heads, -1, -1)

        output = F.scaled_dot_product_attention(q_recombined, k_recombined, value, is_causal=True, dropout_p=self.dropout)

        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.num_heads * self.v_head_dim)

        output = self.proj(output)
        output = self.res_dropout(output)
        return output
