from math import pi, log

import torch
import torch.nn as nn
from torch.cuda.amp import autocast
from torch import einsum, broadcast_tensors, Tensor
from torch.nn import functional as F
from efficient_kan.model import KAN as EFFICIENT_KAN
from kan.KAN import KAN as ORIGINAL_KAN
from einops import rearrange, repeat

def exists(val):
    return val is not None
    
def default(val, d):
    return val if exists(val) else d

def rotate_half(x):
    x = rearrange(x, '... (d r) -> ... d r', r = 2)
    x1, x2 = x.unbind(dim = -1)
    x = torch.stack((-x2, x1), dim = -1)
    return rearrange(x, '... d r -> ... (d r)')
    
@autocast(enabled = False)
def apply_rotary_emb(freqs, t, start_index = 0, scale = 1., seq_dim = -2):
    if t.ndim == 3:
        seq_len = t.shape[seq_dim]
        freqs = freqs[-seq_len:].to(t)

    rot_dim = freqs.shape[-1]
    end_index = start_index + rot_dim

    assert rot_dim <= t.shape[-1], f'feature dimension {t.shape[-1]} is not of sufficient size to rotate in all the positions {rot_dim}'

    t_left, t, t_right = t[..., :start_index], t[..., start_index:end_index], t[..., end_index:]
    t = (t * freqs.cos() * scale) + (rotate_half(t) * freqs.sin() * scale)
    return torch.cat((t_left, t, t_right), dim = -1)
    
class RotaryEmbedding(nn.Module):
    def __init__(self, dim, custom_freqs=None, freqs_for='lang', theta=10000, max_freq=10, num_freqs=1, learned_freq=False, use_xpos=False, xpos_scale_base=512, interpolate_factor=1., theta_rescale_factor=1., seq_before_head_dim=False, cache_if_possible=True):
        super().__init__()
        theta *= theta_rescale_factor ** (dim / (dim - 2))
        self.freqs_for = freqs_for
        if custom_freqs is not None:
            freqs = custom_freqs
        elif freqs_for == 'lang':
            freqs = 1. / (theta ** (torch.arange(0, dim, 2)[:(dim // 2)].float() / dim))
        elif freqs_for == 'pixel':
            freqs = torch.linspace(1., max_freq / 2, dim // 2) * pi
        elif freqs_for == 'constant':
            freqs = torch.ones(num_freqs).float()

        self.cache_if_possible = cache_if_possible
        self.tmp_store('cached_freqs', None)
        self.tmp_store('cached_scales', None)

        self.freqs = nn.Parameter(freqs, requires_grad=learned_freq)
        self.learned_freq = learned_freq
        self.tmp_store('dummy', torch.tensor(0))
        self.seq_before_head_dim = seq_before_head_dim
        self.default_seq_dim = -3 if seq_before_head_dim else -2
        self.interpolate_factor = interpolate_factor

        

        if not use_xpos:
            self.tmp_store('scale', None)
            return

        scale = (torch.arange(0, dim, 2) + 0.4 * dim) / (1.4 * dim)
        self.scale_base = xpos_scale_base
        self.tmp_store('scale', scale)

    @property
    def device(self):
        return self.dummy.device

    def tmp_store(self, key, value):
        self.register_buffer(key, value, persistent=False)

    def get_seq_pos(self, seq_len, device, dtype, offset=0):
        return (torch.arange(seq_len, device=device, dtype=dtype) + offset) / self.interpolate_factor

    def rotate_queries_or_keys(self, t, seq_dim=None, offset=0):
        seq_dim = default(seq_dim, self.default_seq_dim)
        device, dtype, seq_len = t.device, t.dtype, t.shape[seq_dim]
        freqs = self.forward(self.get_seq_pos(seq_len, device=device, dtype=dtype, offset=offset), seq_len=seq_len, offset=offset)
        if seq_dim == -3:
            freqs = rearrange(freqs, 'n d -> n 1 d')
        return apply_rotary_emb(freqs, t, seq_dim=seq_dim)

    def rotate_queries_and_keys(self, q, k, seq_dim=None):
        seq_dim = default(seq_dim, self.default_seq_dim)
        device, dtype, seq_len = q.device, q.dtype, q.shape[seq_dim]
        seq = self.get_seq_pos(seq_len, dtype=dtype, device=device)
        freqs = self.forward(seq, seq_len=seq_len)
        scale = self.get_scale(seq, seq_len=seq_len).to(dtype)
        if seq_dim == -3:
            freqs = rearrange(freqs, 'n d -> n 1 d')
            scale = rearrange(scale, 'n d -> n 1 d')
        rotated_q = apply_rotary_emb(freqs, q, scale=scale, seq_dim=seq_dim)
        rotated_k = apply_rotary_emb(freqs, k, scale=scale**-1, seq_dim=seq_dim)
        return rotated_q.type(q.dtype), rotated_k.type(k.dtype)

    def get_scale(self, t, seq_len=None, offset=0):
        if self.use_xpos:
            should_cache = self.cache_if_possible and seq_len is not None
            if should_cache and self.cached_scales is not None and (seq_len + offset) <= self.cached_scales.shape[0]:
                return self.cached_scales[offset:(offset + seq_len)]
            power = (t - len(t) // 2) / self.scale_base
            scale = self.scale ** rearrange(power, 'n -> n 1')
            scale = torch.cat((scale, scale), dim=-1)
            if should_cache:
                self.tmp_store('cached_scales', scale)
            return scale
        return 1.

    def forward(self, t, seq_len=None, offset=0):
        should_cache = self.cache_if_possible and not self.learned_freq and seq_len is not None and self.freqs_for != 'pixel'
        if should_cache and self.cached_freqs is not None and (offset + seq_len) <= self.cached_freqs.shape[0]:
            return self.cached_freqs[offset:(offset + seq_len)].detach()
        freqs = self.freqs
        freqs = einsum('..., f -> ... f', t.type(freqs.dtype), freqs)
        freqs = repeat(freqs, '... n -> ... (n r)', r=2)
        if should_cache:
            self.tmp_store('cached_freqs', freqs.detach())
        return freqs
        
class InfiniKANTransformer(nn.Module):
    def __init__(
        self,
        num_tokens,
        dim,
        depth,
        heads,
        dim_head,
        attn_dropout,
        ff_dropout,
        use_mem_delta_rule=False,
        kan_implementation="EFFICIENT_KAN",
    ):
        super().__init__()
        self.token_emb = nn.Embedding(num_tokens, dim)
        self.pos_emb = nn.Embedding(1024, dim)
        self.drop = nn.Dropout(0.1)

        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(
                nn.ModuleList(
                    [
                        CausalAttention(
                            dim,
                            dim_head,
                            heads,
                            use_mem_delta_rule,
                            dropout=attn_dropout,
                        ),
                        KANFeedForward(
                            dim, kan_implementation, dropout=ff_dropout
                        ),
                    ]
                )
            )

        self.norm = RMSNorm(dim)
        KAN = self.get_KAN(kan_implementation)
        self.to_logits = KAN(width=[dim, num_tokens], bias_trainable=False)

    def get_KAN(self, kan_implementation):
        if kan_implementation == "EFFICIENT_KAN":
            return EFFICIENT_KAN
        elif kan_implementation == "ORIGINAL_KAN":
            return ORIGINAL_KAN
        else:
            raise NotImplementedError()

    def forward(self, x, past_memories=None, return_new_memories=False):
        b, t = x.size()
        pos_emb = self.pos_emb(torch.arange(t, device=x.device))
        x = self.drop(self.token_emb(x) + pos_emb)

        new_memories = []
        for attn, ff in self.layers:
            x, new_mem = attn(x, past_memories, return_new_memories)
            x = ff(x) + x
            new_memories.append(new_mem)

        x = self.norm(x)
        logits = self.to_logits(x)

        if not return_new_memories:
            return logits

        return logits, new_memories

class KANFeedForward(nn.Module):
    def __init__(self, dim, kan_implementation, dropout):
        super().__init__()
        KAN = self.get_KAN(kan_implementation)
        self.norm = RMSNorm(dim)
        self.kan1 = KAN(width=[dim, dim * 4])
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.kan2 = KAN(width=[dim * 4, dim])

    def get_KAN(self, kan_implementation):
        if kan_implementation == "EFFICIENT_KAN":
            return EFFICIENT_KAN
        elif kan_implementation == "ORIGINAL_KAN":
            return ORIGINAL_KAN
        else:
            raise NotImplementedError()

    def forward(self, x):
        x = self.norm(x)
        x = self.kan1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.kan2(x)
        return x

class RMSNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.scale = dim ** 0.5
        self.gamma = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        return F.normalize(x, dim=-1) * self.scale * self.gamma

class CausalAttention(nn.Module):
    def __init__(
        self,
        dim,
        dim_head,
        heads,
        kan_implementation="EFFICIENT_KAN",
        use_mem_delta_rule=False,
        dropout=0.0,
    ):
        super().__init__()
        KAN = EFFICIENT_KAN#self.get_KAN(kan_implementation)
        self.scale = dim_head ** -0.5
        self.heads = heads
        self.use_mem_delta_rule = use_mem_delta_rule
        self.norm = RMSNorm(dim)
        self.rotary_emb = RotaryEmbedding(dim_head)
        self.to_qkv = KAN(width=[dim, dim * 3], bias_trainable=False)
        self.to_out = KAN(width=[dim, dim], bias_trainable=False)
        self.dropout = nn.Dropout(dropout)
        self.fastweight_mem = FastweightMemory(
            heads, use_mem_delta_rule=use_mem_delta_rule
        )

    def forward(self, x, past_memories=None, return_new_memories=False):
        x = self.norm(x)
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> b h n d", h=self.heads), qkv)
        q, k = map(lambda t: self.rotary_emb.rotate_queries_or_keys(t), (q, k))
        sim = torch.einsum("b h i d, b h j d -> b h i j", q, k) * self.scale
        attn = sim.softmax(dim=-1)
        attn = self.dropout(attn)
        out = torch.einsum("b h i j, b h j d -> b h i d", attn, v)
        out = self.fastweight_mem.retrieve_and_add_to_output(out, q, past_memories)
        out = rearrange(out, "b h n d -> b n (h d)")
        out = self.to_out(out)

        if not return_new_memories:
            return out, None

        new_memories = self.fastweight_mem.create_new_memories(k, v, past_memories)
        return out, new_memories
        
    def get_KAN(self, kan_implementation):
        if kan_implementation == "EFFICIENT_KAN":
            return EFFICIENT_KAN
        elif kan_implementation == "ORIGINAL_KAN":
            return ORIGINAL_KAN
        else:
            raise NotImplementedError()


class FastweightMemory(nn.Module):
    def __init__(self, heads, use_mem_delta_rule=False):
        super().__init__()
        self.use_mem_delta_rule = use_mem_delta_rule
        self.head_gates = nn.Parameter(torch.ones(heads) * 10.0)

    def create_new_memories(self, keys, values, past_memories):
        keys = F.elu(keys) + 1
        if self.use_mem_delta_rule and past_memories is not None:
            delta_v = retrieve_from_kv_memories(keys, past_memories)
            values = values - delta_v
        new_memories_kv = torch.einsum("... n d, ... n e -> ... d e", keys, values)
        new_memories_norm = keys.sum(dim=-2)
        if past_memories is not None:
            new_memories_kv = new_memories_kv + past_memories[0]
            new_memories_norm = new_memories_norm + past_memories[1]
        return (new_memories_kv, new_memories_norm)

    def retrieve_and_add_to_output(self, out, queries, past_memories):
        if past_memories is None:
            return out
        queries = F.elu(queries) + 1
        mem_out = retrieve_from_kv_memories(queries, past_memories)
        gates = self.head_gates.sigmoid().view(-1, 1, 1)
        out = out * gates + mem_out * (1 - gates)
        return out

def retrieve_from_kv_memories(queries, past_memories, eps=1e-10):
    past_memories_kv, past_memories_norm = past_memories
    numer = torch.einsum("... n d, ... d e -> ... n e", queries, past_memories_kv)
    denom = torch.einsum("... n d, ... d -> ... n", queries, past_memories_norm)
    denom = rearrange(denom, "... n -> ... n 1")
    return numer / denom.clamp(min=eps)


if __name__ == "__main__":
    # Model parameters
    num_tokens = 10000
    dim = 512
    depth = 12
    heads = 8
    dim_head = 64
    attn_dropout = 0.1
    ff_dropout = 0.1
    use_mem_delta_rule = True
    kan_implementation = "EFFICIENT_KAN"

    # Initialize the model
    model = InfiniKANTransformer(
        num_tokens=num_tokens,
        dim=dim,
        depth=depth,
        heads=heads,
        dim_head=dim_head,
        attn_dropout=attn_dropout,
        ff_dropout=ff_dropout,
        use_mem_delta_rule=use_mem_delta_rule,
        kan_implementation=kan_implementation,
    )

    x = torch.randint(0, num_tokens, (1, 512))

    # Forward pass with memory retrieval
    logits, memories = model(x, return_new_memories=True) # Set False if you don't want memory

    # Output shapes
    print(f"Logits shape: {logits.shape}")  # (1, 512, 10000)
    print(f"Number of layers with memories: {len(memories)}")  # 12 (number of layers)
    print(f"Memory shape (KV matrix): {memories[0][0].shape}")  # (1, 8, 64, 64) (batch, heads, dim_head, dim_head)
    print(f"Memory shape (Normalization vector): {memories[0][1].shape}")  # (1, 8, 64) (batch, heads, dim_head)
