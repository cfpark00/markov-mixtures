import torch
import torch.nn as nn
from torch.nn import functional as F
import math

from mltools.networks.network_tools import get_conv, zero_init, patch_interpolate


class AttnBlock(nn.Module):
    def __init__(self, in_channels, n_heads=4, dim=2, **kwargs):
        super().__init__()
        self.in_channels = in_channels
        assert (
            self.in_channels % n_heads == 0
        ), "in_channels must be divisible by n_heads"
        self.n_heads = n_heads
        self.dim = dim
        assert self.dim == 2 or self.dim == 3, "dim must be 2 or 3"

        norm_params = kwargs.get("norm_params", {})

        self.norm = nn.GroupNorm(num_channels=in_channels, **norm_params)

        self.q = get_conv(
            in_channels, in_channels, dim=self.dim, kernel_size=1, stride=1, padding=0
        )
        self.k = get_conv(
            in_channels, in_channels, dim=self.dim, kernel_size=1, stride=1, padding=0
        )
        self.v = get_conv(
            in_channels, in_channels, dim=self.dim, kernel_size=1, stride=1, padding=0
        )
        self.proj_out = get_conv(
            in_channels, in_channels, dim=self.dim, kernel_size=1, stride=1, padding=0
        )

    def forward(self, x):
        h_ = x
        h_ = self.norm(h_)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        # compute attention
        if self.dim == 2:
            b, c, h, w = q.shape
            c_ = c // self.n_heads
            q = q.reshape(b, c_, self.n_heads, h * w)
            k = k.reshape(b, c_, self.n_heads, h * w)
            w_ = torch.einsum("bcnq,bcnk->bqkn", q, k)
            w_ = w_ * (int(c_) ** (-0.5))
            w_ = torch.nn.functional.softmax(w_, dim=2)
            v = v.reshape(b, c_, self.n_heads, h * w)
            h_ = torch.einsum("bcnd,bqdn->bcnq", v, w_)
            h_ = h_.reshape(b, c, h, w)
            h_ = self.proj_out(h_)
        elif self.dim == 3:
            b, c, d, h, w = q.shape
            c_ = c // self.n_heads
            q = q.reshape(b, c_, self.n_heads, d * h * w)
            k = k.reshape(b, c_, self.n_heads, d * h * w)
            w_ = torch.einsum("bcnq,bcnk->bqkn", q, k)
            w_ = w_ * (int(c_) ** (-0.5))
            w_ = torch.nn.functional.softmax(w_, dim=2)
            v = v.reshape(b, c_, self.n_heads, d * h * w)
            h_ = torch.einsum("bcnd,bqdn->bcnq", v, w_)
            h_ = h_.reshape(b, c, d, h, w)
            h_ = self.proj_out(h_)
        return x + h_


class ResNetBlock(nn.Module):
    def __init__(
        self,
        ch_in,
        ch_out,
        dim=2,
        conditioning_dims=None,
        dropout_prob=0.0,
        nca_params={},
        cond_proj_type="zerolinear"
    ):
        super().__init__()
        self.ch_in = ch_in
        self.ch_out = ch_out
        self.dim = dim
        assert self.dim in [2, 3], "dim must be 2 or 3"
        self.conditioning_dims = conditioning_dims

        self.nca_params = nca_params
        norm_params = self.nca_params.get("norm_params", {})
        get_act = self.nca_params.get("get_act", lambda: nn.GELU())
        conv_params = self.nca_params.get("conv_params", {})

        self.net1 = nn.Sequential(
            nn.GroupNorm(num_channels=ch_in, **norm_params),
            get_act(),
            get_conv(ch_in, ch_out, dim=self.dim, **conv_params),
        )
        if conditioning_dims is not None:
            self.cond_projs = nn.ModuleList()
            for condition_dim in self.conditioning_dims:
                if cond_proj_type == "zerolinear":
                    self.cond_projs.append(zero_init(nn.Linear(condition_dim, ch_out)))
                elif cond_proj_type == "linear":
                    self.cond_projs.append(nn.Linear(condition_dim, ch_out))
                elif cond_proj_type == "mlp":
                    self.cond_projs.append(
                        nn.Sequential(
                            nn.Linear(condition_dim, ch_out),
                            get_act(),
                            nn.Linear(ch_out, ch_out),
                            get_act(),
                        )
                    )
                else:
                    raise ValueError(f"Unknown cond_proj_type: {cond_proj_type}")
        self.net2 = nn.Sequential(
            nn.GroupNorm(num_channels=ch_out, **norm_params),
            get_act(),
            *([nn.Dropout(dropout_prob)] * (dropout_prob > 0.0)),
            get_conv(ch_out, ch_out, dim=self.dim, init=zero_init, **conv_params),
        )
        if ch_in != ch_out:
            self.skip_conv = get_conv(
                ch_in, ch_out, dim=self.dim, kernel_size=1, padding=0
            )

    def forward(self, x, conditionings=None):
        h = self.net1(x)
        if conditionings is not None:
            assert len(conditionings) == len(self.conditioning_dims)
            assert all(
                [
                    conditionings[i].shape == (x.shape[0], self.conditioning_dims[i])
                    for i in range(len(conditionings))
                ]
            )
            for i, conditioning in enumerate(conditionings):
                conditioning_ = self.cond_projs[i](conditioning)
                if self.dim == 2:
                    h = h + conditioning_[:, :, None, None]
                elif self.dim == 3:
                    h = h + conditioning_[:, :, None, None, None]
        h = self.net2(h)
        if x.shape[1] != self.ch_out:
            x = self.skip_conv(x)
        return x + h


class ResNetDown(nn.Module):
    def __init__(self, resnet_blocks, attention_blocks=None):
        super().__init__()
        self.resnet_blocks = resnet_blocks
        self.attention_blocks = attention_blocks
        self.dim = self.resnet_blocks[-1].dim
        self.down = get_conv(
            self.resnet_blocks[-1].ch_out,
            self.resnet_blocks[-1].ch_out,
            dim=self.dim,
            kernel_size=2,
            stride=2,
            padding=0,
        )

    def forward(self, x, conditionings, no_down=False):
        for i, resnet_block in enumerate(self.resnet_blocks):
            x = resnet_block(x, conditionings)
            if self.attention_blocks is not None:
                x = self.attention_blocks[i](x)
        if no_down:
            return x, None
        x_skip = x
        x = self.down(x)
        return x, x_skip


class ResNetUp(nn.Module):
    def __init__(
        self, resnet_blocks, attention_blocks=None, ch_out=None, conv_params={}
    ):
        super().__init__()
        self.resnet_blocks = resnet_blocks
        self.ch_out = ch_out if ch_out is not None else self.resnet_blocks[-1].ch_out
        self.attention_blocks = attention_blocks
        self.dim = self.resnet_blocks[-1].dim
        self.up = get_conv(
            self.resnet_blocks[-1].ch_out,
            self.ch_out,
            dim=self.dim,
            kernel_size=2,
            stride=2,
            padding=0,
            transposed=True,
        )

    def forward(self, x, x_skip=None, conditionings=None, no_up=False):
        for i, resnet_block in enumerate(self.resnet_blocks):
            x = resnet_block(x, conditionings)
            if self.attention_blocks is not None:
                x = self.attention_blocks[i](x)
        if not no_up:
            x = self.up(x)
        if x_skip is not None:
            x = torch.cat([x, x_skip], dim=1)
        return x

class LayerNorm(nn.Module):
    """ LayerNorm but with an optional bias. PyTorch 2.0 doesn't support simply bias=False """

    def __init__(self, ndim, bias=True):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)

class SelfAttentionBlock(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.n_embd_head = config.n_embd // config.n_head

        self.block_size = config.block_size
        self.causal= config.causal
        self.rope=config.get("rope",False)
        if self.rope:
            self.create_rope_cache()
        
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        # regularization
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.dropout = config.dropout
        # flash attention make GPU go brrrrr but support is only in PyTorch >= 2.0
        if config.flash:
            assert hasattr(torch.nn.functional, 'scaled_dot_product_attention'), "PyTorch >= 2.0 is required for Flash Attention"
            self.flash = True
        else:
            print("WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0")
            # causal mask to ensure that attention is only applied to the left in the input sequence
            self.create_bias()
            self.flash = False

    def create_rope_cache(self,base=10_000):
        dim=self.n_embd_head
        theta = torch.exp(torch.arange(0, dim, 2).float() * (-math.log(base)/ dim))
        self.register_buffer("theta", theta, persistent=False)
        seq_idx = torch.arange(self.block_size, dtype=self.theta.dtype, device=self.theta.device)
        idx_theta = torch.einsum("i, j -> ij", seq_idx, self.theta).float()
        rope_cache = torch.stack([torch.cos(idx_theta), torch.sin(idx_theta)], dim=-1)
        self.register_buffer("rope_cache", rope_cache, persistent=False)

    def create_bias(self,device="cpu"):
        self.register_buffer("bias", torch.tril(torch.ones(self.block_size, self.block_size))
                                    .view(1, 1, self.block_size, self.block_size).to(device))

    def apply_rope(self,x,input_pos=None):
        seq_len = x.size(1)
        # extract the values based on whether input_pos is set or not
        rope_cache_ = self.rope_cache[:seq_len] if input_pos is None else self.cache[input_pos]
        # reshape input; the last dimension is used for computing the output.
        # Cast to float to match the reference implementation
        # tensor has shape [b, s, n_h, h_d // 2, 2]
        xshaped = x.float().reshape(*x.shape[:-1], -1, 2)

        # reshape the cache for broadcasting
        # tensor has shape [b, s, 1, h_d // 2, 2] if packed samples,
        # otherwise has shape [1, s, 1, h_d // 2, 2]
        rope_cache_ = rope_cache_.view(-1, xshaped.size(1), 1, xshaped.size(3), 2)

        # tensor has shape [b, s, n_h, h_d // 2, 2]
        x_out = torch.stack(
            [
                xshaped[..., 0] * rope_cache_[..., 0]
                - xshaped[..., 1] * rope_cache_[..., 1],
                xshaped[..., 1] * rope_cache_[..., 0]
                + xshaped[..., 0] * rope_cache_[..., 1],
            ],
            -1,
        )
        # tensor has shape [b, s, n_h, h_d]
        x_out = x_out.flatten(3)
        return x_out.type_as(x)

    def forward(self, x):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)
        assert C == self.n_embd, f"Input embedding dimension {C} does not match model embedding dimension {self.n_embd}"

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k, v  = self.c_attn(x).split(self.n_embd, dim=2)
        if self.rope:
            kT=k.view(B, T, self.n_head, self.n_embd_head)
            qT=q.view(B, T, self.n_head, self.n_embd_head)
            kT=self.apply_rope(kT)
            qT=self.apply_rope(qT)
            k=kT.transpose(1, 2)
            q=qT.transpose(1, 2)
            v = v.view(B, T, self.n_head, self.n_embd_head).transpose(1, 2)
        else:
            k = k.view(B, T, self.n_head, self.n_embd_head).transpose(1, 2) # (B, nh, T, hs)
            q = q.view(B, T, self.n_head, self.n_embd_head).transpose(1, 2) # (B, nh, T, hs)
            v = v.view(B, T, self.n_head, self.n_embd_head).transpose(1, 2) # (B, nh, T, hs)


        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        if self.flash:
            # efficient attention using Flash Attention CUDA kernels
            y = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=self.dropout if self.training else 0, is_causal=self.causal)
        else:
            # manual implementation of attention
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        # output projection
        y = self.resid_dropout(self.c_proj(y))
        return y
    
    def cached_forward(self,x,hiddens,suffix="",**kwargs):
        assert suffix.startswith("^") or suffix==""
        if not hasattr(self,"bias"):
            self.create_bias(x.device)

        B, T, C = x.size()
        q, k, v  = self.c_attn(x).split(self.n_embd, dim=2)
        if self.rope:
            kT = k.view(B, T, self.n_head, self.n_embd_head)
            qT = q.view(B, T, self.n_head, self.n_embd_head)
            hiddens["kT"+suffix]=kT.detach().clone()
            hiddens["qT"+suffix]=qT.detach().clone()
            kT=self.apply_rope(kT)
            qT=self.apply_rope(qT)
            k=kT.transpose(1, 2)
            q=qT.transpose(1, 2)
            hiddens["k_rope"+suffix]=k.detach().clone()
            hiddens["q_rope"+suffix]=q.detach().clone()
            v = v.view(B, T, self.n_head, self.n_embd_head).transpose(1, 2)
            hiddens["v"+suffix]=v.detach().clone()
        else:
            k = k.view(B, T, self.n_head, self.n_embd_head).transpose(1, 2)
            hiddens["k"+suffix]=k.detach().clone()
            q = q.view(B, T, self.n_head, self.n_embd_head).transpose(1, 2)
            hiddens["q"+suffix]=q.detach().clone()
            v = v.view(B, T, self.n_head, self.n_embd_head).transpose(1, 2)
            hiddens["v"+suffix]=v.detach().clone()
        #don't flash
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        hiddens["attn_um"+suffix]=att.detach().clone()
        att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        hiddens["attn"+suffix]=att.detach().clone()
        att = self.attn_dropout(att)
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        hiddens["y_out"+suffix]=y.detach().clone()
        y = self.resid_dropout(self.c_proj(y))
        hiddens["y_out_proj"+suffix]=y.detach().clone()
        return y
    
    @torch.no_grad()
    def patched_forward(self,x,patches,suffix="",**kwargs):
        assert suffix.startswith("^") or suffix==""
        if not hasattr(self,"bias"):
            self.create_bias(x.device)

        B, T, C = x.size()
        q, k, v  = self.c_attn(x).split(self.n_embd, dim=2)
        if self.rope:
            kT = k.view(B, T, self.n_head, self.n_embd_head)
            qT = q.view(B, T, self.n_head, self.n_embd_head)
            kT=patch_interpolate(kT,"kT"+suffix,patches)
            qT=patch_interpolate(qT,"qT"+suffix,patches)
            kT=self.apply_rope(kT)
            qT=self.apply_rope(qT)
            k=kT.transpose(1, 2)
            q=qT.transpose(1, 2)
            k=patch_interpolate(k,"k_rope"+suffix,patches)
            q=patch_interpolate(q,"q_rope"+suffix,patches)
            v = v.view(B, T, self.n_head, self.n_embd_head).transpose(1, 2)
        else:
            k = k.view(B, T, self.n_head, self.n_embd_head).transpose(1, 2)
            k=patch_interpolate(k,"k"+suffix,patches)
            q = q.view(B, T, self.n_head, self.n_embd_head).transpose(1, 2)
            q=patch_interpolate(q,"q"+suffix,patches)
            v = v.view(B, T, self.n_head, self.n_embd_head).transpose(1, 2)
            v=patch_interpolate(v,"v"+suffix,patches)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att=patch_interpolate(att,"attn_um"+suffix,patches)
        att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        att=patch_interpolate(att,"attn"+suffix,patches)
        att = self.attn_dropout(att)
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y=patch_interpolate(y,"y_out"+suffix,patches)
        y = self.resid_dropout(self.c_proj(y))
        y=patch_interpolate(y,"y_out_proj"+suffix,patches)
        return y

    
class MLPBlock(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.rmlp=config.get("rmlp",4)
        #assert integer self.rmlp*n_embd, for e.g. rmlp=0.25
        self.d_hidden = int(self.rmlp * config.n_embd)
        assert self.rmlp*config.n_embd==self.d_hidden, "rmlp*n_embd must be an integer"
        self.c_fc    = nn.Linear(config.n_embd, self.d_hidden, bias=config.bias)
        self.gelu    = nn.GELU()
        self.c_proj  = nn.Linear(self.d_hidden, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x

class TransformerBlock(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.has_mlp=config.get("mlp",True)
        self.has_ln=config.get("ln",True)
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias) if self.has_ln else nn.Identity()
        self.attn = SelfAttentionBlock(config)
        if self.has_mlp:
            self.ln_2 = LayerNorm(config.n_embd, bias=config.bias) if self.has_ln else nn.Identity()
            self.mlp = MLPBlock(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        if self.has_mlp:
            x = x + self.mlp(self.ln_2(x))
        return x

    def cached_forward(self,x,hiddens,suffix="",**kwargs):
        assert suffix.startswith("^") or suffix==""
        attn_res=self.attn.cached_forward(self.ln_1(x),hiddens,suffix=suffix,**kwargs)
        hiddens["attn_res"+suffix]=attn_res
        x=x+attn_res
        hiddens["x_attn"+suffix]=x.detach().clone()
        if self.has_mlp:
            mlp_res=self.mlp(self.ln_2(x))
            hiddens["mlp_res"+suffix]=mlp_res
            x=x+mlp_res
        return x
    
    @torch.no_grad()
    def patched_forward(self,x,patches,suffix="",**kwargs):
        assert suffix.startswith("^") or suffix==""
        attn_res=self.attn.patched_forward(self.ln_1(x),patches,suffix=suffix,**kwargs)
        attn_res=patch_interpolate(attn_res,"attn_res"+suffix,patches)
        x=x+attn_res
        x=patch_interpolate(x,"x_attn"+suffix,patches)
        if self.has_mlp:
            mlp_res=self.mlp(self.ln_2(x))
            mlp_res=patch_interpolate(mlp_res,"mlp_res"+suffix,patches)
            x=x+mlp_res
        return x

