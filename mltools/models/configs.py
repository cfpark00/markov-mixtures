from dataclasses import dataclass

@dataclass
class GPTConfig:
    block_size: int = 1024
    tokenized: bool = True
    in_size: int = 50304
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    rmlp: int = 4
    dropout: float = 0.0
    bias: bool = True 
    causal: bool = True
    flash: bool = True
    verbose: int =1
    pos_embed: bool = True
    rope: bool = False
    mlp: bool = True# MLP
    ln: bool = True# LayerNorm
    tie_emb: bool = True

    def get(self,key,default=None):
        return getattr(self,key,default)
