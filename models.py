import torch
import torch.nn as nn
import torch.nn.functional as F

from mltools.networks import networks

class Transformer(nn.Module):
    def __init__(self,gpt_config):
        super(Transformer, self).__init__()
        self.tokenized=gpt_config.tokenized
        self.transformer=networks.Transformer(gpt_config)

    def forward(self, x,**kwargs):
        if "two_token_input" in kwargs and kwargs["two_token_input"]:
            k=kwargs["k"]#maximum indice+1 of x
            assert x.dtype==torch.long
            x=x.clone()
            x_prevs=x[:,:-1]
            x[:,1:]=(x[:,1:]+1)*k+x_prevs
            #if x.shape[1]>20:
            #    print(x)
            #    assert False
        return self.transformer(x)
    
    def get_loss_from_logits(self,logits_next,ids_next,loss_mask=None,reduction="mean"):
        if loss_mask is not None:
            loss=F.cross_entropy(logits_next,ids_next,reduction=reduction)
        else:
            loss=F.cross_entropy(logits_next.reshape(-1,logits_next.shape[-1]),ids_next.reshape(-1),reduction=reduction)
            if reduction=="none":
                loss=loss.view(ids_next.shape)
        return loss

    def get_loss(self, x,token_mask=None,loss_mask=None,reduction="mean",**kwargs):
        if self.tokenized:
            ids_now=x[:, :-1]
            ids_next=x[:, 1:]
            logits_next=self(ids_now,**kwargs)
            if token_mask is not None:
                logits_next=logits_next[:,:,token_mask]
            if loss_mask is not None:
                loss_mask=loss_mask[:,1:]
                ids_next=ids_next[loss_mask]
                logits_next=logits_next[loss_mask]
            return self.get_loss_from_logits(logits_next,ids_next,loss_mask,reduction)
        else:
            assert token_mask is None
            x_now = x[:, :-1]
            x_next= x[:, 1:]
            x_next_pred = self(x_now)
            if loss_mask is not None:
                loss_mask=loss_mask[:,1:]
                x_next=x_next[loss_mask,:]
                x_next_pred=x_next_pred[loss_mask,:]
                loss=F.mse_loss(x_next_pred,x_next,reduction=reduction)
            else:
                loss=F.mse_loss(x_next_pred,x_next,reduction=reduction)
        return loss

    def get_num_params(self, non_embedding=True):
        """
        Return the number of parameters in the model.
        For non-embedding count (default), the position embeddings get subtracted.
        The token embeddings would too, except due to the parameter sharing these
        params are actually used as weights in the final layer, so we include them.
        """
        n_params = sum(p.numel() for p in self.transformer.parameters())
        if non_embedding:
            if self.transformer.transformer.wpe is not None:
                n_params -= self.transformer.transformer.wpe.weight.numel()
            if self.transformer.transformer.wte is not None:
                n_params -= self.transformer.transformer.wte.weight.numel()
        return n_params

    @torch.no_grad()
    def generate(self,x,n,T=1.0):
        for i in range(n):
            if self.tokenized:
                if T<=0.0:
                    logits_next=self(x)[:,-1,:]
                    next_token=torch.argmax(logits_next,dim=-1)
                    x=torch.cat([x,next_token[:,None]],dim=1)
                else:
                    logits_next=self(x)[:,-1,:] / T
                    next_token=torch.multinomial(F.softmax(logits_next,dim=-1),num_samples=1)
                    x=torch.cat([x,next_token],dim=1)
            else:
                x_next=self(x)[:,-1,:]
                x=torch.cat([x,x_next],dim=1)
        return x
    
    def load_state_dict_nc(self,state_dict,**kwargs):#decompile
        state_dict_new={}
        for key,p in state_dict.items():
            if key=="_orig_mod.transformer.lin.weight" or key=="_orig_mod.transformer.lin.bias":
                print("Skipping",key)
                continue
            state_dict_new[key.replace("_orig_mod.","")]=p
        self.load_state_dict(state_dict_new,**kwargs,strict=False)
    
    def cached_forward(self,x,*args,**kwargs):
        return self.transformer.cached_forward(x,*args,**kwargs)

    def patched_forward(self,x,*args,**kwargs):
        return self.transformer.patched_forward(x,*args,**kwargs)