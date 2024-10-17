import torch
import torch.nn as nn
import torch.nn.functional as F

class MaskedEncoder(nn.Module):
    def __init__(self, net,random_masks_func,**kwargs):
        super().__init__()
        self.net = net
        self.random_masks_func=random_masks_func
        if "mask_channels" in kwargs:
            self.mask_channels=kwargs["mask_channels"]
            if self.mask_channels is not None:
                self.mask_channels=torch.tensor(self.mask_channels,dtype=bool)
        else:
            self.mask_channels=None
        if "input_mask" in kwargs:
            self.input_mask=kwargs["input_mask"]
        else:
            self.input_mask=False

    def forward(self, x):
        x = self.net(x)
        return x

    def get_masked_x(self,data):
        x=data["x"]
        batch_size=x.shape[0]
        masks=self.random_masks_func(batch_size).to(x.device)
        if self.mask_channels is not None:
            masks1d=masks
            masks=masks[:,:,None]*self.mask_channels[None,None,:].to(dtype=masks.dtype,device=masks.device)
        else:
            masks1d=masks
        x_masked = x.clone()
        x_masked[masks] = 0.
        if self.input_mask:
            x_masked = torch.cat([x_masked, masks1d.to(dtype=x_masked.dtype)[:,:,None]], dim=-1)
        return x_masked,masks

    def masked_pred(self, data):
        x=data["x"]
        x_masked,masks=self.get_masked_x(data)
        data["x"]=x_masked
        x_pred = self(data)
        data["x"]=x
        if self.input_mask:
            x_pred=x_pred[...,:-1]
        return x[masks], x_pred[masks]
    
    def get_loss(self,data):
        x, x_pred = self.masked_pred(data)
        loss = F.mse_loss(x, x_pred)
        return loss