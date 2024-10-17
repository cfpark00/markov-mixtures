import torch
import tqdm
import numpy as np
import time
import random

def to_np(ten):
    return ten.detach().cpu().numpy()

def seed_all(seed,deterministic_algos=False):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic=True
    torch.backends.cudnn.benchmark=False
    np.random.seed(seed)
    random.seed(seed)
    if deterministic_algos:
        torch.use_deterministic_algorithms()

def train(model,dl_tr,batch_to_kwargs,n_steps,
          get_lr=False,clip_grad_norm=0.01,
          callback_steps=[],callbacks=[],
          callbacks_before_every_step=[],callbacks_after_every_step=[],
          device=None,
          verbose=1,
          **kwargs):
    if device is None:
        device=next(model.parameters()).device
    if kwargs.get("wandb",None) is not None:
        import wandb
    if len(callback_steps)>0:
        if type(callback_steps[0])!=list:
            any_callback_steps=set(callback_steps)
            callback_steps=[callback_steps]*len(callbacks)
        else:
            any_callback_steps=set([step_ for callback_steps_ in callback_steps for step_ in callback_steps_ ])
    else:
        any_callback_steps=set()
    online_losses=[]
    step=0

    model.to(device)
    model.optimizer.zero_grad()
    pbar=tqdm.tqdm(total=n_steps,desc="Training",disable=verbose==0)
    time_tr=0
    time_callbacks=0
    timer=time.perf_counter()
    try:
        while True:
            model.train()
            for batch in dl_tr:
                if get_lr:
                    lr = model.get_lr(step)
                    for param_group in model.optimizer.param_groups:
                        param_group['lr'] = lr
                for callback in callbacks_before_every_step:
                    callback(step=step,model=model)
                loss=model.get_loss(**batch_to_kwargs(batch))
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad_norm)
                model.optimizer.step()
                model.optimizer.zero_grad(set_to_none=True)

                online_losses.append(loss.item())
                if kwargs.get("wandb",None) is not None:
                    wandb.log({"loss":online_losses[-1],"step":step})
                for callback in callbacks_after_every_step:
                    callback(step=step,model=model)
                step+=1
                pbar.update()
                
                #sparse callbacks
                if step in any_callback_steps:
                    time_tr+=time.perf_counter()-timer
                    timer=time.perf_counter()
                    model.eval()
                    for callback_steps_,callback in zip(callback_steps,callbacks):
                        if step in callback_steps_:
                            callback(step=step,model=model)
                    model.train()
                    time_callbacks+=time.perf_counter()-timer
                    timer=time.perf_counter()
                if step==n_steps:
                    break
            if step==n_steps:
                break
        pbar.close()
    except KeyboardInterrupt:
        pbar.close()

    return_dict={
        "callback_steps":callback_steps,
        "online_losses":online_losses,
        "time_tr":time_tr,
        "time_callbacks":time_callbacks
    }
    return return_dict

def get_sqrt_steps(n_steps,n_start=10,n=100):
    steps=(np.linspace(np.sqrt(n_start),np.sqrt(n_steps),n)**2).astype(int).tolist()
    return steps