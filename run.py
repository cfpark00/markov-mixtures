import torch
import utils
import argparse
import os
import shutil
import math

from mltools import ml_utils
from mltools.utils import cuda_tools

def get_args():
    parser=argparse.ArgumentParser()
    parser.add_argument("yaml_path",type=str,help="Path to the configuration file")
    args=parser.parse_args()
    assert args.yaml_path.endswith(".yaml"), "Configuration file must be a .yaml file"
    return args


if __name__=="__main__":
    yaml_path=get_args().yaml_path
    config=utils.load_config(yaml_path)
    if config["wandb"] is not None:
        import wandb
        wandb.init(project=config["wandb"]["project"],
        name=config["wandb"]["name"],
        config=config,
        group=config["wandb"].get("group",None),
        )

    device=cuda_tools.get_freer_device()

    exp_dir=config["exp_dir"]
    if os.path.exists(exp_dir):
        shutil.rmtree(exp_dir)
    os.makedirs(exp_dir,exist_ok=True)

    #copy config
    config_path=os.path.join(exp_dir,"config.yaml")
    shutil.copy(yaml_path,config_path)

    seed=config["seed"]
    ml_utils.seed_all(seed)

    if "device" in config:
        device=torch.device(config["device"])

    task=config["task"]
    if True:
        dl_tr=utils.get_dataloader(config)
        if task=="star_graph":
            dl_val=utils.get_dataloader(config,split="val")
            n_val=config["data_params"].get("n_val",100)
        else:
            dl_val=dl_tr
            n_val=config["data_params"].get("n_val",100)
        model=utils.get_model(config,device)
        model=torch.compile(model)

        n_steps=config["training_params"]["n_steps"]
        save_steps=config["training_params"]["save_steps"]
        mask_loss=config.get("mask_loss",False)

        ckpt_dir=os.path.join(exp_dir,"checkpoints")
        os.makedirs(ckpt_dir,exist_ok=True)

        k=config["data_params"]["ks"][0]
        if config["data_params"].get("two_token_input",False):
            def batch_to_kwargs(batch):
                x,loss_mask=batch
                if mask_loss:
                    return {"x":x.to(device),"loss_mask":loss_mask.to(device),"two_token_input":True,"k":k}
                else:
                    return {"x":x.to(device),"two_token_input":True,"k":k}
        else:
            def batch_to_kwargs(batch):
                x,loss_mask=batch
                if mask_loss:
                    return {"x":x.to(device),"loss_mask":loss_mask.to(device)}
                else:
                    return {"x":x.to(device)}
        val_losses=[]
        ckpt_paths=[]
        def get_val_loss(step,model):
            model.eval()
            with torch.no_grad():
                if n_val is None:
                    loss_avg=0
                    for batch in dl_val:
                        loss=model.get_loss(**batch_to_kwargs(batch))
                        loss_avg+=loss.item()
                    loss_avg/=len(dl_val)
                    val_losses.append(loss_avg)
                else:
                    loss_avg=0
                    for i,batch in enumerate(dl_val):
                        if i>=n_val:
                            break
                        loss=model.get_loss(**batch_to_kwargs(batch))
                        loss_avg+=loss.item()
                    loss_avg/=n_val
                    val_losses.append(loss_avg)
            if config["wandb"] is not None:
                wandb.log({"val_loss":val_losses[-1],"step":step})
        def save_ckpt(step,model):
            ckpt_path=os.path.join(ckpt_dir,f"ckpt_step={step}.pth")
            torch.save(model.state_dict(),ckpt_path)
            if "save_opt" in config["training_params"] and config["training_params"]["save_opt"]:
                opt_ckpt_path=os.path.join(ckpt_dir,f"opt_ckpt_step={step}.pth")
                torch.save(model.optimizer.state_dict(),opt_ckpt_path)
            ckpt_paths.append(ckpt_path)
        if "curriculum" in config["data_params"]:
            if config["data_params"]["curriculum"]["type"]=="div":
                n_div=config["data_params"]["curriculum"]["n_div"]
                n_steps=config["training_params"]["n_steps"]
                change_intv=n_steps//n_div
                i_curriculum_dict={}
                for i in range(n_div):
                    i_curriculum_dict[i*change_intv]=i
                print(n_steps,i_curriculum_dict)
                def change(step,model):
                    i=i_curriculum_dict[step]
                    print(f"Changing curriculum to {i}")
                    dl_tr.dataset.set_curriculum(i)
                callbacks=[get_val_loss,save_ckpt,change]
                callback_steps=[save_steps,save_steps,list(i_curriculum_dict.keys())]
        else:
            callbacks=[get_val_loss,save_ckpt]
            callback_steps=save_steps
        train_results=ml_utils.train(
            model=model,
            dl_tr=dl_tr,
            batch_to_kwargs=batch_to_kwargs,
            n_steps=n_steps,
            callback_steps=callback_steps,
            callbacks=callbacks,
            get_lr=hasattr(model,"get_lr"),
            device=device,
            verbose=True,
            wandb=config.get("wandb",None),
            )
        train_results["save_steps"]=save_steps
        train_results["val_losses"]=val_losses
        train_results["ckpt_paths"]=ckpt_paths

        #other must save(e.g. training set)
        if task=="star_graph":
            edgess=dl_tr.dataset.get_edgess()
            train_results["edgess"]=edgess
        elif task=="markov":
            train_results["markov_T"]=dl_tr.dataset.T

        train_results_path=os.path.join(exp_dir,"logs.pth")
        torch.save(train_results,train_results_path)
    else:
        raise ValueError("Invalid task")
    if config["wandb"] is not None:
        wandb.finish()
    print("Done!")
