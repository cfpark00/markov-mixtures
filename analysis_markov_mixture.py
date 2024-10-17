from mltools.utils import cuda_tools
import torch
import matplotlib.pyplot as plt
import numpy as np
import os
import copy
import glob
import argparse
import tqdm

import utils

torch.set_float32_matmul_precision('high')

def get_args():
    parser=argparse.ArgumentParser()
    parser.add_argument("yaml_path",type=str,help="Path to the configuration file")
    parser.add_argument("mode",type=str,default="kl",help="Mode of analysis",choices=["kl","loss"])
    parser.add_argument("set_name",type=str,default="all",help="Name of the set to analyze",choices=["all","tt","train_shuffled","test_shuffled","train","test","structured","train_all_shuffled","train_context_shuffled","train_last_shuffled","test_all_shuffled","test_context_shuffled","test_last_shuffled"])
    parser.add_argument("n_context",type=int,default=-1,help="context length, if -1, all context lengths are used")
    parser.add_argument("i_ckpt",type=int,default=-1,help="Index of the checkpoint to load, -1 for last, -2 for all")
    parser.add_argument("suffix",type=str,default="",help="Suffix to add to the results file")
    args=parser.parse_args()
    assert args.yaml_path.endswith(".yaml"), "Configuration file must be a .yaml file"
    return args


if __name__=="__main__":
    args=get_args()
    mode=args.mode
    yaml_path=args.yaml_path

    config=utils.load_config(yaml_path)
    #remove curriculum for analysis
    if "curriculum" in config["data_params"]:
        del config["data_params"]["curriculum"]

    if mode=="kl":
        ############ Hyperparameters ############
        n_rep=30
        randomizing_seed=2430234
        randomizing_seed_randomset=3145789 #only to make random set of Ts
        #########################################

        ############ Environment ############
        device=cuda_tools.get_freer_device()
        fol=config["exp_dir"]
        #####################################
        
        ############ Parse Arguments ############
        # Set name
        set_name=args.set_name
        if set_name=="all":
            set_names=["train","test","structured"]
        elif set_name=="tt":
            set_names=["train","test"]
        elif set_name=="train_shuffled":
            set_names=["train_all_shuffled","train_context_shuffled","train_last_shuffled"]
        elif set_name=="test_shuffled":
            set_names=["test_all_shuffled","test_context_shuffled","test_last_shuffled"]
        elif set_name=="train_all_shuffled":
            set_names=["train_all_shuffled"]
        elif set_name=="train_context_shuffled":
            set_names=["train_context_shuffled"]
        elif set_name=="train_last_shuffled":
            set_names=["train_last_shuffled"]
        else:
            set_names=[set_name]
        
        # Context length
        n_context=args.n_context
        if n_context==-2:
            n_contexts=[1,10,40,70,100,130,160,190,220,250,280,310,340,370,400]
        else:
            n_contexts=[n_context]

        # Checkpoint index and path
        ckpt_paths=utils.get_ckpt_paths(fol)
        ckpt_steps=list(ckpt_paths.keys())
        i_ckpt=args.i_ckpt
        if i_ckpt==-2:
            i_ckpts=range(len(ckpt_steps))
        elif i_ckpt==-1:
            i_ckpts=[-1]
        else:
            i_ckpts=[i_ckpt]
        steps=[int(ckpt_steps[i]) for i in i_ckpts]

        # Save suffix
        suffix=args.suffix
        if suffix!="":
            suffix="_"+suffix
        ##########################################

        ############ Setup ############
        # Setup Model
        model=utils.get_model(config, device)
        model=torch.compile(model)

        #Setup original dataset
        ks=torch.tensor(config["data_params"]["ks"])
        assert len(torch.unique(ks))==1
        k=ks[0]
        priors=torch.tensor(config["data_params"]["ps"])
        dataset=utils.get_dataset(config)
        for d in dataset.datasets:
            d.online_permute=False
        Ts=torch.tensor(np.stack([d.T for d in dataset.datasets],axis=0))
        pis=torch.tensor(np.array([utils.get_stationary_distribution(T.cpu().numpy()) for T in Ts]))
        markov_cross_entropies=torch.tensor([utils.get_markov_cross_entropy(T) for T in Ts])

        ###Setup random dataset
        config_random=copy.deepcopy(config)
        np.random.seed(randomizing_seed_randomset)
        for i in range(len(config_random["data_params"]["ks"])):
            config_random["data_params"]["transition_paramss"][i]["random_state"]=np.random.randint(1000000)
        dataset_random=utils.get_dataset(config_random)
        Ts_random=torch.tensor(np.stack([d.T for d in dataset_random.datasets],axis=0))
        ################################

        method_names=["GT","bayes_ul_up","bayes_ul_bp","bayes_bl_up","bayes_bl_bp","icl_u","icl_b"]
        for set_name in set_names:
            Ts_star=[]
            contexts=[]

            Ts_bayes_ul_up=[]
            Ts_bayes_ul_bp=[]
            Ts_bayes_bl_up=[]
            Ts_bayes_bl_bp=[]
            Ts_icl_u=[]
            Ts_icl_b=[]
            Ts_model=[]#only this and kl will have a steps dimension

            np.random.seed(randomizing_seed)#master seed
            randseeds=np.random.randint(3000000,size=n_rep, dtype=int)
            for i_rep in tqdm.trange(n_rep,desc=f"Calc. KL {set_name}"):
                np.random.seed(randseeds[i_rep])#seed for this rep
                #Draw dataset
                if "train" in set_name:
                    i_sel=np.random.choice(len(Ts),p=priors.numpy())
                    dataset_star=dataset.datasets[i_sel]
                else:
                    structured=set_name=="structured"
                    rs=np.random.randint(1000000)
                    dataset_star=utils.get_markov_dataset_single(k=k,l=2*n_contexts[-1],structured=structured,random_state=rs)
                T_star=torch.tensor(dataset_star.T)
                #pi_star=torch.tensor(utils.get_stationary_distribution(T_star))
                seq_single,i_last_locs=utils.sample_seq_all_last(dataset=dataset_star,k=k,max_context=n_contexts[-1])
                check_sequential_last=True
                if "all_shuffled" in set_name:
                    #shuffle all
                    inds=torch.randperm(len(seq_single))
                    seq_single=seq_single[inds]
                    check_sequential_last=False
                elif "context_shuffled" in set_name:
                    #shuffle everything except i_last_locs
                    inds_bool=np.ones(len(seq_single),dtype=bool)
                    inds_bool[i_last_locs]=False
                    inds_shuffle=np.arange(len(seq_single))[inds_bool]
                    np.random.shuffle(inds_shuffle)
                    seq_single[inds_bool]=seq_single[inds_shuffle]
                elif "last_shuffled" in set_name:
                    #shuffle only i_last_locs
                    inds=torch.randperm(len(i_last_locs))
                    seq_single[i_last_locs]=seq_single[i_last_locs][inds]
                    check_sequential_last=False
                Ts_star.append(T_star)

                contexts_,loglikess_u,loglikess_b=utils.get_aligned_contexts(seq=seq_single,k=k,n_contexts=n_contexts,i_last_locs=i_last_locs,get_loglikess=True,Ts=Ts,check_sequential_last=check_sequential_last)
                logposterior_u=torch.log(priors)+loglikess_u
                posterior_u=torch.softmax(logposterior_u,dim=-1)
                logposterior_b=torch.log(priors)+loglikess_b
                posterior_b=torch.softmax(logposterior_b,dim=-1)
                contexts.append(contexts_)

                T_bayes_ul_up=[]
                T_bayes_ul_bp=[]
                T_bayes_bl_up=[]
                T_bayes_bl_bp=[]
                for contexts__,posterior_u,posterior_b in zip(contexts_,posterior_u,posterior_b):
                    ps_ul_up=utils.get_ps_bayes_up(posterior_u,pis);T_bayes_ul_up.append(ps_ul_up)
                    ps_ul_bp=utils.get_ps_bayes_bp(contexts__,posterior_u,Ts);T_bayes_ul_bp.append(ps_ul_bp)
                    ps_bl_up=utils.get_ps_bayes_up(posterior_b,pis);T_bayes_bl_up.append(ps_bl_up)
                    ps_bl_bp=utils.get_ps_bayes_bp(contexts__,posterior_b,Ts);T_bayes_bl_bp.append(ps_bl_bp)
                T_bayes_ul_up=torch.stack(T_bayes_ul_up,dim=0);Ts_bayes_ul_up.append(T_bayes_ul_up)
                T_bayes_ul_bp=torch.stack(T_bayes_ul_bp,dim=0);Ts_bayes_ul_bp.append(T_bayes_ul_bp)
                T_bayes_bl_up=torch.stack(T_bayes_bl_up,dim=0);Ts_bayes_bl_up.append(T_bayes_bl_up)
                T_bayes_bl_bp=torch.stack(T_bayes_bl_bp,dim=0);Ts_bayes_bl_bp.append(T_bayes_bl_bp)

                T_icl_u=[]
                T_icl_b=[]
                for contexts__ in contexts_:
                    ps_icl_b=utils.get_ps_icl_b(contexts__,k);T_icl_b.append(ps_icl_b)
                    #compute icl_u from ps_icl_b (which is T_icl_b)
                    pis_icl=utils.get_stationary_distribution(ps_icl_b)
                    ps_icl_u=pis_icl.unsqueeze(0).repeat(k,1);T_icl_u.append(ps_icl_u)
                T_icl_u=torch.stack(T_icl_u,dim=0);Ts_icl_u.append(T_icl_u)
                T_icl_b=torch.stack(T_icl_b,dim=0);Ts_icl_b.append(T_icl_b)

                #model
                for step in steps:
                    ckpt_path=ckpt_paths[step]
                    model.load_state_dict_nc(torch.load(ckpt_path))
                    model.eval()

                    two_token_input=config["data_params"].get("two_token_input",False)
                    T_model=[]
                    for contexts__ in contexts_:
                        ps_model=utils.get_ps_model(model,contexts__,k=k,two_token_input=two_token_input);T_model.append(ps_model)
                    T_model=torch.stack(T_model,dim=0);Ts_model.append(T_model)

            n_ckpts=len(steps)
            n_n_contexts=len(n_contexts)

            Ts_star=torch.stack(Ts_star,dim=0).reshape(n_rep,1,k,k).repeat(1,n_n_contexts,1,1)#temporary for kl

            Ts_bayes_ul_up=torch.stack(Ts_bayes_ul_up,dim=0).reshape(n_rep,n_n_contexts,k,k)
            Ts_bayes_ul_bp=torch.stack(Ts_bayes_ul_bp,dim=0).reshape(n_rep,n_n_contexts,k,k)
            Ts_bayes_bl_up=torch.stack(Ts_bayes_bl_up,dim=0).reshape(n_rep,n_n_contexts,k,k)
            Ts_bayes_bl_bp=torch.stack(Ts_bayes_bl_bp,dim=0).reshape(n_rep,n_n_contexts,k,k)
            Ts_icl_u=torch.stack(Ts_icl_u,dim=0).reshape(n_rep,n_n_contexts,k,k)
            Ts_icl_b=torch.stack(Ts_icl_b,dim=0).reshape(n_rep,n_n_contexts,k,k)

            Ts_model=torch.stack(Ts_model,dim=0).reshape(n_rep,n_ckpts,n_n_contexts,k,k)#i_rep,i_ckpt,i_n_contexts,k,k

            #calculate kl
            kl_methods_methods=[]
            T_methods=[Ts_star,Ts_bayes_ul_up,Ts_bayes_ul_bp,Ts_bayes_bl_up,Ts_bayes_bl_bp,Ts_icl_u,Ts_icl_b]
            for i_n_context in range(n_n_contexts):
                for T1 in T_methods:
                    for T2 in T_methods:
                        kl_methods_methods.append(utils.get_markov_kl_batched(T1[:,i_n_context],T2[:,i_n_context],batched="both"))
            kl_methods_methods=torch.stack(kl_methods_methods,dim=0).reshape(n_n_contexts,len(T_methods),len(T_methods),n_rep)
            kl_methods_methods=kl_methods_methods.permute(3,0,1,2)#i_rep,i_n_contexts,method,method
            
            kl_methods_model=[]
            kl_model_methods=[]
            for i_step in range(n_ckpts):
                for i_n_context in range(n_n_contexts):
                    Ts_model_=Ts_model[:,i_step,i_n_context]#(n_rep,k,k)
                    for T_method in T_methods:
                        kl_methods_model.append(utils.get_markov_kl_batched(T_method[:,i_n_context],Ts_model_,batched="both"))
                        kl_model_methods.append(utils.get_markov_kl_batched(Ts_model_,T_method[:,i_n_context],batched="both"))
            kl_methods_model=torch.stack(kl_methods_model,dim=0).reshape(n_ckpts,n_n_contexts,len(T_methods),n_rep)
            kl_methods_model=kl_methods_model.permute(3,0,1,2)#i_rep,i_ckpt,i_n_contexts,method
            kl_model_methods=torch.stack(kl_model_methods,dim=0).reshape(n_ckpts,n_n_contexts,len(T_methods),n_rep)
            kl_model_methods=kl_model_methods.permute(3,0,1,2)#i_rep,i_ckpt,i_n_contexts,method

            Ts_star=Ts_star[:,0]#remove n_contexts dimension

            ##Implementation and memoriaztion
            if "train" not in set_name:#these metrics are not well defined on the training set
                kl_gt_model=kl_methods_model[:,:,:,0]
                #i_rep,i_ckpt,i_n_contexts,k,k
                kl_trainset_model=[]
                for i_rep in range(n_rep):
                    for i_ckpt in range(n_ckpts):
                        for i_n_context in range(n_n_contexts):
                            T_model=Ts_model[i_rep,i_ckpt,i_n_context]
                            kl_trainset_model_=utils.get_markov_kl_batched(Ts,T_model,batched="T_hat")
                            kl_trainset_model.append(kl_trainset_model_)
                kl_trainset_model=torch.stack(kl_trainset_model,dim=0).reshape(n_rep,n_ckpts,n_n_contexts,len(Ts))
                kl_randomset_model=[]
                for i_rep in range(n_rep):
                    for i_ckpt in range(n_ckpts):
                        for i_n_context in range(n_n_contexts):
                            T_model=Ts_model[i_rep,i_ckpt,i_n_context]
                            kl_randomset_model_=utils.get_markov_kl_batched(Ts_random,T_model,batched="T_hat")
                            kl_randomset_model.append(kl_randomset_model_)
                kl_randomset_model=torch.stack(kl_randomset_model,dim=0).reshape(n_rep,n_ckpts,n_n_contexts,len(Ts_random))
                Ts_uni=torch.stack([utils.get_stationary_T(T) for T in Ts],dim=0)
                kl_trainset_model_uni=[]
                for i_rep in range(n_rep):
                    for i_ckpt in range(n_ckpts):
                        for i_n_context in range(n_n_contexts):
                            T_model=Ts_model[i_rep,i_ckpt,i_n_context]
                            kl_trainset_model_uni_=utils.get_markov_kl_batched(Ts_uni,T_model,batched="T_hat")
                            kl_trainset_model_uni.append(kl_trainset_model_uni_)
                kl_trainset_model_uni=torch.stack(kl_trainset_model_uni,dim=0).reshape(n_rep,n_ckpts,n_n_contexts,len(Ts_uni))
                Ts_random_uni=torch.stack([utils.get_stationary_T(T) for T in Ts_random],dim=0)
                kl_randomset_model_uni=[]
                for i_rep in range(n_rep):
                    for i_ckpt in range(n_ckpts):
                        for i_n_context in range(n_n_contexts):
                            T_model=Ts_model[i_rep,i_ckpt,i_n_context]
                            kl_randomset_model_uni_=utils.get_markov_kl_batched(Ts_random_uni,T_model,batched="T_hat")
                            kl_randomset_model_uni.append(kl_randomset_model_uni_)
                kl_randomset_model_uni=torch.stack(kl_randomset_model_uni,dim=0).reshape(n_rep,n_ckpts,n_n_contexts,len(Ts_random_uni))
                Ts_star_uni=torch.stack([utils.get_stationary_T(T) for T in Ts_star],dim=0)
                #add context dimension
                Ts_star_uni=Ts_star_uni.reshape(n_rep,1,k,k).repeat(1,n_n_contexts,1,1)
                kl_gt_model_uni=[]
                for i_rep in range(n_rep):
                    for i_ckpt in range(n_ckpts):
                        T_model=Ts_model[i_rep,i_ckpt]
                        T_star_uni=Ts_star_uni[i_rep]
                        print(T_star_uni.shape,T_model.shape)
                        kl_gt_model_uni_=utils.get_markov_kl_batched(T_star_uni,T_model,batched="both")
                        kl_gt_model_uni.append(kl_gt_model_uni_)
                kl_gt_model_uni=torch.stack(kl_gt_model_uni,dim=0).reshape(n_rep,n_ckpts,n_n_contexts)

            method_names_plot=["Bayes UL UP","Bayes UL BP","Bayes BL UP","Bayes BL BP","ICL U","ICL B"]
            method_colors=["blue","green","orange","purple","brown","pink"]
            model_color="red"
            #plot with x axis context
            avg_kl_model_gt=kl_model_methods[:,-1,:,0].mean(0)
            sem_kl_model_gt=kl_model_methods[:,-1,:,0].std(0)/np.sqrt(n_rep)
            avg_kl_methods_gt=kl_methods_methods[:,:,1:,0].mean(0)#T2 is GT
            sem_kl_methods_gt=kl_methods_methods[:,:,1:,0].std(0)/np.sqrt(n_rep)

            plt.figure(figsize=(10,7))
            for i_method in range(6):
                plt.errorbar(n_contexts,avg_kl_methods_gt[:,i_method],yerr=sem_kl_methods_gt[:,i_method],label=method_names_plot[i_method],color=method_colors[i_method])
            plt.errorbar(n_contexts,avg_kl_model_gt,yerr=sem_kl_model_gt,label="Model",color=model_color)
            plt.legend(loc="upper right")
            plt.xlabel("Context Length")
            plt.ylabel("KL divergence")
            fig_path=os.path.join(fol,f"kl_vs_n_context_{set_name}{suffix}.png")
            plt.savefig(fig_path)
            plt.close()

            #plot with x axis steps
            avg_kl_model_gt=kl_model_methods[:,:,-1,0].mean(0)
            sem_kl_model_gt=kl_model_methods[:,:,-1,0].std(0)/np.sqrt(n_rep)
            avg_kl_methods_gt=kl_methods_methods[:,-1,1:,0].mean(0)#T2 is GT, this doesn't depend on step, plot as line
            sem_kl_methods_gt=kl_methods_methods[:,-1,1:,0].std(0)/np.sqrt(n_rep)

            plt.figure(figsize=(10,7))
            plt.errorbar(steps,avg_kl_model_gt,yerr=sem_kl_model_gt,label="Model",color=model_color)
            xlims=plt.gca().get_xlim()
            for i_method in range(6):
                m=avg_kl_methods_gt[i_method]
                sem=sem_kl_methods_gt[i_method]
                plt.axhline(m,xmin=xlims[0],xmax=xlims[1],label=method_names_plot[i_method],color=method_colors[i_method])
                plt.fill_between(x=steps,y1=m-sem,y2=m+sem,alpha=0.3,color=method_colors[i_method])
            plt.legend(loc="upper right")
            plt.xscale("log")
            plt.xlabel("Steps")
            plt.ylabel("KL divergence")
            fig_path=os.path.join(fol,f"kl_vs_steps_{set_name}{suffix}.png")
            plt.savefig(fig_path)
            plt.close()


            results={}
            results["steps"]=torch.tensor(steps)
            results["n_contexts"]=n_contexts
            results["Ts"]=Ts
            results["pis"]=pis
            results["priors"]=priors
            results["markov_cross_entropies"]=markov_cross_entropies

            ####
            results["Ts_star"]=Ts_star

            results["Ts_bayes_ul_up"]=Ts_bayes_ul_up
            results["Ts_bayes_ul_bp"]=Ts_bayes_ul_bp
            results["Ts_bayes_bl_up"]=Ts_bayes_bl_up
            results["Ts_bayes_bl_bp"]=Ts_bayes_bl_bp
            results["Ts_icl_u"]=Ts_icl_u
            results["Ts_icl_b"]=Ts_icl_b
            results["Ts_model"]=Ts_model

            results["kl_methods_methods"]=kl_methods_methods
            results["kl_methods_model"]=kl_methods_model
            results["kl_model_methods"]=kl_model_methods

            if "train" not in set_name:
                results["kl_gt_model"]=kl_gt_model
                results["kl_trainset_model"]=kl_trainset_model
                results["kl_randomset_model"]=kl_randomset_model
                results["kl_trainset_model_uni"]=kl_trainset_model_uni
                results["kl_randomset_model_uni"]=kl_randomset_model_uni
                results["kl_gt_model_uni"]=kl_gt_model_uni

            results_path=os.path.join(fol,f"results_{set_name}{suffix}.pth")
            torch.save(results,results_path)

    elif mode=="loss":
        test_seed=12034123

        device=cuda_tools.get_freer_device()
        fol=config["exp_dir"]

        train_results=torch.load(os.path.join(fol,"logs.pth"))
        save_steps=config["training_params"]["save_steps"]
        online_losses=train_results["online_losses"]
        val_losses=train_results["val_losses"]

        #loss plot
        plt.figure(figsize=(8,5))
        plt.plot(online_losses)
        plt.plot(save_steps[:len(val_losses)], val_losses)
        plt.yscale("log")
        plt.xscale("log")
        plt.xlim(left=100)
        plt.xlabel("Steps")
        plt.ylabel("Cross Entropy")
        fig_path=os.path.join(fol,"losses.png")
        plt.savefig(fig_path)
        plt.close()

        ckpt_paths=utils.get_ckpt_paths(fol)
        ckpt_steps=np.array(list(ckpt_paths.keys()))
        steps=ckpt_steps
        ckpts={}
        for i_step,step in enumerate(steps):
            ckpt=torch.load(ckpt_paths[step],map_location=device)
            ckpts[step]=ckpt
        model=utils.get_model(config,device=device)

        mask_loss=config["mask_loss"]
        if config["data_params"].get("two_token_input",False):
            k=config["data_params"]["ks"][0]
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

        dl_val=utils.get_dataloader(config)
        config_test=copy.deepcopy(config)

        #dl_val_test
        np.random.seed(test_seed)
        n=len(config_test["data_params"]["ks"])
        ps=np.random.random(n)
        ps=ps/ps.sum()
        rss=np.random.randint(0,100000,n).tolist()
        config_test["data_params"]["ps"]=list(ps)
        for i in range(n):
            config_test["data_params"]["transition_paramss"][i]["random_state"]=rss[i]
        dl_test=utils.get_dataloader(config_test)


        val_losses_file=f"{fol}/val_losses.pth"
        test_losses_file=f"{fol}/test_losses.pth"

        val_losses={}
        test_losses={}
        for step,ckpt in tqdm.tqdm(ckpts.items()):
            model.load_state_dict_nc(ckpt)#no compile
            with torch.no_grad():
                model.eval()
                val_loss=model.get_loss(**batch_to_kwargs(next(iter(dl_val))))
                val_losses[step]=val_loss.item()
                test_loss=model.get_loss(**batch_to_kwargs(next(iter(dl_test))))
                test_losses[step]=test_loss.item()
        torch.save(val_losses,val_losses_file)
        torch.save(test_losses,test_losses_file)