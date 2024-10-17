import torch
import numpy as np
import yaml
import os
import shutil
import glob
import math

from mltools.models import configs

import models
import utils


def load_config(yaml_file):
    config=yaml.safe_load(open(yaml_file))
    return config

#LinReg
class LinRegDataset(torch.utils.data.IterableDataset):
    def __init__(self, d,n,vector_output=False,w=None):
        self.d=d
        self.n=n
        self.vector_output=vector_output
        if w is not None:
            if isinstance(w,list):
                self.w=torch.tensor(w).reshape(d,1)
            elif isinstance(w,torch.Tensor):
                self.w=w.reshape(d,1)
            else:
                raise ValueError("w must be a list or a torch.Tensor")
        else:
            self.w=None
    
    def get_loss_mask(self):
        if self.vector_output:
            y_inds=torch.zeros(self.n*2,dtype=torch.bool)
            y_inds[torch.arange(1,self.n*2,2)]=True
        else:
            y_inds=torch.zeros(self.n*(self.d+1),dtype=torch.bool)
            y_inds[torch.arange(self.d,self.n*(self.d+1),self.d+1)]=True
        return y_inds

    def get_data(self,d=None,n=None,return_w=False):
        if d is None:
            d=self.d
        if n is None:
            n=self.n
        if self.w is not None:
            w=self.w
        else:
            w=torch.randn(d,1)
        x=torch.randn(n,d)
        y=x@w#n,1
        loss_mask=self.get_loss_mask()
        if self.vector_output:
            y=y.repeat(1,d)
            data=torch.stack([x,y],dim=1).reshape(n*2,d)#(n,2,d)->(n*2,d)
            if return_w:
                return data,loss_mask,w
            return data,loss_mask
        else:
            data=torch.cat([x,y],dim=1).reshape(n*(d+1),1)
            if return_w:
                return data,loss_mask,w
            return data,loss_mask
        
    def get_seq_len(self):
        if self.vector_output:
            return self.n*2
        else:
            return self.n*(self.d+1)

    def __iter__(self):
        while True:
            yield self.get_data()

# Binary
class BinaryRepeatDataset(torch.utils.data.IterableDataset):
    def __init__(self,l_min,l_max,n):
        self.l_min=l_min
        self.l_max=l_max
        self.n=n
        self.values=np.array(["0","1"])
        self.one_len=(3+2*self.l_max)
        self.seq_len=self.n*self.one_len
        self.tokenizer=Tokenizer()

    def get_data(self,return_seq=False):
        ls=np.random.randint(self.l_min,self.l_max+1,self.n)
        
        seq=[]
        loss_mask=[]
        for l in ls:
            seq_=[]
            seq_.append("DO")
            loss_mask.append(False)
            vals=self.values[np.random.randint(0,2,l)].tolist()
            seq_.extend(vals)
            loss_mask.extend([False]*len(vals))
            seq_.append("REP")
            loss_mask.append(False)
            seq_.extend(vals)
            loss_mask.extend([True]*len(vals))
            seq_.append("DONE")
            loss_mask.append(True)
            n_extra=self.one_len-len(seq_)
            seq_.extend(["NULL"]*n_extra)
            loss_mask.extend([False]*n_extra)
            assert len(seq_)==self.one_len
            seq.extend(seq_)
        assert len(seq)==self.seq_len
        assert len(loss_mask)==self.seq_len
        if return_seq:
            return seq,loss_mask
        return torch.tensor(self.tokenizer.encode(seq),dtype=torch.long),torch.tensor(loss_mask,dtype=torch.bool)
    
    def get_seq_len(self):
        return self.seq_len

    def __iter__(self):
        while True:
            yield self.get_data()
    
def xor2(vals,inner=True):
    vals=np.array(vals)
    odd=len(vals)%2==1
    if odd:
        last=[vals[-1]]
        vals=vals[:-1]
    is1=(vals=="1")
    if inner:
        is1=is1.reshape(-1,2)
        is1=np.logical_xor(is1[:,0],is1[:,1]).astype(int)
    else:
        is1=is1.reshape(2,-1)
        is1=np.logical_xor(is1[0],is1[1]).astype(int)
    return np.array(["0","1"])[is1].tolist()

def npattern(vals,pattern):
    pattern=np.array(pattern)
    l=len(pattern)
    count=0
    for i in range(len(vals)-(l-1)):
        if all(vals[i:i+l]==pattern):
            count+=1
    return ["1"]*count
    
def sum_halves(vals):
    vals=np.array(vals)
    first_half=vals[:len(vals)//2]
    second_half=vals[len(vals)//2:]
    is1_f=(first_half=="1")
    val_f=((2**np.arange(len(is1_f))[::-1])*is1_f ).sum()
    #print(val_f,"{0:b}".format(val_f))
    is1_s=(second_half=="1")
    val_s=((2**np.arange(len(is1_s))[::-1])*is1_s ).sum()
    #print(val_s,"{0:b}".format(val_s))
    val_sum=val_f+val_s
    #print(val_sum,"{0:b}".format(val_sum))
    #binarize
    val_sum_bin=[]
    while val_sum>0:
        val_sum_bin.append(str(val_sum%2))
        val_sum//=2
    if len(val_sum_bin)==0:
        val_sum_bin=["0"]
    #print(val_sum_bin[::-1])
    return val_sum_bin[::-1]


task_funcs={
    "REP":lambda vals:vals,
    "INV":lambda vals:[str(1-int(v)) for v in vals],
    "REV":lambda vals:vals[::-1],
    "SUM":lambda vals:["1"]*sum([int(v) for v in vals]),
    "PAR":lambda vals: ["0" if sum([int(v) for v in vals])%2==0 else "1"],
    "RAND":lambda vals:np.random.choice(["0","1"],len(vals)).tolist(),
    "FHALF":lambda vals:vals[:len(vals)//2],
    "SHALF":lambda vals:vals[len(vals)//2:],
    "INXOR": lambda vals: xor2(vals,inner=True),
    "OUTXOR": lambda vals: xor2(vals,inner=False),
    "NTRIPLET111": lambda vals: npattern(vals,["1","1","1"]),
    "NTRIPLET010": lambda vals: npattern(vals,["0","1","0"]),
    "NDOUBLET11": lambda vals: npattern(vals,["1","1"]),
    "NDOUBLET01": lambda vals: npattern(vals,["0","1"]),
    "NQUAD1111": lambda vals: npattern(vals,["1","1","1","1"]),
    "NQUAD0101": lambda vals: npattern(vals,["0","1","0","1"]),
    "ALL1": lambda vals: ["1"]*len(vals),
    "ALL0": lambda vals: ["0"]*len(vals),
    "SUMHALVES": lambda vals: sum_halves(vals)
}

class BinarySingleTaskDataset(torch.utils.data.IterableDataset):
    def __init__(self,task_token,l_min,l_max,n,l_max_fix=None):
        self.task_token=task_token
        self.l_min=l_min
        self.l_max=l_max
        self.n=n
        if l_max_fix is not None:
            self.l_max_fix=l_max_fix
        else:
            self.l_max_fix=l_max
        self.values=np.array(["0","1"])
        self.one_len=(3+2*self.l_max_fix)#assuming that the output is always smaller or equal to the input
        self.seq_len=self.n*self.one_len
        self.tokenizer=Tokenizer()

    def get_data(self,return_seq=False):
        ls=np.random.randint(self.l_min,self.l_max+1,self.n)
        
        seq=[]
        loss_mask=[]
        for l in ls:
            seq_=[]
            seq_.append("DO")
            loss_mask.append(False)
            vals=self.values[np.random.randint(0,2,l)].tolist()
            seq_.extend(vals)
            loss_mask.extend([False]*len(vals))
            seq_.append(self.task_token)
            loss_mask.append(False)
            task_vals=task_funcs[self.task_token](vals)
            seq_.extend(task_vals)
            loss_mask.extend([True]*len(task_vals))#can be variable !=len(vals)
            seq_.append("DONE")
            loss_mask.append(True)
            n_extra=self.one_len-len(seq_)
            assert n_extra>=0
            seq_.extend(["NULL"]*n_extra)
            loss_mask.extend([False]*n_extra)
            assert len(seq_)==self.one_len
            seq.extend(seq_)
        assert len(seq)==self.seq_len
        assert len(loss_mask)==self.seq_len
        if return_seq:
            return seq,loss_mask
        return torch.tensor(self.tokenizer.encode(seq),dtype=torch.long),torch.tensor(loss_mask,dtype=torch.bool)
    
    def get_seq_len(self):
        return self.seq_len

    def __iter__(self):
        while True:
            yield self.get_data()

class BinaryMultiTaskDataset(torch.utils.data.IterableDataset):
    def __init__(self,task_tokens,task_ps,l_min,l_max,n,l_max_fix=None,same_in_sample=False):
        self.task_tokens=task_tokens
        self.task_ps=np.array(task_ps)#probs
        self.task_ps/=np.sum(self.task_ps)
        self.l_min=l_min
        self.l_max=l_max
        self.n=n
        if l_max_fix is not None:
            self.l_max_fix=l_max_fix
        else:
            self.l_max_fix=l_max
        self.same_in_sample=same_in_sample

        self.values=np.array(["0","1"])
        self.one_len=(3+2*self.l_max_fix)
        self.seq_len=self.n*self.one_len
        self.tokenizer=Tokenizer()

    def get_data(self,return_seq=False):
        ls=np.random.randint(self.l_min,self.l_max+1,self.n)
        
        seq=[]
        loss_mask=[]
        if self.same_in_sample:
            task_token=np.random.choice(self.task_tokens,p=self.task_ps)
        for l in ls:
            seq_=[]
            seq_.append("DO")
            loss_mask.append(False)
            vals=self.values[np.random.randint(0,2,l)].tolist()
            seq_.extend(vals)
            loss_mask.extend([False]*len(vals))
            if not self.same_in_sample:
                task_token=np.random.choice(self.task_tokens,p=self.task_ps)
            seq_.append(task_token)
            loss_mask.append(False)
            task_vals=task_funcs[task_token](vals)
            seq_.extend(task_vals)
            loss_mask.extend([True]*len(task_vals))#can be variable !=len(vals)
            seq_.append("DONE")
            loss_mask.append(True)
            n_extra=self.one_len-len(seq_)
            assert n_extra>=0
            seq_.extend(["NULL"]*n_extra)
            loss_mask.extend([False]*n_extra)
            assert len(seq_)==self.one_len
            seq.extend(seq_)
        assert len(seq)==self.seq_len
        assert len(loss_mask)==self.seq_len
        #print(seq)
        if return_seq:
            return seq,loss_mask
        return torch.tensor(self.tokenizer.encode(seq),dtype=torch.long),torch.tensor(loss_mask,dtype=torch.bool)

    def get_seq_len(self):
        return self.seq_len
    
    def __iter__(self):
        while True:
            yield self.get_data()


##### Star graph
def get_star_graph(k,l,names):
    n_nodes=k*l+1
    assert len(names)>=n_nodes
    nodes=np.random.choice(names,n_nodes,replace=False)
    start_node=nodes[0]
    arms=nodes[1:].reshape(l,k)
    edges_0=np.stack([np.full(k,start_node),arms[0,:]],axis=1)
    if l==0:
        edges=edges_0
    else:
        edges=np.stack([arms[:-1,:],arms[1:,:]],axis=-1).reshape(-1,2)
        edges=np.concatenate([edges_0,edges],axis=0)
    return edges

def get_star_graph_task_solution(edges,l):
    #assume the above function makes the graph element [0,0] is the start node
    edges_struct=edges.reshape(l,-1,2)
    k=edges_struct.shape[1]
    i_end=np.random.randint(0,k)
    seq=edges_struct[:,i_end,:]
    task=np.array([seq[0,0],seq[-1,1]])
    solution=np.array([seq[0,0],*seq[:,1]])
    return task,solution

class SingleStarGraphDataset(torch.utils.data.IterableDataset):
    def __init__(self,k,l,names,n,n_graphs,rev=False,randomize_edges=False):
        self.k=k
        self.l=l
        self.names=names
        if isinstance(self.names,int):
            self.names=[str(i) for i in range(self.names)]
        else:
            assert all([isinstance(name,str) for name in self.names])
        self.n_names=len(self.names)
        self.n=n
        self.n_graphs=n_graphs
        self.rev=rev
        self.randomize_edges=randomize_edges

        self.n_edges=self.k*self.l
        self.n_nodes=self.k*self.l+1
        self.one_len=1+(3*self.n_edges-1)+1+3+1  + (self.l+1)+1
        self.seq_len=self.n*self.one_len
        self.edgess=[]
        for i in range(self.n_graphs):
            self.edgess.append(get_star_graph(self.k,self.l,self.names))
        self.seqs=[]
        self.loss_masks=[]
        for i in range(self.n_graphs):
            edges=self.edgess[i]
            task,solution=get_star_graph_task_solution(edges,self.l)
            if self.randomize_edges:
                edges=np.random.permutation(edges)
            edges_=np.concatenate([edges,np.full((len(edges),1),"/")],axis=1).reshape(-1)[:-1]
            task_=[task[0],"TO",task[1]]
            solution_=solution if not self.rev else solution[::-1]
            loss_mask_=[]
            seq_=[]
            seq_.append("GRAPH")
            loss_mask_.append(False)
            seq_.extend(edges_)
            loss_mask_.extend([False]*len(edges_))
            seq_.append("DO")
            loss_mask_.append(False)
            seq_.extend(task_)
            loss_mask_.extend([False]*len(task_))
            seq_.append("FIND" if not self.rev else "FINDREV")
            loss_mask_.append(False)
            seq_.extend(solution_)
            loss_mask_.extend([True]*len(solution_))
            seq_.append("DONE")
            loss_mask_.append(True)
            assert len(seq_)==self.one_len
            assert len(loss_mask_)==self.one_len
            self.seqs.append(seq_)
            self.loss_masks.append(loss_mask_)

        self.tokenizer=Tokenizer()

    def get_n_tot(self):
        return math.perm(self.n_names,self.n_nodes)//math.factorial(self.k)
    
    def get_data(self,return_seq=False):
        #"GRAPH","/","FIND","TO"
        seq=[]
        loss_mask=[]
        i_edges=np.random.choice(self.n_graphs,self.n,replace=True)
        for i in range(self.n):
            seq.extend(self.seqs[i_edges[i]])
            loss_mask.extend(self.loss_masks[i_edges[i]])
        assert len(seq)==self.seq_len
        assert len(loss_mask)==self.seq_len

        if return_seq:
            return seq,loss_mask
        return torch.tensor(self.tokenizer.encode(seq),dtype=torch.long),torch.tensor(loss_mask,dtype=torch.bool)
    
    def get_seq_len(self):
        return self.seq_len

    def get_edgess(self):
        return self.edgess

    def __iter__(self):
        while True:
            yield self.get_data()

#### Synthetic Language
#TODO


### Binary String Generation

class BinaryStringDataset(torch.utils.data.IterableDataset):
    def __init__(self,l,method_name):
        self.l=l
        self.method_name=method_name
        assert self.method_name in ["GENBS1","GENBS2","GENBS3","GENBS4"]
        self.one_len=2+l
        self.seq_len=self.one_len

        if self.method_name=="GENBS1":
            def get_vals():
                return np.random.choice(["0","1"],self.l,replace=True).tolist()
        elif self.method_name=="GENBS2":
            assert self.l%2==0
            def get_vals():
                inds=np.random.choice(2,self.l//2,replace=True)
                vals=np.array([["0","1"],["1","0"]])[inds].flatten().tolist()
                return vals
        elif self.method_name=="GENBS3":
            assert self.l%2==0
            def get_vals():
                inds=np.random.choice(2,self.l//2,replace=True)
                vals=np.array([["0","0"],["1","1"]])[inds].flatten().tolist()
                return vals
        elif self.method_name=="GENBS4":
            assert self.l%4==0
            def get_vals():
                inds=np.random.choice(2,self.l//4,replace=True)
                vals=np.array([["0","0","0","0"],["1","1","1","1"]])[inds].flatten().tolist()
                return vals

            
        self.get_vals=get_vals

        self.tokenizer=Tokenizer()
    
    def get_data(self,return_seq=False):
        seq=[]
        loss_mask=[]
        seq.append("DO")
        loss_mask.append(False)
        seq.append(self.method_name)
        loss_mask.append(False)
        vals=self.get_vals()
        seq.extend(vals)
        loss_mask.extend([True]*len(vals))
        assert len(seq)==self.one_len
        assert len(loss_mask)==self.one_len
        if return_seq:
            return seq,loss_mask
        return torch.tensor(self.tokenizer.encode(seq),dtype=torch.long),torch.tensor(loss_mask,dtype=torch.bool)

    def get_seq_len(self):
        return self.seq_len

    def __iter__(self):
        while True:
            yield self.get_data()


###DFA

from typing import Tuple, Dict
from pythomata import SimpleDFA

class DFA:
    """Represents a DFA"""

    def __init__(
        self,
        num_nodes: int,
        alphabet: Tuple[str],
        transitions: Tuple[dict],
        rng: np.random.Generator,
    ):
        assert len(transitions) == num_nodes
        transitions = {i: v for i, v in enumerate(transitions)}
        dfa = SimpleDFA(
            states=set(list(range(num_nodes))),
            alphabet=set(alphabet),
            initial_state=0,
            accepting_states=set(list(range(num_nodes))),
            transition_function=transitions,
        )
        self.dfa = dfa
        self.rng = rng

    def _sorted_transitions(self):
        nodes = sorted(list(self.dfa._transition_function.keys()))
        transitions = []
        for node in nodes:
            node_transitions = self.dfa._transition_function[node]
            # sort node transitions by outgoing state
            transitions.append(
                tuple(sorted(node_transitions.items(), key=lambda item: item[1]))
            )
        return tuple(transitions)

    def _minimize(self):
        # minimize super
        self.dfa = self.dfa.minimize()
        return self

    def _trim(self):
        # trim super
        self.dfa = self.dfa.trim()
        return self

    def __hash__(self):
        # Here I assume the initial state is always the smallest node
        return hash(self._sorted_transitions())

    def __call__(self, word: str):
        current_node = self.dfa._initial_state
        for symbol in word.split():
            if symbol not in self.dfa._transition_function[current_node]:
                return False
            else:
                current_node = self.dfa._transition_function[current_node][symbol]
        return True

    def forward(self, word: str):
        current_node = self.dfa._initial_state
        for symbol in word.split():
            if symbol not in self.dfa._transition_function[current_node]:
                return None
            else:
                current_node = self.dfa._transition_function[current_node][symbol]
        return current_node

    def trace(self, word: str):
        current_node = self.dfa._initial_state
        path = [current_node]
        for symbol in word.split():
            try:
                self.dfa._transition_function[current_node]
            except:
                breakpoint()
            if symbol not in self.dfa._transition_function[current_node]:
                return path
            else:
                current_node = self.dfa._transition_function[current_node][symbol]
                path.append(current_node)
        return path

    def sample(self, length=1):
        """Samples a random word from the DFA"""
        current_node = self.dfa._initial_state
        word = ""
        for _ in range(length):
            outgoing_symbols = list(self.dfa._transition_function[current_node].keys())
            symbol = self.rng.choice(outgoing_symbols)
            word += symbol + " "
            current_node = self.dfa._transition_function[current_node][symbol]
        word = word.rstrip()
        return word


class RandomDFASampler:
    """Samples random DFAs given configs"""

    num_nodes: int
    alphabet: Tuple[str]
    max_outgoing_edge: int
    rng: np.random.Generator = None

    def __init__(
        self,
        num_nodes: int,
        alphabet: Tuple[str],
        max_outgoing_edge: int,
        seed: int = 42,
    ):
        self.num_nodes = num_nodes
        self.alphabet = alphabet
        self.max_outgoing_edge = max_outgoing_edge
        self.rng = np.random.default_rng(seed)

    def sample(self):
        transitions = [{} for _ in range(self.num_nodes)]
        for node in range(self.num_nodes):
            num_transitions = self.rng.integers(1, self.max_outgoing_edge)
            transition_symbols = self.rng.choice(
                self.alphabet, size=num_transitions, replace=False
            )
            # exclude self loops
            possible_nodes = [n for n in range(self.num_nodes) if n != node]
            transition_nodes = self.rng.choice(
                possible_nodes, size=num_transitions, replace=False
            )
            transitions[node] = dict(zip(transition_symbols, transition_nodes))
        dfa_rng = np.random.default_rng(self.rng.integers(0, 2**32))
        return DFA(self.num_nodes, self.alphabet, tuple(transitions), dfa_rng)

### Markov
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

def seq_to_x(seq,device="cpu"):
    return torch.tensor(seq,dtype=torch.long)[None,:].to(device)

def get_markov_matrix_dirichlet(n,random_state=None):
    if random_state is not None:
        np.random.seed(random_state)
    matrix=np.random.dirichlet(np.ones(n),size=n)
    return matrix

def get_markov_matrix_block_dirichlet(k,blocksize=5,a=5,random_state=None):
    if random_state is not None:
        np.random.seed(random_state)
    assert k%blocksize==0
    alphas=np.ones((k,k))
    for i in range(k//blocksize):
        alphas[i*blocksize:(i+1)*blocksize,i*blocksize:(i+1)*blocksize]=a
    rows=np.stack([np.random.dirichlet(alpha) for alpha in alphas],axis=0)
    return rows

def get_markov_matrix_sparse(n, sparsity=0.5,random_state=None):
    if random_state:
        #print("Setting random state",random_state)
        np.random.seed(random_state)
    
    num_elements = int(n * n * sparsity)
    assert num_elements >= n, "Sparsity too low"
    while True:# Ensure matrix is strongly connected
        matrix=np.zeros(n**2,dtype=bool)
        matrix[np.random.choice(n**2,num_elements,replace=False)]=True
        matrix=matrix.reshape(n,n)
        sparse_matrix = csr_matrix(matrix)
        _, labels = connected_components(sparse_matrix, directed=True, connection='strong')
        if len(set(labels)) == 1:
            break
    # draw each rows non zero elements from dirichlet
    matrix = matrix.astype(float)
    for i in range(n):
        nonzero=matrix[i]>0
        n_nonzero =nonzero.sum()
        matrix[i][nonzero] = np.random.dirichlet(np.ones(n_nonzero))
    return matrix

def get_stationary_distribution(T):
    is_torch=False
    if isinstance(T,torch.Tensor):
        dtype,device=T.dtype,T.device
        is_torch=True
        T=T.detach().cpu().numpy()
    n = T.shape[0]
    A = np.vstack((T.T - np.eye(n), np.ones(n)))
    b = np.zeros(n + 1)
    b[-1] = 1
    # Solve the linear system
    pi = np.linalg.lstsq(A, b, rcond=None)[0]
    if is_torch:
        pi=torch.tensor(pi,dtype=dtype,device=device)
    return pi

def get_stationary_T(T):
    pi=get_stationary_distribution(T)
    return pi[None,:].repeat(T.shape[0],1)

def get_markov_cross_entropy(T, pi=None):
    if isinstance(T,torch.Tensor):
        T=T.detach().cpu().numpy()
    if pi is None:
        pi = get_stationary_distribution(T)
    #no warning
    with np.errstate(divide='ignore', invalid='ignore'):
        cond_entropy_=np.where(T>0,T*np.log(T),0)
    cond_entropy=np.sum(cond_entropy_,axis=1)
    return -np.sum(pi * cond_entropy)

def get_markov_loglike_unigram(seq,T,pi=None,return_unreduced=False,skip_first=False):
    if isinstance(seq,list):
        seq=torch.tensor(seq,dtype=torch.long)
    if pi is None:
        pi = get_stationary_distribution(T)
    loglikes=torch.log(pi[seq])
    if skip_first:
        loglikes[0]=0.
    if return_unreduced:
        return loglikes
    else:
        return loglikes.sum()

def get_markov_loglike_bigram(seq,T,pi=None,return_unreduced=False,skip_first=False):
    #see if this sequence can be generated from T
    if isinstance(seq,list):
        seq=torch.tensor(seq,dtype=torch.long)
    if pi is None:
        pi = get_stationary_distribution(T)
    if skip_first:
        loglike_i=0.
    else:
        loglike_i=torch.log(pi[seq[0]])
    #transitions=torch.stack([seq[:-1],seq[1:]],dim=1)
    loglikes=torch.log(T[seq[:-1],seq[1:]])
    loglikes=torch.cat([torch.tensor([loglike_i]).to(seq.device),loglikes],dim=0)
    if return_unreduced:
        return loglikes
    else:
        return loglikes.sum()

def get_kl(p_hat,p,eps=1e-8):
    p=p+eps
    kl_=torch.where(p_hat>0,p_hat*torch.log(p_hat/p),torch.tensor(0.0))
    kl=kl_.sum()
    return kl

def get_markov_kl(T_hat,T,pi=None,eps=1e-8):
    if not isinstance(T,torch.Tensor):
        T=torch.tensor(T,dtype=torch.float32)
    if not isinstance(T_hat,torch.Tensor):
        T_hat=torch.tensor(T_hat,dtype=torch.float32)
    if pi is None:
        pi = get_stationary_distribution(T)
    T=T+eps
    assert torch.all(T>0)
    kl_=torch.where(T_hat>0,T_hat*torch.log(T_hat/T),torch.tensor(0.0))
    kl=(torch.sum(kl_,dim=1)*pi).sum()
    return kl

def get_markov_kl_batched(T_hat,T,batched="T_hat",eps=1e-8):
    assert batched in ["T_hat","T","both","outer"]
    if not isinstance(T,torch.Tensor):
        T=torch.tensor(T,dtype=torch.float32)
    if not isinstance(T_hat,torch.Tensor):
        T_hat=torch.tensor(T_hat,dtype=torch.float32)
    T=T+eps
    assert torch.all(T>0)
    if batched=="outer":
        pis=[get_stationary_distribution(T_) for T_ in T]
        pi=torch.stack(pis,dim=0)
        T_hat=T_hat[:,None].repeat(1,len(T),1,1)
        T=T[None].repeat(len(T_hat),1,1,1)
        assert pi.shape==T.shape[1:3]
        kl_=torch.where(T_hat>0,T_hat*torch.log(T_hat/T),torch.tensor(0.0))
        kl=(torch.sum(kl_,dim=3)*pi[None,:,:]).sum(2)
        return kl
    if batched=="both":
        pis=[get_stationary_distribution(T_) for T_ in T]
        pi=torch.stack(pis,dim=0)
    elif batched=="T":
        pis=[get_stationary_distribution(T_) for T_ in T]
        pi=torch.stack(pis,dim=0)
        T_hat=T_hat[None].repeat(len(T),1,1)
    elif batched=="T_hat":
        pi=get_stationary_distribution(T)
        T=T[None].repeat(len(T_hat),1,1)
        pi=pi[None].repeat(len(T_hat),1)
    assert pi.shape==T.shape[:2]
    kl_=torch.where(T_hat>0,T_hat*torch.log(T_hat/T),torch.tensor(0.0))
    kl=(torch.sum(kl_,dim=2)*pi).sum(1)
    return kl

def get_markov_js(T1,T2,eps=1e-8):
    assert False, "reimplement this function"
    if not isinstance(T1,torch.Tensor):
        T1=torch.tensor(T1,dtype=torch.float32)
    if not isinstance(T2,torch.Tensor):
        T2=torch.tensor(T2,dtype=torch.float32)
    T1=T1+eps
    T2=T2+eps
    assert torch.all(T1>0)
    assert torch.all(T2>0)
    m=(T1+T2)/2
    kl1=torch.sum(torch.where(T1>0,T1*torch.log(T1/m),torch.tensor(0.0)))
    kl2=torch.sum(torch.where(T2>0,T2*torch.log(T2/m),torch.tensor(0.0)))
    js=(kl1+kl2)/2
    return js

def get_markov_js_batched(T1,T2,eps=1e-8,i_batched=0):
    assert False, "reimplement this function"
    assert i_batched in [0,1]
    if not isinstance(T1,torch.Tensor):
        T1=torch.tensor(T1,dtype=torch.float32)
    if not isinstance(T2,torch.Tensor):
        T2=torch.tensor(T2,dtype=torch.float32)
    T1=T1+eps
    T2=T2+eps
    assert torch.all(T1>0)
    assert torch.all(T2>0)
    if i_batched==0:
        T2=T2[None].repeat(len(T1),1,1)
    else:
        T1=T1[None].repeat(len(T2),1,1)
    m=(T1+T2)/2
    kl1=torch.sum(torch.where(T1>0,T1*torch.log(T1/m),torch.tensor(0.0)),dim=(1,2))
    kl2=torch.sum(torch.where(T2>0,T2*torch.log(T2/m),torch.tensor(0.0)),dim=(1,2))
    js=(kl1+kl2)/2
    return js

def get_markov_dataset_single(k,l,structured=False,random_state=None):
    if not structured:
        transition_params={"type":"dense","random_state":random_state}
    else:
        transition_params={"type":"block_dirichlet","blocksize":5,"a":5,"random_state":random_state}
    config_single={
            "task":"markov",
            "data_params":{
                "k":k.item() if isinstance(k,torch.Tensor) else k,
                "transition_params":transition_params,
                "l":l,
                "batch_size":128,
            },
        }
    return get_dataset(config_single)


def sample_seq_all_last(dataset,k,max_context):
    while True:#almost 100% true but just make sure
        seq_single=dataset.get_data()
        if any([i not in seq_single for i in range(k)]):
            continue
        i_last_locs=[]
        for i_last in range(k):
            i_last_locs.append(torch.nonzero(seq_single==i_last)[:,0].max().item())
        if np.min(i_last_locs)<max_context:
            continue
        break
    return seq_single,i_last_locs


def get_aligned_contexts(seq,k,n_contexts,i_last_locs,get_loglikess=False,Ts=None,check_sequential_last=True):
    if get_loglikess:
        assert Ts is not None
        all_loglikes_u=torch.stack([get_markov_loglike_unigram(seq,T,return_unreduced=True) for T in Ts],dim=0)
        all_loglikes_b=torch.stack([get_markov_loglike_bigram(seq,T,return_unreduced=True) for T in Ts],dim=0)
    #organize data so that i_last is aligned
    contextss=[]
    loglikess_u=[]
    loglikess_b=[]
    for n_context in n_contexts:
        contexts=[]
        loglikes_u=[]
        loglikes_b=[]
        for i_last,i_last_loc in enumerate(i_last_locs):
            contexts.append(seq[i_last_loc-n_context+1:i_last_loc+1])
            if get_loglikess:
                loglikes_u.append(all_loglikes_u[:,i_last_loc-n_context+1:i_last_loc+1].sum(-1))
                loglikes_b.append(all_loglikes_b[:,i_last_loc-n_context+1:i_last_loc+1].sum(-1))
        contexts=torch.stack(contexts,dim=0)
        if get_loglikess:
            loglikes_u=torch.stack(loglikes_u,dim=0)
            loglikes_b=torch.stack(loglikes_b,dim=0)
        if check_sequential_last:
            assert all(contexts[:,-1]==torch.arange(k).to(contexts.device))
        assert contexts.shape==(k,n_context)
        contextss.append(contexts)
        if get_loglikess:
            loglikess_u.append(loglikes_u)
            loglikess_b.append(loglikes_b)
    if get_loglikess:
        loglikess_u=torch.stack(loglikess_u,dim=0)
        loglikess_b=torch.stack(loglikess_b,dim=0)
        return contextss,loglikess_u,loglikess_b
    return contextss

def get_next_probs(model,input_seq,k,return_logits=False,device="cpu",**kwargs):
    if isinstance(input_seq,list):
        input_ids=torch.tensor(input_seq)[None,:].to(device)
    elif isinstance(input_seq,torch.Tensor):
        input_ids=input_seq[None,:].to(device)
    else:
        raise ValueError("Invalid input_seq type")
    return get_next_probs_batched(model,input_ids,k,return_logits=return_logits,device=device,**kwargs)[0]


def get_next_probs_batched(model,input_seqs,k,return_logits=False,**kwargs):
    assert isinstance(input_seqs,torch.Tensor)
    if not isinstance(k,int):
        k=int(k)
    with torch.no_grad():
        next_logits=model(input_seqs,k=k,**kwargs)
    if return_logits:
        return next_logits[:,-1,:k]
    next_probs=torch.softmax(next_logits[:,-1,:k], -1).detach().cpu()
    return next_probs

def get_ps_bayes_up(posteriors,pis):
    ps=[]
    for posterior in posteriors:
        p=(pis*posterior[:,None]).sum(0)
        ps.append(p)
    ps=torch.stack(ps,dim=0)
    return ps

def get_ps_bayes_bp(contexts,posteriors,Ts):
    ps=[]
    for context,posterior in zip(contexts,posteriors):
        i_last=context[-1]
        p=(Ts[:,i_last,:]*posterior[:,None]).sum(0)
        ps.append(p)
    ps=torch.stack(ps,dim=0)
    return ps

def get_ps_icl_b(contexts,k,prior="dirichlet"):#infer probabilities from context
    assert prior in ["dirichlet"]
    ps=[]
    for context in contexts:#count transitions
        i_last=context[-1]
        inds=torch.nonzero(context[:-1]==i_last)[:,0]
        i_nexts=context[inds+1]
        counts=torch.tensor([torch.sum(i_nexts==i).item() for i in range(k)]).float().to(context.device)
        p=(counts+1)/(counts.sum()+k)
        ps.append(p)
    ps=torch.stack(ps,dim=0)
    return ps

def get_ps_model(model,contexts,k,**kwargs):
    device=next(model.parameters()).device
    next_probs=get_next_probs_batched(model,contexts.to(device),k,**kwargs)
    return next_probs

####################### Deprecated #######################

def get_ps_bayes(contexts,priors,logprobs,Ts):
    assert False, "Deprecated"
    ps_sel=[]
    ps_avg=[]
    for context,logprob in zip(contexts,logprobs):
        logposterior=torch.log(priors)+logprob
        posterior=torch.softmax(logposterior,dim=-1)
        i_last=context[-1]
        p_sel=Ts[posterior.argmax(),i_last]
        p_avg=(Ts*posterior[:,None,None]).sum(0)[i_last]
        ps_sel.append(p_sel)
        ps_avg.append(p_avg)
    ps_sel=torch.stack(ps_sel,dim=0)
    ps_avg=torch.stack(ps_avg,dim=0)
    return ps_sel,ps_avg



def get_ps_icl(contexts,k,prior="dirichlet"):#infer probabilities from context
    assert False, "Deprecated"
    assert prior in ["dirichlet"]
    ps=[]
    for context in contexts:#count transitions
        i_last=context[-1]
        inds=torch.nonzero(context[:-1]==i_last)[:,0]
        i_nexts=context[inds+1]
        counts=torch.tensor([torch.sum(i_nexts==i).item() for i in range(k)]).float().to(context.device)
        p=(counts+1)/(counts.sum()+k)
        #below is naive
        #p=torch.tensor([torch.sum(i_nexts==i).item() for i in range(k)]).float().to(context.device)
        #p/=p.sum()
        ps.append(p)
    ps=torch.stack(ps,dim=0)
    return ps

def get_ps_bayes_stationary_likelihood(contexts,priors,logprobs_stationary,Ts):
    assert False, "Deprecated"
    ps_avg=[]
    for context,logprob in zip(contexts,logprobs_stationary):
        logposterior=torch.log(priors)+logprob
        posterior=torch.softmax(logposterior,dim=-1)
        i_last=context[-1]
        p_avg=(Ts*posterior[:,None,None]).sum(0)[i_last]
        ps_avg.append(p_avg)
    ps_avg=torch.stack(ps_avg,dim=0)
    return ps_avg

def get_ps_bayes_stationary(priors,logprobs,pis):
    assert False, "Deprecated"
    ps_avg=[]
    for logprob in logprobs:
        logposterior=torch.log(priors)+logprob
        posterior=torch.softmax(logposterior,dim=-1)
        p_avg=(pis*posterior[:,None]).sum(0)
        ps_avg.append(p_avg)
    ps_avg=torch.stack(ps_avg,dim=0)
    return ps_avg

def get_ps_model_old(model,contexts,k,**kwargs):
    assert False, "Deprecated"
    device=next(model.parameters()).device
    batchable=all([len(context)==len(contexts[0]) for context in contexts])
    if batchable:
        print("batching")
        contexts=torch.cat(contexts,dim=0)
        print("Batched")
        next_probs=get_next_probs_batched(model,contexts,k,device=device,**kwargs)
        return next_probs
    else:
        ps=[]
        kls=[]
        for context in contexts:
            next_probs=get_next_probs(model,context,k,device=device,**kwargs)
            ps.append(next_probs)
        ps=torch.stack(ps,dim=0)
        return ps

def generate_markov(model,input_seq,k,
                    n=100,temperature=1.0,device="cpu"):
    assert False, "check needed"
    if isinstance(input_seq,list):
        input_seq=torch.tensor(input_seq)
    input_ids=input_seq[None,:].to(device)
    x=input_ids
    with torch.no_grad():
        for i in range(n):
            if temperature<=0.0:
                logits_next=model(x)[:,-1,:k]
                next_token=torch.argmax(logits_next,dim=-1)
                x=torch.cat([x,next_token[:,None]],dim=1)
            else:
                logits_next=model(x)[:,-1,:k] / temperature
                next_token=torch.multinomial(torch.softmax(logits_next,dim=-1),num_samples=1)
                x=torch.cat([x,next_token],dim=1)
    return x[0].detach().cpu()[len(input_seq):]
    #return model.generate(input_ids,n=n,T=temperature)[0].detach().cpu()[len(input_seq):]

def generate_markov_batched(model,input_seqs,k,
                            n=100,temperature=1.0,device="cpu"):
    assert False, "check needed"
    assert isinstance(input_seqs,torch.Tensor)
    x=input_seqs.to(device)
    with torch.no_grad():
        for i in range(n):
            if temperature<=0.0:
                logits_next=model(x)[:,-1,:k]
                next_token=torch.argmax(logits_next,dim=-1)
                x=torch.cat([x,next_token[:,None]],dim=1)
            else:
                logits_next=model(x)[:,-1,:k] / temperature
                next_token=torch.multinomial(torch.softmax(logits_next,dim=-1),num_samples=1)
                x=torch.cat([x,next_token],dim=1)
    return x[:,input_seqs.shape[1]:].detach().cpu()
    #return model.generate(input_ids,n=n,T=temperature)[0].detach().cpu()[len(input_seq):]

def get_seq_logprob(model,input_seq,k,
                    logprob_from=None,return_logprobs=False,skip_first=True,device="cpu"):
    assert False, "check needed"
    assert skip_first
    assert len(input_seq)>1
    if isinstance(input_seq,list):
        input_seq=torch.tensor(input_seq)
    input_ids=input_seq[None,:].to(device)
    arr=torch.arange(input_ids.shape[1]-1)
    with torch.no_grad():
        logits=model(input_ids)
        next_probs=torch.softmax(logits[0,:,:k], -1)
        sel_prob=next_probs[arr,input_seq[1:]]
    logselprob=torch.log(sel_prob)
    if logprob_from is not None:
        logselprob=logselprob[logprob_from:]
    if return_logprobs:
        return logselprob.detach().cpu()
    return logselprob.sum().detach().cpu()

def get_markov_posterior(seq,ps,Ts,pis):
    assert False, "Deprecated"
    loglikelihoods=get_markov_loglikelihoods(seq,Ts,pis)
    logprior=torch.log(ps)
    logposterior=logprior+loglikelihoods
    posterior=torch.softmax(logposterior,dim=-1)
    return posterior


########Datasets

class MarkovDataset(torch.utils.data.IterableDataset):
    def __init__(self,k,l,transition_params,online_permute=False):
        self.k=k
        self.l=l
        self.transition_params=transition_params
        if self.transition_params["type"]=="sparse":
            sparsity=self.transition_params["sparsity"]
            random_state=self.transition_params.get("random_state",None)
            self.T=get_markov_matrix_sparse(self.k,sparsity=sparsity,random_state=random_state)
        elif self.transition_params["type"]=="dense":
            random_state=self.transition_params.get("random_state",None)
            self.T=get_markov_matrix_dirichlet(self.k,random_state=random_state)
        elif self.transition_params["type"]=="block_dirichlet":
            blocksize=self.transition_params.get("blocksize",5)
            a=self.transition_params.get("a",5)
            random_state=self.transition_params.get("random_state",None)
            self.T=get_markov_matrix_block_dirichlet(k,blocksize=blocksize,a=a,random_state=random_state)
        elif self.transition_params["type"]=="dirichlet_beta_from_T0":
            random_state=self.transition_params.get("random_state",None)
            np.random.seed(random_state)
            T0=np.array(self.transition_params["T0"])
            beta=self.transition_params["beta"]
            T=[]
            for row in T0:
                T.append(np.random.dirichlet(row*beta))
            self.T=np.array(T)
        else:
            raise ValueError("Invalid transition type: {}".format(self.transition_params["type"]))
        self.pi=get_stationary_distribution(self.T)
        self.tokenizer=Tokenizer_v2()
        self.tokens=self.tokenizer.get_all_tokens()[:self.k]
        self.seq_len=self.l

        self.online_permute=online_permute

    def get_data(self,return_seq=False):
        if self.online_permute:
            perm=np.random.permutation(self.k)
            ind=np.random.choice(self.k,p=self.pi)
            seq=[]
            seq.append(self.tokens[perm[ind]])
            for i in range(self.l-1):
                ind=np.random.choice(self.k,p=self.T[ind])
                seq.append(self.tokens[perm[ind]])
        else:
            ind=np.random.choice(self.k,p=self.pi)
            seq=[]
            seq.append(self.tokens[ind])
            for i in range(self.l-1):
                ind=np.random.choice(self.k,p=self.T[ind])
                seq.append(self.tokens[ind])
        if return_seq:
            return seq
        return torch.tensor(self.tokenizer.encode(seq),dtype=torch.long)

    
    def get_seq_len(self):
        return self.seq_len

    def __iter__(self):
        while True:
            yield self.get_data()
    
class MarkovMixtureDataset(torch.utils.data.IterableDataset):
    def __init__(self,ks,ps,l,transition_paramss,online_permute=False):
        self.ks=ks
        self.n_mix=len(ks)
        assert len(ps)==self.n_mix and len(transition_paramss)==self.n_mix
        self.ps=np.array(ps)
        self.ps=self.ps/np.sum(self.ps)
        self.l=l
        self.transition_paramss=transition_paramss
        self.datasets=[]
        for i in range(len(ks)):
            k=ks[i]
            transition_params=transition_paramss[i]
            dataset=MarkovDataset(k,l,transition_params,online_permute=online_permute)
            self.datasets.append(dataset)
        self.tokenizer=Tokenizer_v2()
        self.tokens=self.tokenizer.get_all_tokens()
        self.seq_len=self.l

    def get_data(self,return_seq=False):
        ind=np.random.choice(len(self.ks),p=self.ps)
        seq=self.datasets[ind].get_data(return_seq=True)
        if return_seq:
            return seq
        return torch.tensor(self.tokenizer.encode(seq),dtype=torch.long)

    def get_seq_len(self):
        return self.seq_len

    def __iter__(self):
        while True:
            yield self.get_data()

class CurriculumDatasets(torch.utils.data.IterableDataset):
    def __init__(self,datasets):
        self.datasets=datasets
        self.seq_len=self.datasets[0].get_seq_len()
        for dataset in self.datasets:
            assert dataset.get_seq_len()==self.seq_len
        self.ind_current=0

    def get_data(self,*args,**kwargs):
        return self.datasets[self.ind_current].get_data(*args,**kwargs)

    def get_seq_len(self):
        return self.seq_len

    def __iter__(self):
        while True:
            yield self.get_data()
    
    def set_curriculum(self,ind):
        self.ind_current=ind


####Generic

def get_dataset(config,split="train"):
    task=config["task"]
    data_params=config["data_params"]

    if task=="linreg":
        d=data_params["d"]
        n_over_d=data_params["n_over_d"]
        vector_output=data_params.get("vector_output",False)
        n=d*n_over_d
        dataset=LinRegDataset(d=d,n=n,vector_output=vector_output)
    elif task=="binary_repeat":
        #simply repeat a binary sequence
        l_min=data_params["l_min"]
        l_max=data_params["l_max"]
        n=data_params["n"]
        dataset=BinaryRepeatDataset(l_min=l_min,l_max=l_max,n=n)
    elif "binary_single_task" in task:
        #apply a single task to a binary sequence
        task_token=task.split("_")[-1]
        l_min=data_params["l_min"]
        l_max=data_params["l_max"]
        n=data_params["n"]
        dataset=BinarySingleTaskDataset(task_token=task_token,l_min=l_min,l_max=l_max,n=n)
    elif task=="binary_multi_task":
        task_tokens=data_params["task_tokens"]
        task_ps=data_params["task_ps"]
        l_min=data_params["l_min"]
        l_max=data_params["l_max"]
        l_max_fix=data_params.get("l_max_fix",None)
        n=data_params["n"]
        same_in_sample=data_params.get("same_in_sample",False)
        dataset=BinaryMultiTaskDataset(task_tokens=task_tokens,task_ps=task_ps,l_min=l_min,l_max=l_max,n=n,
                                       l_max_fix=l_max_fix,same_in_sample=same_in_sample)
    elif task=="star_graph":
        k=data_params["k"]
        l=data_params["l"]
        names=data_params["names"]
        n=data_params["n"]
        n_graphs=data_params["n_graphs"]
        rev=data_params.get("rev",False)
        randomize_edges=data_params.get("randomize_edges",False)
        dataset=SingleStarGraphDataset(k=k,l=l,names=names,n=n,n_graphs=n_graphs,rev=rev,randomize_edges=randomize_edges)
    elif task=="binary_string_generation":
        l=data_params["l"]
        method_name=data_params["method_name"]
        dataset=BinaryStringDataset(l=l,method_name=method_name)
    elif task=="dfa":
        pass
    elif task=="markov":
        assert config.get("mask_loss",False)==False
        k=data_params["k"]
        l=data_params["l"]
        transition_params=data_params["transition_params"]
        dataset=MarkovDataset(k=k,l=l,transition_params=transition_params)
    elif task=="markov_mixture":
        assert config.get("mask_loss",False)==False
        ks=data_params["ks"]
        ps=data_params["ps"]
        l=data_params["l"]
        transition_paramss=data_params["transition_paramss"]
        online_permute=data_params.get("online_permute",False)
        if "curriculum" in data_params:
            if data_params["curriculum"]["type"]=="div":
                n_div=data_params["curriculum"]["n_div"]
                assert len(ks)%n_div==0
                assert len(ps)%n_div==0
                assert len(transition_paramss)%n_div==0
                chunk_size=len(ks)//n_div
                datasets=[]
                for i in range(n_div):
                    ks_div=ks[i*chunk_size:(i+1)*chunk_size]
                    ps_div=ps[i*chunk_size:(i+1)*chunk_size]
                    transition_paramss_div=transition_paramss[i*chunk_size:(i+1)*chunk_size]
                    #print(len(ks_div),len(ps_div),len(transition_paramss_div))
                    datasets.append(MarkovMixtureDataset(ks=ks_div,ps=ps_div,l=l,transition_paramss=transition_paramss_div,online_permute=online_permute))
                dataset=CurriculumDatasets(datasets)
        else:
            dataset=MarkovMixtureDataset(ks=ks,ps=ps,l=l,transition_paramss=transition_paramss,online_permute=online_permute)
    else:
        raise ValueError("Invalid task")
    return dataset

def get_dataloader(config,split="train",dataset=None):
    import multiprocessing
    cpu_count=multiprocessing.cpu_count()
    num_workers=config.get("num_workers",cpu_count//2)

    if dataset is None:
        dataset=get_dataset(config,split=split)
    data_params=config["data_params"]

    batch_size=data_params["batch_size"]
    def collate_fn(data):
        if isinstance(data[0],tuple):
            assert len(data[0])==2
            seqs,loss_masks=zip(*data)
            seqs=torch.stack(seqs,dim=0)
            loss_masks=torch.stack(loss_masks,dim=0)
            return seqs,loss_masks
        elif isinstance(data[0],torch.Tensor):
            seqs=torch.stack(data,dim=0)
            return seqs,None
        else:
            raise ValueError("Invalid data type")
    dl=torch.utils.data.DataLoader(dataset,batch_size=batch_size,num_workers=num_workers,collate_fn=collate_fn)
    return dl

def get_model(config,device):
    model_params=config["model_params"]
    model_type=model_params["model_type"]
    optimizer_type=model_params["optimizer_type"]
    optimizer_params=model_params["optimizer_params"]

    if model_type=="transformer":
        gpt_config=model_params["gpt_config"]
        gpt_config=configs.GPTConfig(**gpt_config)
        model=models.Transformer(gpt_config)
        model=model.to(device)
        if optimizer_type=="auto":
            model.optimizer=model.transformer.configure_optimizers(**optimizer_params,device_type=device.type)
        else:
            model.optimizer=getattr(torch.optim,optimizer_type)(model.parameters(),**optimizer_params)
    elif model_type=="nano_gpt":
        from nanoGPT.model import GPTConfig, GPT
        gpt_config=model_params["gpt_config"]
        gpt_config=GPTConfig(**gpt_config)
        model=GPT(gpt_config)
        model=model.to(device)
        if optimizer_type=="auto":
            model.optimizer=model.configure_optimizers(**optimizer_params,device_type=device.type)
        else:
            model.optimizer=getattr(torch.optim,optimizer_type)(model.parameters(),**optimizer_params)
    else:
        raise ValueError("Invalid model type")

    if "scheduler_params" in model_params and model_params["scheduler_params"] is not None:
        learning_rate=optimizer_params["learning_rate"]

        scheduler_params=model_params["scheduler_params"]
        warmup_iters=scheduler_params["warmup_iters"]
        lr_decay_iters=scheduler_params["lr_decay_iters"]
        min_lr=scheduler_params["min_lr"]

        def get_lr(step):
            # 1) linear warmup for warmup_iters steps
            if step < warmup_iters:
                return learning_rate * step / warmup_iters
            # 2) if step > lr_decay_iters, return min learning rate
            if step > lr_decay_iters:
                return min_lr
            # 3) in between, use cosine decay down to min learning rate
            decay_ratio = (step - warmup_iters) / (lr_decay_iters - warmup_iters)
            assert 0 <= decay_ratio <= 1
            coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff ranges 0..1
            return min_lr + coeff * (learning_rate - min_lr)
        model.get_lr=get_lr

    return model

class Tokenizer():
    def __init__(self):
        self.encoding={}
        n=0
        for i in range(100):
            self.encoding[str(i)]=n
            n+=1
        tokens=["NULL","DO","DONE",
                "REP","INV","REV","SUM","PAR","RAND",
                "FHALF","SHALF","INXOR","OUTXOR","NTRIPLET111",
                "NTRIPLET010","NDOUBLET11","NDOUBLET01","NQUAD1111","NQUAD0101","ALL1","ALL0",
                "SUMHALVES",
                #star graph
                "GRAPH","/","TO","FIND","FINDREV",
                "GENBS1","GENBS2","GENBS3","GENBS4",
                ]#DO NOT CHANGE ORDER, only append
        for s in tokens:
            self.encoding[s]=n
            n+=1

        assert len(self.encoding)==n
        self.decoding=dict(zip(self.encoding.values(),self.encoding.keys()))
        #assert type
        assert all([type(k)==str and type(v)==int for k,v in self.encoding.items()])
        assert all([type(k)==int and type(v)==str for k,v in self.decoding.items()])
        #assert same length
        assert len(self.encoding)==len(self.decoding)
    
    def get_n_tokens(self):
        return len(self.encoding)
    
    def get_null_id(self):
        return self.encoding["NULL"]

    def encode_(self,s):
        if s not in self.encoding.keys():
            assert False
        return self.encoding[s]
    
    def encode(self,strs):
        assert type(strs)==list
        ids=[]
        for s in strs:
            ids.append(self.encode_(s))
        return ids

    def decode_(self,i):
        if i not in self.decoding.keys():
            assert False
        return self.decoding[i]
    
    def decode(self,ids):
        strs=[]
        for i in ids:
            strs.append(self.decode_(i))
        return strs

    def decode_str(self,inputs):
        if type(inputs[0])==str:
            strs=inputs
        else:
            strs=self.decode(inputs)
        return " ".join(strs)
    

class Tokenizer_v2():
    def __init__(self):
        self.encoding={}
        n=0
        #26*26 tokens
        for al1 in "abcdefghijklmnopqrstuvwxyz":
            for al2 in "abcdefghijklmnopqrstuvwxyz":
                self.encoding[al1+al2]=n
                n+=1

        assert len(self.encoding)==n
        self.decoding=dict(zip(self.encoding.values(),self.encoding.keys()))
        #assert type
        assert all([type(k)==str and type(v)==int for k,v in self.encoding.items()])
        assert all([type(k)==int and type(v)==str for k,v in self.decoding.items()])
        #assert same length
        assert len(self.encoding)==len(self.decoding)

    def get_all_tokens(self):
        return list(self.encoding.keys())
    
    def get_n_tokens(self):
        return len(self.encoding)
    
    def get_null_id(self):
        return self.encoding["NULL"]

    def encode_(self,s):
        if s not in self.encoding.keys():
            assert False
        return self.encoding[s]
    
    def encode(self,strs):
        assert type(strs)==list
        ids=[]
        for s in strs:
            ids.append(self.encode_(s))
        return ids

    def decode_(self,i):
        if i not in self.decoding.keys():
            assert False
        return self.decoding[i]
    
    def decode(self,ids):
        strs=[]
        for i in ids:
            strs.append(self.decode_(i))
        return strs

    def decode_str(self,inputs):
        if type(inputs[0])==str:
            strs=inputs
        else:
            strs=self.decode(inputs)
        return " ".join(strs)

def get_step(ckpt_path):
    filename=os.path.basename(ckpt_path)
    step=int(filename.split("step=")[-1].split(".")[0])
    return step

def get_ckpt_paths(exp_dir):
    ckpt_dir=os.path.join(exp_dir,"checkpoints")
    ckpt_paths=glob.glob(os.path.join(ckpt_dir,"ckpt_*.pt*"))
    ckpt_paths=sorted(ckpt_paths,key=get_step)
    steps=[get_step(ckpt_path) for ckpt_path in ckpt_paths]
    return dict(zip(steps,ckpt_paths))


def get_nano_gpt_forward_flops(gpt_config,seq_len):
    #chinchilla paper
    n_embd=gpt_config["n_embd"]
    n_head=gpt_config["n_head"]
    n_layer=gpt_config["n_layer"]
    rmlp=gpt_config.get("rmlp",4)
    ####
    key_size=n_embd//n_head
    ffw_size=rmlp*n_embd
    ###
    #kqv: 2 × 3 × seq_len × d_model × (key_size × num_heads)
    attn_kqv_flops=2*3*seq_len*n_embd*(key_size*n_head)
    #k@q: 2 × seq_len × seq_len × (key_size × num_heads)
    attn_kq_flops=2*seq_len*seq_len*(key_size*n_head)
    #softmax: 3 × num_heads × seq_len × seq_len
    attn_softmax_flops=3*n_head*seq_len*seq_len
    #softmax@q: 2 × seq_len × seq_len × (key_size × num_heads)
    attn_softmax_q_flops=2*seq_len*seq_len*(key_size*n_head)
    #output: : 2 × seq_len × (key_size × num_heads) × d_model
    attn_output_flops=2*seq_len*(key_size*n_head)*n_embd
    #mlp: 2 × seq_len × (d_model × ffw_size + d_model × ffw_size)
    mlp_flops=2*seq_len*(n_embd*ffw_size+n_embd*ffw_size)
    #total: num_layers× (total_attention+dense_block)
    flops=n_layer*(attn_kqv_flops+attn_kq_flops+attn_softmax_flops+attn_softmax_q_flops+attn_output_flops+mlp_flops)
    return flops





