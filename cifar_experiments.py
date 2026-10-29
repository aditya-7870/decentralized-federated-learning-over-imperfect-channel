"""Five-seed CIFAR-10 extension: local-only vs Ring, Dirichlet alpha=.1.

Uses the same training/aggregation implementation as the MNIST suite.
"""
import argparse
from pathlib import Path
import numpy as np
import torch
from torchvision.datasets import CIFAR10
import experiments as e
from model import CIFARNet

def load_tensors(device):
    if 'cifar' not in e._DATA:
        train,test=CIFAR10('./data',train=True,download=True),CIFAR10('./data',train=False,download=True)
        mean=torch.tensor([.4914,.4822,.4465],device=device).view(1,3,1,1)
        std=torch.tensor([.2470,.2435,.2616],device=device).view(1,3,1,1)
        def prepare(d):
            x=torch.as_tensor(np.asarray(d.data),device=device).permute(0,3,1,2).contiguous().float().div_(255)
            return (x-mean)/std, torch.tensor(d.targets,device=device)
        e._DATA['cifar']=(*prepare(train),*prepare(test))
    return e._DATA['cifar']

def main():
    p=argparse.ArgumentParser();p.add_argument('--download-only',action='store_true');args=p.parse_args()
    if args.download_only:
        CIFAR10('./data',train=True,download=True); CIFAR10('./data',train=False,download=True);return
    torch.set_num_threads(4)
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    e.MNISTNet=CIFARNet
    e.load_tensors=load_tensors
    for seed in [42,43,44,45,46]:
        for method in ['local','gossip']:
            spec=e.RunSpec(partition='dirichlet',method=method,seed=seed,dataset='CIFAR10')
            path=Path('results/review/runs')/('CIFAR10_'+e.run_id(spec)+'.json')
            if path.exists():continue
            print(path.stem,flush=True)
            result=e.run(spec,path,device)
            print(f"Completed in {result['elapsed_seconds']:.1f}s",flush=True)

if __name__=='__main__':main()
