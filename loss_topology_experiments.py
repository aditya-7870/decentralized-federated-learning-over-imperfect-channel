"""Complete the five-seed Dirichlet MNIST Bernoulli-loss topology comparison."""
from pathlib import Path
import torch
from experiments import RunSpec, run, run_id

def main():
    torch.set_num_threads(4)
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    for seed in [42,43,44,45,46]:
        for topology in ['grid','random']:
            spec=RunSpec(seed=seed,partition='dirichlet',topology=topology,loss=.2)
            path=Path('results/review/runs')/(run_id(spec)+'.json')
            if path.exists():continue
            print(path.stem,flush=True)
            run(spec,path,device)
if __name__=='__main__':main()
