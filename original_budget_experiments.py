"""Matched local-only/Ring control at the original 20-round, two-epoch budget."""
from pathlib import Path
import torch
from experiments import RunSpec, run, run_id

def main():
    torch.set_num_threads(4)
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    for seed in [42,43,44,45,46]:
        for method in ['local','gossip']:
            spec=RunSpec(seed=seed,method=method,rounds=20,local_epochs=2,batch_size=64)
            path=Path('results/review/runs')/(run_id(spec)+'.json')
            if path.exists():continue
            print(path.stem,flush=True)
            result=run(spec,path,device)
            print(f"Completed in {result['elapsed_seconds']:.1f}s",flush=True)
if __name__=='__main__':main()
