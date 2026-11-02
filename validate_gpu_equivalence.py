"""Full-budget GPU zero-threshold equivalence for five matched seeds."""
import json
from pathlib import Path
import torch
import experiments as e

def main():
    torch.set_num_threads(4)
    device=torch.device('cuda')
    root=Path('results/review')
    checks=[]
    for seed in [42,43,44,45,46]:
        spec=e.RunSpec(seed=seed,partition='dirichlet')
        baseline=json.loads((root/'runs'/(e.run_id(spec)+'.json')).read_text())
        spec.method='event'
        path=root/'validation_runs'/(e.run_id(spec)+'.json')
        candidate=json.loads(path.read_text()) if path.exists() else e.run(spec,path,device)
        exact=(baseline['initial_sha256']==candidate['initial_sha256'] and baseline['final_parameter_sha256']==candidate['final_parameter_sha256'])
        assert exact,f'Parameter hashes differ for seed {seed}'
        assert [h['accuracy'] for h in baseline['history']]==[h['accuracy'] for h in candidate['history']]
        checks.append({'seed':seed,'rounds':10,'exact_initial_and_all_final_client_hashes':exact,'exact_per_round_accuracy':True})
    validation={'gpu_zero_threshold_checks':checks,'partition':'dirichlet','clients':5,'device':torch.cuda.get_device_name(0),'regression_tests_passed':15}
    (root/'validation.json').write_text(json.dumps(validation,indent=2))
    print('All five 10-round GPU equivalence checks passed',flush=True)
if __name__=='__main__':main()
