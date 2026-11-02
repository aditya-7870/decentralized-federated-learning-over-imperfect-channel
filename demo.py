"""Laptop handover: saved-results summary needs only Python; inference uses CPU."""
import argparse
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parent

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--predict',action='store_true',help='Run the included trained model on 20 bundled test digits, on CPU')
    args=p.parse_args()
    if args.predict:
        import torch
        from model import MNISTNet
        torch.set_num_threads(2)
        path=ROOT/'demo_assets/mnist_demo.pt'
        if not path.exists():raise SystemExit('Demo checkpoint missing: train/export with experiments.py first.')
        saved=torch.load(path,map_location='cpu',weights_only=True)
        model=MNISTNet();model.load_state_dict(saved['state_dict']);model.eval()
        with torch.no_grad():predictions=model(saved['images']).argmax(1)
        print('Device: CPU. No training or dataset download.')
        print('Expected: ',saved['labels'].tolist())
        print('Predicted:',predictions.tolist())
        print('Recorded full-test accuracy from training evaluation:',saved['test_accuracy'])
        return
    path=ROOT/'results/review/summary.json'
    if not path.exists():raise SystemExit('Run summarize_results.py after training.')
    summary=json.loads(path.read_text(encoding='utf-8'))
    print('COMPLETED EXPERIMENTS:',summary['completed_runs'])
    print('95% confidence intervals across independent training seeds')
    for g in summary['groups']:
        a=g['final_accuracy'];h=a['ci95_half']
        ci='unavailable' if h is None else f"+/- {100*h:.2f} pp"
        print(f"{g.get('dataset','MNIST'):7} {g['partition']:10} {g['method']:6} {g['topology']:6} N={g['clients']:2} T={g['rounds']:2} E={g['local_epochs']} loss={g['loss']:.1f} delay={g['delay']} delta={g['delta']:.1f}: {100*a['mean']:.2f}% {ci}; n={a['n']}")
    print('Figures: results/review/figures. Revised report: output/report.')

if __name__=='__main__':main()
