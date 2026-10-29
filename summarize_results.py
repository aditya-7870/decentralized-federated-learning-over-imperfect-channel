"""Aggregate independent seeds with 95% Student-t intervals and create figures."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import numpy as np
from scipy.stats import t
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path('results/review')
FIELDS = ['dataset','partition','method','topology','clients','rounds','local_epochs','batch_size','loss','delay','delta','loss_model','lr','momentum','alpha']

def interval(values):
    a=np.asarray(values,dtype=float)
    half=float(t.ppf(.975,len(a)-1)*a.std(ddof=1)/np.sqrt(len(a))) if len(a)>1 else None
    return {'mean':float(a.mean()),'ci95_half':half,'n':len(a)}

def summarize(root=ROOT):
    root=Path(root)
    runs=[json.loads(p.read_text(encoding='utf-8')) for p in sorted((root/'runs').glob('*.json'))]
    for r in runs:
        r['config'].setdefault('dataset','MNIST')
    groups=defaultdict(list)
    for r in runs:
        groups[tuple(r['config'][f] for f in FIELDS)].append(r)
    summaries=[]
    for key,rs in groups.items():
        entry=dict(zip(FIELDS,key))
        entry['seeds']=sorted(r['config']['seed'] for r in rs)
        for field in ['final_accuracy','final_consensus_error','message_savings_pct','broadcast_suppression_pct','elapsed_seconds']:
            entry[field]=interval([r[field] for r in rs])
        entry['attempted_messages']=interval([r['communication']['attempted'] for r in rs])
        entry['bytes_attempted']=interval([r['communication']['bytes_attempted'] for r in rs])
        entry['dropped_messages']=interval([r['communication']['dropped'] for r in rs])
        entry['pending_messages']=interval([r['communication']['pending'] for r in rs])
        entry['realized_drop_pct']=interval([100*r['communication']['dropped']/max(1,r['communication']['attempted']) for r in rs])
        entry['mean_drift']=interval([np.mean([h['mean_drift'] for h in r['history']]) for r in rs])
        entry['accuracy_curve']=[interval([r['history'][i]['accuracy'] for r in rs]) for i in range(entry['rounds'])]
        entry['consensus_curve']=[interval([r['history'][i]['consensus_error'] for r in rs]) for i in range(entry['rounds'])]
        summaries.append(entry)
    paired=[]
    for s in summaries:
        if s['method']=='local' or s['clients']!=5 or s['loss']!=0 or s['delay']!=0 or s['method']!='gossip': continue
        diffs=[]
        for r in runs:
            if all(r['config'][f]==s[f] for f in FIELDS):
                cfg=r['config']
                base=next((b for b in runs if b['config']['dataset']==cfg['dataset'] and b['config']['method']=='local' and b['config']['partition']==cfg['partition'] and b['config']['seed']==cfg['seed'] and b['config']['clients']==cfg['clients'] and all(b['config'][k]==cfg[k] for k in ['rounds','local_epochs','batch_size','lr','momentum','alpha'])),None)
                if base:
                    diffs.append(100*(r['final_accuracy']-base['final_accuracy']))
        if diffs:
            paired.append({'dataset':s['dataset'],'partition':s['partition'],'topology':s['topology'],'rounds':s['rounds'],'local_epochs':s['local_epochs'],'batch_size':s['batch_size'],'accuracy_gain_pp':interval(diffs)})
    result={'schema_version':2,'confidence_interval':'95% two-sided Student t across independent seed-level results; no clipping at 0/1',
            'groups':summaries,'paired_vs_local':paired,'completed_runs':len(runs)}
    root.mkdir(parents=True,exist_ok=True)
    (root/'summary.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    return result

def plot(summary,root=ROOT):
    root=Path(root); out=root/'figures'; out.mkdir(exist_ok=True)
    gs=summary['groups']
    colors={'local':'#555555','ring':'#2776af','grid':'#22865c','random':'#d07c29'}
    fig,axes=plt.subplots(1,3,figsize=(14,4.2))
    for ax,partition in zip(axes,['iid','two-class','dirichlet']):
        for g in gs:
            if g['rounds']!=10 or g['dataset']!='MNIST' or g['partition']!=partition or g['clients']!=5 or g['method']=='event' or g['loss'] or g['delay']:continue
            label='Local only' if g['method']=='local' else g['topology'].title()
            curve=g['accuracy_curve']; mean=np.array([v['mean'] for v in curve])*100
            half=np.array([v['ci95_half'] or 0 for v in curve])*100
            x=np.arange(1,len(mean)+1); color=colors['local' if g['method']=='local' else g['topology']]
            ax.plot(x,mean,label=label,color=color)
            ax.fill_between(x,mean-half,mean+half,color=color,alpha=.12)
        ax.set_title(partition.upper() if partition=='iid' else partition.title())
        ax.set_xlabel('Gossip round'); ax.set_ylabel('Mean client test accuracy (%)'); ax.grid(alpha=.2); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(out/'baseline_accuracy.png',dpi=180); plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(11,4.2))
    for topo in ['ring','grid']:
        ss=[g for g in gs if g['dataset']=='MNIST' and g['partition']=='dirichlet' and g['clients']==5 and g['topology']==topo and g['loss']==0 and g['delay']==0 and g['method'] in ['gossip','event']]
        ss.sort(key=lambda g:g['delta'])
        if not ss:continue
        axes[0].errorbar([g['delta'] for g in ss],[100*g['final_accuracy']['mean'] for g in ss],yerr=[100*(g['final_accuracy']['ci95_half'] or 0) for g in ss],fmt='o-',label=topo.title())
        axes[1].errorbar([g['delta'] for g in ss],[g['message_savings_pct']['mean'] for g in ss],yerr=[g['message_savings_pct']['ci95_half'] or 0 for g in ss],fmt='o-',label=topo.title())
    for ax in axes:ax.set_xlabel('Fixed trigger threshold delta'); ax.grid(alpha=.2); ax.legend()
    axes[0].set_ylabel('Final accuracy (%)'); axes[1].set_ylabel('Actual message savings (%)')
    fig.tight_layout();fig.savefig(out/'event_tradeoff.png',dpi=180);plt.close(fig)
    ss=[g for g in gs if g['dataset']=='MNIST' and g['partition']=='dirichlet' and g['method']=='gossip' and g['topology']=='ring' and g['loss']==.2 and g['loss_model']=='bernoulli']
    ss.sort(key=lambda g:g['delay'])
    if ss:
        fig,axes=plt.subplots(1,2,figsize=(10,4))
        for ax,field,label in zip(axes,['final_accuracy','final_consensus_error'],['Final accuracy (fraction)','Final consensus error']):
            ax.errorbar([g['delay'] for g in ss],[g[field]['mean'] for g in ss],yerr=[g[field]['ci95_half'] or 0 for g in ss],fmt='o-'); ax.set_xlabel('Maximum delay (rounds)');ax.set_ylabel(label);ax.grid(alpha=.2)
        fig.tight_layout();fig.savefig(out/'delay_effect.png',dpi=180);plt.close(fig)
    return out

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',default=str(ROOT));args=p.parse_args()
    summary=summarize(args.root);plot(summary,args.root)
    print(f"Summarized {summary['completed_runs']} completed runs in {args.root}")
