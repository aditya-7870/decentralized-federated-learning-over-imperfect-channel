"""Audit saved experiment invariants without retraining."""
import json
import math
from pathlib import Path

def main():
    root=Path('results/review')
    paths=sorted((root/'runs').glob('*.json'))
    failures=[]
    for path in paths:
        r=json.loads(path.read_text());c=r['config'];comm=r['communication']
        def check(condition,message):
            if not condition:failures.append(f'{path.name}: {message}')
        expected=50000 if c.get('dataset')=='CIFAR10' else 60000
        check(sum(r['partition_sizes'])==expected,'training coverage size')
        check(all(sum(h)==n for h,n in zip(r['partition_class_counts'],r['partition_sizes'])),'class histogram coverage')
        if c['partition']=='two-class':check(all(sum(v>0 for v in h)==2 for h in r['partition_class_counts']),'exact two labels')
        check(comm['attempted']==comm['dropped']+comm['delivered']+comm['pending'],'message conservation')
        check(comm['bytes_attempted']==comm['attempted']*r['bytes_per_message'],'attempted byte accounting')
        check(comm['bytes_delivered']==comm['delivered']*r['bytes_per_message'],'arrived byte accounting')
        check(r['full_gossip_messages']==c['rounds']*2*len(r['edges']),'full-gossip denominator')
        if c['method']=='local':check(comm['attempted']==0,'local-only sends messages')
        elif c['method']=='gossip':check(comm['attempted']==r['full_gossip_messages'],'full gossip missing attempts')
        check(len(r['history'])==c['rounds'],'round count')
        check(abs(r['history'][-1]['accuracy']-r['final_accuracy'])<1e-12,'final accuracy consistency')
        for h in r['history']:
            check(abs(sum(h['client_accuracies'])/c['clients']-h['accuracy'])<1e-12,'mean accuracy')
            check(0<=h['accuracy']<=1 and math.isfinite(h['consensus_error']) and h['consensus_error']>=0,'finite metric domain')
    if failures:raise SystemExit('\n'.join(failures))
    report={'completed_runs_audited':len(paths),'failures':[],'checks':['training/class coverage','exact two-label partitions','attempt/drop/arrival/pending conservation','byte accounting','graph denominator','zero local traffic','complete gossip traffic','round and accuracy consistency','finite metrics']}
    (root/'result_audit.json').write_text(json.dumps(report,indent=2))
    print(f'{len(paths)} saved runs passed all result invariants')
if __name__=='__main__':main()
