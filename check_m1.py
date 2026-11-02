import json, os

d = json.load(open('results/m1_all_results.json'))
print('M1 Results Summary:')
print('-'*65)
print(f"{'Topology':<10} {'PLR':<6} {'Acc':<10} {'ConsErr':<14} {'Gap':<10} {'Msgs':<8} {'Dropped'}")
print('-'*65)
for r in d:
    tc = r['total_communication']
    topo = r['topology']
    plr  = r['packet_loss_rate']
    acc  = r['final_avg_accuracy']
    ce   = r['final_consensus_error']
    sg   = r['spectral_gap']
    msgs = tc['attempted']
    drop = tc['dropped']
    print(f"{topo:<10} {plr:<6} {acc:<10.4f} {ce:<14.6f} {sg:<10.4f} {msgs:<8} {drop}")

print('-'*65)
print()
print('Checkpoints exist:', os.path.isdir('results/checkpoints'))
ckpt_dir = 'results/checkpoints'
if os.path.isdir(ckpt_dir):
    print('Checkpoint files:', os.listdir(ckpt_dir))
else:
    print('No checkpoints saved (M1 trained before checkpoint-saving was added)')

print()
print('M1 Result files:')
for f in os.listdir('results'):
    if f.endswith('.json'):
        print(' ', f)
