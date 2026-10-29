"""Deterministic report illustrations and spectral figures."""
from pathlib import Path
import json
import numpy as np
import networkx as nx
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from network import create_topology,get_mixing_matrix
from spectral_analysis import profile_topology,simulate_gossip_consensus,compute_theoretical_convergence

OUT=Path('results/review/figures');OUT.mkdir(parents=True,exist_ok=True)
fig,ax=plt.subplots(figsize=(12,3.2));ax.set_xlim(0,12);ax.set_ylim(0,3.2);ax.axis('off')
labels=['Partition data\nIID / two class / Dirichlet','Local SGD\nOne epoch per round','Trigger decision\nM1 always / M2 threshold','Loss and delay\nBernoulli / burst channel','Neighbor averaging\nNormalize received weights']
for i,label in enumerate(labels):
 x=.1+i*2.4
 ax.add_patch(FancyBboxPatch((x,1.3),2.05,1,boxstyle='round,pad=.04',facecolor='#eef3f7',edgecolor='#33495e'))
 ax.text(x+1.025,1.8,label,ha='center',va='center',fontsize=9)
 if i<4:ax.annotate('',xy=(x+2.35,1.8),xytext=(x+2.08,1.8),arrowprops={'arrowstyle':'->','color':'#33495e'})
ax.text(6,.55,'Evaluate all client models: test accuracy | consensus error | actual messages and bytes | mean drift',ha='center',fontsize=10)
ax.text(6,2.8,'M3 separately analyzes the ideal fixed mixing matrix; it does not train a model',ha='center',fontsize=10)
fig.tight_layout();fig.savefig(OUT/'architecture.png',dpi=180,bbox_inches='tight');plt.close(fig)
fig,axes=plt.subplots(1,3,figsize=(12,3.5))
profiles=[]
for ax,topo in zip(axes,['ring','grid','random']):
 g=create_topology(topo,5,seed=42);p=profile_topology(topo,5);profiles.append(p)
 pos=nx.circular_layout(g) if topo!='grid' else {0:(0,1),1:(1,1),2:(2,1),3:(0,0),4:(1,0)}
 nx.draw_networkx(g,pos,ax=ax,node_color='#e3edf4',edge_color='#33495e',node_size=650,font_size=10)
 ax.set_title(f"{topo.title()}: {g.number_of_edges()} edges, gap={p['spectral_gap']:.3f}")
 ax.axis('off')
fig.tight_layout();fig.savefig(OUT/'topologies.png',dpi=180);plt.close(fig)
fig,axes=plt.subplots(1,3,figsize=(12,3.8))
for ax,topo in zip(axes,['ring','grid','random']):
 w=get_mixing_matrix(create_topology(topo,5,seed=42));p=profile_topology(topo,5)
 emp=simulate_gossip_consensus(w,35);bound=compute_theoretical_convergence(p['spectral_gap'],35,emp[0])
 ax.semilogy(emp,label='Scalar gossip variance',color='#2776af')
 ax.semilogy(bound,'--',label='E0 * rho^(2t)',color='black')
 ax.set_title(topo.title());ax.set_xlabel('Averaging round');ax.set_ylabel('Squared disagreement');ax.grid(alpha=.2);ax.legend(fontsize=8)
fig.tight_layout();fig.savefig(OUT/'spectral_bound.png',dpi=180);plt.close(fig)
(OUT/'spectral_table.json').write_text(json.dumps(profiles,indent=2))
# Legacy audit computed from original saved counts, without rewriting evidence.
legacy=json.loads(Path('results/m2/m2_all_results.json').read_text())
audit=[]
for r in legacy:
 audit.append({'topology':r['topology'],'loss':r['packet_loss_rate'],'delta':r['delta'],
               'broadcasts_suppressed':r['total_suppressed'],'broadcast_opportunities':100,
               'actual_messages':r['total_communication']['attempted'],'full_messages':r['full_gossip_msgs'],
               'actual_message_savings_pct':100*(1-r['total_communication']['attempted']/r['full_gossip_msgs'])})
Path('results/review/legacy_accounting_audit.json').write_text(json.dumps(audit,indent=2))
print('Created report diagrams, spectral figures, and legacy accounting audit')

# CPU inference illustration from the portable checkpoint.
import torch
from model import MNISTNet
saved=torch.load('demo_assets/mnist_demo.pt',map_location='cpu',weights_only=True)
model=MNISTNet();model.load_state_dict(saved['state_dict']);model.eval()
with torch.no_grad():pred=model(saved['images']).argmax(1)
fig,axes=plt.subplots(4,5,figsize=(8,6))
for i,ax in enumerate(axes.flat):
 ax.imshow(saved['images'][i,0]*.3081+.1307,cmap='gray')
 ax.set_title(f"True {int(saved['labels'][i])}, predicted {int(pred[i])}",fontsize=9)
 ax.axis('off')
fig.tight_layout();fig.savefig('demo_assets/demo_predictions.png',dpi=160);plt.close(fig)
