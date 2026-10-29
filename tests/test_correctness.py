"""Regression checks for the review's scientific correctness issues."""
import copy
import numpy as np
import pytest
import torch
from channel import ImperfectChannel
from gossip import GossipAggregator
from gossip_event_triggered import EventTriggeredGossipAggregator
from network import create_topology, get_mixing_matrix
from spectral_analysis import profile_topology, simulate_gossip_consensus, compute_theoretical_convergence
from dataset import partition_two_class, partition_dirichlet, partition_non_iid
from model import MNISTNet, get_param_count

class Client:
    def __init__(self, cid, value):
        self.client_id, self.params = cid, torch.tensor([value], dtype=torch.float64)
    def get_params(self):
        return self.params.clone()
    def set_params(self, p):
        self.params = p.clone()

@pytest.mark.parametrize('loss,delay', [(0,0),(.2,0),(.2,4)])
def test_zero_threshold_is_exact_gossip(loss, delay):
    g = create_topology('grid',5)
    adj = {i:list(g.neighbors(i)) for i in g}
    a = [Client(i,float(i)) for i in range(5)]
    b = copy.deepcopy(a)
    ca = ImperfectChannel(loss, delay, 42, 8)
    cb = ImperfectChannel(loss, delay, 42, 8)
    ga = GossipAggregator(a,adj,get_mixing_matrix(g),ca)
    gb = EventTriggeredGossipAggregator(b,adj,get_mixing_matrix(g),cb,0)
    for rnd in range(1,10):
        for c in a+b:
            c.params += c.client_id * .1
        ga.aggregate_round(rnd)
        gb.aggregate_round(rnd)
        assert all(torch.equal(c.get_params(),d.get_params()) for c,d in zip(a,b))
    assert ca.log == cb.log
    assert all(h['triggered']==5 for h in gb.trigger_history)

def test_pending_is_not_delivered():
    c = ImperfectChannel(0,100,42,100)
    for i in range(10):
        c.transmit(0,i,torch.tensor([1.]),1)
    stats = c.get_total_stats()
    assert stats['pending'] > 0
    assert stats['attempted'] == stats['dropped'] + stats['pending'] + stats['delivered']
    assert stats['bytes_delivered'] == 100 * stats['delivered']

def test_channel_randomness_independent_of_suppressed_sends():
    a,b = ImperfectChannel(.2,4,42), ImperfectChannel(.2,4,42)
    for rnd in range(1,9):
        a.transmit(2,3,None,rnd)
        a.transmit(0,1,None,rnd)
        b.transmit(0,1,None,rnd)
    assert [e for e in a.log if e.sender==0] == b.log

def test_old_packet_cannot_replace_current_arrival():
    class Channel(ImperfectChannel):
        def collect_delayed(self,rnd):
            return [(0,1,torch.tensor([-100.],dtype=torch.float64),0)]
    clients = [Client(0,10.),Client(1,0.)]
    a = GossipAggregator(clients,{0:[1],1:[0]},np.full((2,2),.5),Channel())
    a.aggregate_round(1)
    assert clients[1].params.item()==5

def test_grid_message_savings_differs_from_broadcast_savings():
    g = create_topology('grid',5)
    clients = [Client(i,0.) for i in range(5)]
    a = EventTriggeredGossipAggregator(clients,{i:list(g.neighbors(i)) for i in g},get_mixing_matrix(g),ImperfectChannel(),.5)
    clients[1].params += 1 # only degree-three node sends
    stats = a.aggregate_round(1)
    assert stats['suppressed']==4
    assert a.channel.get_total_stats()['attempted']==3 # 70% messages vs 80% broadcasts

def test_spectra_and_parameter_count():
    assert get_param_count(MNISTNet())==206922
    p=profile_topology('ring',5)
    assert p['lambda_min'] < 0
    assert np.allclose(p['all_eigenvalues'],[1,.53934466,.53934466,-.20601133,-.20601133])
    g=profile_topology('grid',5)
    assert np.isclose(g['lambda_min'],-.25)

def test_squared_error_bound():
    for t in ['ring','grid','random']:
        g=create_topology(t,5)
        emp=simulate_gossip_consensus(get_mixing_matrix(g),15)
        bound=compute_theoretical_convergence(profile_topology(t,5)['spectral_gap'],15,emp[0])
        assert np.all(emp <= bound + 1e-14)

def test_partitions_cover_all_samples():
    targets=np.repeat(np.arange(10),201)
    for splits in [partition_two_class(targets,5),partition_dirichlet(targets,5,min_size=10)]:
        assert sorted(i for s in splits for i in s)==list(range(len(targets)))
        assert all(s for s in splits)
    assert all(len(set(targets[s]))==2 for s in partition_two_class(targets,5))
    class Data:
        def __len__(self):return len(targets)
    d=Data(); d.targets=targets
    assert sum(map(len,partition_non_iid(d,7)))==len(targets)

def test_renormalized_directed_loss_can_drift_average():
    class OneWay(ImperfectChannel):
        def transmit(self,sender,receiver,params,round_num):return sender==0
    cs=[Client(0,0.),Client(1,10.)]
    a=GossipAggregator(cs,{0:[1],1:[0]},np.full((2,2),.5),OneWay())
    a.aggregate_round(1)
    assert a.mean_drift_history[-1]==2.5

def test_training_entry_points_match(tmp_path):
    from torch.utils.data import DataLoader, TensorDataset
    from config import Config
    from main import run_experiment
    from main_m2 import run_m2_experiment
    torch.set_num_threads(2)
    torch.manual_seed(7)
    data=TensorDataset(torch.randn(10,1,28,28),torch.arange(10))
    loaders=[DataLoader(data,batch_size=5,shuffle=True,generator=torch.Generator().manual_seed(1)) for _ in range(5)]
    test=DataLoader(data,batch_size=10)
    cfg=Config(num_rounds=2,local_epochs=1,results_dir=str(tmp_path))
    a=run_experiment(cfg,'grid',.2,loaders,test,torch.device('cpu'),2)
    b=run_m2_experiment(cfg,0,'grid',.2,loaders,test,torch.device('cpu'),2)
    assert a['per_round_accuracy']==b['per_round_accuracy']
    assert a['per_round_consensus_error']==b['per_round_consensus_error']
    assert a['total_communication']==b['total_communication']


def test_burst_state_evolves_through_silence():
    a=ImperfectChannel(.2,0,42,loss_model='gilbert-elliott')
    b=ImperfectChannel(.2,0,42,loss_model='gilbert-elliott')
    for rnd in range(1,21):a.transmit(0,1,None,rnd)
    b.transmit(0,1,None,1);b.transmit(0,1,None,20)
    assert a.log[-1]==b.log[-1]

def test_cifar_architecture_count_and_shape():
    from model import CIFARNet
    model=CIFARNet()
    assert get_param_count(model)==268650
    assert model(torch.zeros(2,3,32,32)).shape==(2,10)
    from client import FLClient
    from torch.utils.data import DataLoader, TensorDataset
    data=DataLoader(TensorDataset(torch.zeros(2,3,32,32),torch.tensor([0,1])),batch_size=2)
    client=FLClient(0,data,model_factory=CIFARNet)
    assert np.isfinite(client.local_train(1))


def test_student_t_interval_is_computed_across_seed_values():
    from summarize_results import interval
    estimate=interval([1,2,3,4,5])
    assert estimate['n']==5
    assert estimate['mean']==3
    assert np.isclose(estimate['ci95_half'],1.963243161,atol=1e-8)
    assert interval([1])['ci95_half'] is None
