"""Reproducible, resumable review experiments. Does not overwrite legacy results.

Examples:
  python experiments.py --suite smoke
  python experiments.py --suite core
  python experiments.py --suite extended
"""
import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import platform
import time
import numpy as np
import torch
from dataset import get_mnist, partition_two_class, partition_dirichlet
from model import MNISTNet, get_model_params, get_param_count, get_param_bytes
from client import FLClient
from network import create_topology, get_adjacency_dict, get_mixing_matrix, get_spectral_gap
from channel import ImperfectChannel
from gossip import GossipAggregator
from gossip_event_triggered import EventTriggeredGossipAggregator

@dataclass
class RunSpec:
    partition: str = "iid"
    method: str = "gossip"
    topology: str = "ring"
    seed: int = 42
    clients: int = 5
    rounds: int = 10
    local_epochs: int = 1
    batch_size: int = 128
    lr: float = 0.01
    momentum: float = 0.9
    loss: float = 0.0
    delay: int = 0
    delta: float = 0.0
    loss_model: str = "bernoulli"
    alpha: float = 0.1
    dataset: str = "MNIST"

class TensorBatches:
    """Batched indexing avoids per-image Python transforms during repeated runs."""
    def __init__(self, x, y, batch_size, seed=None):
        self.x, self.y, self.batch_size = x, y, batch_size
        self.generator = None if seed is None else torch.Generator().manual_seed(seed)
    def __iter__(self):
        order = (torch.arange(len(self.y)) if self.generator is None else
                 torch.randperm(len(self.y), generator=self.generator)).to(self.x.device)
        for ids in order.split(self.batch_size):
            yield self.x[ids], self.y[ids]
    def __len__(self):
        return (len(self.y) + self.batch_size - 1) // self.batch_size

_DATA = {}
def load_tensors(device):
    key = str(device)
    if key not in _DATA:
        train, test = get_mnist()
        def prepare(d):
            x = d.data.unsqueeze(1).to(device=device, dtype=torch.float32).div_(255)
            x.sub_(0.1307).div_(0.3081)
            return x, d.targets.to(device)
        _DATA[key] = (*prepare(train), *prepare(test))
    return _DATA[key]

def partitions(targets, spec):
    if spec.partition == "two-class":
        return partition_two_class(targets, spec.clients, spec.seed)
    if spec.partition == "dirichlet":
        return partition_dirichlet(targets, spec.clients, spec.alpha, spec.seed)
    ids = np.random.default_rng(spec.seed).permutation(len(targets))
    return [chunk.tolist() for chunk in np.array_split(ids, spec.clients)]

def run(spec, output, device, save_demo=False):
    torch.manual_seed(spec.seed)
    np.random.seed(spec.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    x, y, tx, ty = load_tensors(device)
    indices = partitions(y.cpu().numpy(), spec)
    reference = MNISTNet().to(device)
    initial = get_model_params(reference)
    clients = []
    for i, ids in enumerate(indices):
        loader = TensorBatches(x[ids], y[ids], spec.batch_size, spec.seed + 1000 + i)
        c = FLClient(i, loader, spec.lr, spec.momentum, device, model_factory=MNISTNet)
        c.set_params(initial)
        clients.append(c)
    graph = create_topology(spec.topology, spec.clients, seed=spec.seed, p=0.5)
    W = get_mixing_matrix(graph)
    adjacency = get_adjacency_dict(graph)
    channel = ImperfectChannel(spec.loss, spec.delay, spec.seed, get_param_bytes(reference), spec.loss_model)
    if spec.method == "event":
        aggregator = EventTriggeredGossipAggregator(clients, adjacency, W, channel, spec.delta)
    else:
        aggregator = GossipAggregator(clients, adjacency, W, channel)
    evaluator = TensorBatches(tx, ty, 1000)
    history = []
    started = time.perf_counter()
    for rnd in range(1, spec.rounds + 1):
        for c in clients:
            c.local_train(spec.local_epochs)
        triggers = {"triggered": 0, "suppressed": 0}
        if spec.method != "local":
            triggers = aggregator.aggregate_round(rnd)
        else:
            channel.current_round = rnd
        metrics = [c.evaluate(evaluator) for c in clients]
        params = torch.stack([c.get_params() for c in clients])
        ce = float(((params - params.mean(0))**2).sum(1).mean())
        history.append({"round": rnd, "accuracy": float(np.mean([m[0] for m in metrics])),
                        "loss": float(np.mean([m[1] for m in metrics])),
                        "client_accuracies": [m[0] for m in metrics], "consensus_error": ce,
                        "mean_drift": aggregator.mean_drift_history[-1] if spec.method != "local" else 0,
                        **triggers, **channel.get_round_stats(rnd)})
        if rnd == 1 or rnd % 5 == 0 or rnd == spec.rounds:
            print(f"  round {rnd}/{spec.rounds}: accuracy={history[-1]['accuracy']:.4f} CE={ce:.5f}", flush=True)
    total = channel.get_total_stats()
    full = spec.rounds * 2 * graph.number_of_edges()
    payload = {"schema_version": 2, "config": asdict(spec), "history": history,
               "final_accuracy": history[-1]["accuracy"], "final_consensus_error": history[-1]["consensus_error"],
               "communication": total, "full_gossip_messages": full,
               "message_savings_pct": 100 * (1 - total["attempted"] / full) if full else 0,
               "broadcast_suppression_pct": 100 * sum(h['suppressed'] for h in history) / (spec.rounds * spec.clients),
               "elapsed_seconds": time.perf_counter() - started,
               "parameter_count": get_param_count(reference), "bytes_per_message": get_param_bytes(reference),
               "spectral_gap": float(get_spectral_gap(W)), "edges": sorted([list(e) for e in graph.edges()]),
               "partition_sizes": list(map(len, indices)),
               "partition_class_counts": [np.bincount(y[ids].cpu().numpy(), minlength=10).tolist() for ids in indices],
               "partition_sha256": hashlib.sha256(json.dumps(indices).encode()).hexdigest(),
               "initial_sha256": hashlib.sha256(initial.cpu().numpy().tobytes()).hexdigest(),
               "final_parameter_sha256": [hashlib.sha256(c.get_params().cpu().numpy().tobytes()).hexdigest() for c in clients],
               "environment": {"python": platform.python_version(), "torch": torch.__version__,
                               "numpy": np.__version__, "device": str(device),
                               "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
                               "amp": device.type == "cuda", "deterministic_algorithms": True},
               "objective": "uniform mean of client losses; optimizer momentum remains local",
               "channel_note": "GE: stationary initialization, pGB=.05, pBG=.2, erasure 0/1; stationary loss=.2"}
    output.parent.mkdir(parents=True, exist_ok=True)
    tmp = output.with_suffix('.tmp')
    tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    tmp.replace(output)
    if save_demo:
        demo_dir = Path("demo_assets")
        demo_dir.mkdir(exist_ok=True)
        torch.save({"state_dict": {k:v.cpu() for k,v in clients[0].model.state_dict().items()},
                    "config": asdict(spec), "test_accuracy": metrics[0][0],
                    "images": tx[:20].cpu(), "labels": ty[:20].cpu()}, demo_dir / "mnist_demo.pt")
    return payload

def schedule(suite):
    if suite == "smoke":
        return [RunSpec(rounds=1), RunSpec(method="event", delta=0, rounds=1)]
    specs = []
    for seed in [42,43,44,45,46]:
        for partition in ["iid", "two-class", "dirichlet"]:
            specs.append(RunSpec(seed=seed, partition=partition, method="local"))
            for topo in ["ring", "grid", "random"]:
                specs.append(RunSpec(seed=seed, partition=partition, topology=topo))
        if suite in ("extended", "all"):
            for topo in ["ring", "grid"]:
                for delta in [0.5, 2.0]:
                    specs.append(RunSpec(seed=seed, partition="dirichlet", method="event", topology=topo, delta=delta))
            for delay in [1,2,4]:
                specs.append(RunSpec(seed=seed, partition="dirichlet", loss=.2, delay=delay))
            specs.append(RunSpec(seed=seed, partition="dirichlet", loss=.2))
            specs.append(RunSpec(seed=seed, partition="dirichlet", loss=.2, loss_model="gilbert-elliott"))
        if suite == "all":
            for n in [20,50]:
                for topo in ["ring", "grid"]:
                    specs.append(RunSpec(seed=seed, partition="two-class", topology=topo, clients=n))
    return specs

def run_id(s):
    return (f"{s.partition}_{s.method}_{s.topology}_n{s.clients}_loss{s.loss}_delay{s.delay}"
            f"_delta{s.delta}_{s.loss_model}_seed{s.seed}_r{s.rounds}_e{s.local_epochs}_b{s.batch_size}")

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--suite', choices=['smoke','core','extended','all'], default='core')
    p.add_argument('--output', default='results/review')
    p.add_argument('--device', default='auto', choices=['auto','cpu','cuda'])
    p.add_argument('--limit', type=int)
    args = p.parse_args()
    torch.set_num_threads(4)
    device = torch.device(('cuda' if torch.cuda.is_available() else 'cpu') if args.device == 'auto' else args.device)
    print('DEVICE', device, flush=True)
    specs = schedule(args.suite)
    if args.limit:
        specs = specs[:args.limit]
    for i, spec in enumerate(specs):
        path = Path(args.output) / 'runs' / (run_id(spec) + '.json')
        if path.exists():
            print(f'[{i+1}/{len(specs)}] cached {path.stem}', flush=True)
            continue
        print(f'[{i+1}/{len(specs)}] {path.stem}', flush=True)
        result = run(spec, path, device, save_demo=(spec.seed==42 and spec.partition=='iid' and spec.method=='gossip' and spec.topology=='ring'))
        print(f"  completed in {result['elapsed_seconds']:.1f}s", flush=True)

if __name__ == '__main__':
    main()
