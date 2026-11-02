"""Shared gossip implementation with latest-version arrival selection.

Receiver-wise renormalization is row-stochastic, but generally not doubly
stochastic after directed losses. The mean drift is measured, not hidden.
"""
import torch

class GossipAggregator:
    def __init__(self, clients, adjacency, mixing_matrix, channel):
        self.clients = {c.client_id: c for c in clients}
        self.adjacency = adjacency
        self.W = mixing_matrix
        self.channel = channel
        self._latest_received = {}
        self.mean_drift_history = []

    def _should_send(self, cid, params):
        return True

    def aggregate_round(self, round_num):
        snapshots = {cid: c.get_params().clone() for cid, c in self.clients.items()}
        triggered = {cid: self._should_send(cid, p) for cid, p in snapshots.items()}
        received = {cid: {} for cid in self.clients}
        # Include source round so old packets cannot overwrite a new model.
        arrivals = self.channel.collect_delayed(round_num)
        for sender, send in triggered.items():
            if send:
                for receiver in self.adjacency.get(sender, []):
                    if self.channel.transmit(sender, receiver, snapshots[sender], round_num):
                        arrivals.append((sender, receiver, snapshots[sender], round_num))
        for sender, receiver, params, sent_round in sorted(arrivals, key=lambda a: a[3]):
            key = (receiver, sender)
            if receiver in received and sent_round > self._latest_received.get(key, -1):
                received[receiver][sender] = params
                self._latest_received[key] = sent_round
        before = torch.stack(list(snapshots.values())).mean(0)
        for cid, client in self.clients.items():
            updates = received[cid]
            if not updates:
                continue
            weight = self.W[cid, cid] + sum(self.W[cid, j] for j in updates)
            if weight <= 1e-12:
                continue
            merged = snapshots[cid] * (self.W[cid, cid] / weight)
            for j in sorted(updates):
                merged = merged + updates[j] * (self.W[cid, j] / weight)
            client.set_params(merged)
        after = torch.stack([c.get_params() for c in self.clients.values()]).mean(0)
        self.mean_drift_history.append(float(torch.linalg.vector_norm(after - before)))
        stats = {"triggered": sum(triggered.values()), "suppressed": len(triggered) - sum(triggered.values())}
        self._after_send(snapshots, triggered, stats)
        return stats

    def _after_send(self, snapshots, triggered, stats):
        pass
