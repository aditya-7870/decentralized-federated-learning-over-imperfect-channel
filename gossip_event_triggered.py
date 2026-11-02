"""Event-triggered gossip sharing exactly the M1 message and averaging path."""
import torch
from gossip import GossipAggregator

class EventTriggeredGossipAggregator(GossipAggregator):
    def __init__(self, clients, adjacency, mixing_matrix, channel, delta=0.1):
        super().__init__(clients, adjacency, mixing_matrix, channel)
        if delta < 0:
            raise ValueError("delta must be nonnegative")
        self.delta = delta
        self._last_sent = {cid: c.get_params().clone() for cid, c in self.clients.items()}
        self.trigger_history = []

    def _should_send(self, cid, params):
        return self.delta == 0 or float(torch.linalg.vector_norm(params - self._last_sent[cid])) > self.delta

    def _after_send(self, snapshots, triggered, stats):
        for cid, did_send in triggered.items():
            if did_send:
                self._last_sent[cid] = snapshots[cid].clone()
        self.trigger_history.append(stats)

    def reset_last_sent(self):
        self._last_sent = {cid: c.get_params().clone() for cid, c in self.clients.items()}
        self.trigger_history.clear()
