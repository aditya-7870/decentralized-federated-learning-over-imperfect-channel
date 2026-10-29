"""
gossip.py — Gossip-based decentralized aggregation engine.

Orchestrates one communication round of gossip-based federated learning:
  1. Each client broadcasts its current parameter vector to all neighbors.
  2. The ImperfectChannel decides which updates are delivered, dropped, or
     delayed.
  3. Each client averages the received neighbor parameters using
     Metropolis–Hastings weights from the mixing matrix, re-normalizing
     when some neighbors are missing.
"""

import torch
import numpy as np
from typing import Dict, List

from client import FLClient
from channel import ImperfectChannel


class GossipAggregator:
    """
    Gossip aggregation engine for one communication round.

    Parameters
    ----------
    clients : list of FLClient
        All participating clients.
    adjacency : dict
        Mapping from node_id → list of neighbor IDs.
    mixing_matrix : np.ndarray
        Row-stochastic Metropolis–Hastings weight matrix W.
    channel : ImperfectChannel
        Simulated imperfect communication channel.
    """

    def __init__(
        self,
        clients: List[FLClient],
        adjacency: Dict[int, List[int]],
        mixing_matrix: np.ndarray,
        channel: ImperfectChannel,
    ):
        self.clients = {c.client_id: c for c in clients}
        self.adjacency = adjacency
        self.W = mixing_matrix
        self.channel = channel

    def aggregate_round(self, round_num: int) -> None:
        """
        Execute one gossip communication round.

        Steps
        -----
        1. Snapshot all client parameters.
        2. Each client attempts to send its params to every neighbor via
           the channel.
        3. Collect any delayed updates from prior rounds that are due now.
        4. For each client, average own params with received neighbor params
           using mixing-matrix weights, re-normalizing for missing neighbors.
        5. Update each client's model with the aggregated parameters.
        """
        num_clients = len(self.clients)

        # 1 ── Snapshot current params before this round's communication
        snapshots: Dict[int, torch.Tensor] = {
            cid: client.get_params() for cid, client in self.clients.items()
        }

        # 2 ── Each client broadcasts to all neighbors
        #      Track which updates were delivered immediately
        #      Key: receiver_id → set of sender_ids delivered immediately
        immediate_deliveries: Dict[int, Dict[int, torch.Tensor]] = {
            cid: {} for cid in self.clients
        }

        for sender_id in self.clients:
            for neighbor_id in self.adjacency.get(sender_id, []):
                delivered = self.channel.transmit(
                    sender=sender_id,
                    receiver=neighbor_id,
                    params=snapshots[sender_id],
                    round_num=round_num,
                )
                if delivered:
                    immediate_deliveries[neighbor_id][sender_id] = snapshots[sender_id]

        # 3 ── Collect delayed updates arriving this round
        delayed_updates = self.channel.collect_delayed(round_num)
        for sender_id, receiver_id, params in delayed_updates:
            if receiver_id in immediate_deliveries:
                # Delayed update: use the stale params that were saved
                immediate_deliveries[receiver_id][sender_id] = params

        # 4 ── Aggregate: weighted average for each client
        for cid, client in self.clients.items():
            received = immediate_deliveries[cid]  # sender_id → params
            neighbors = self.adjacency.get(cid, [])

            if not neighbors:
                # Isolated node — keep own params
                continue

            # Compute re-normalized weights
            # W_ii (self-weight) + W_ij for each *received* neighbor j
            received_ids = set(received.keys())
            missing_ids = set(neighbors) - received_ids

            # Sum of weights for received neighbors
            w_received = sum(self.W[cid][j] for j in received_ids)
            w_self = self.W[cid][cid]
            w_total = w_self + w_received

            if w_total < 1e-12:
                # Fallback: keep own params
                continue

            # Re-normalize so weights sum to 1
            norm_self = w_self / w_total
            aggregated = norm_self * snapshots[cid]

            for j in received_ids:
                norm_j = self.W[cid][j] / w_total
                aggregated = aggregated + norm_j * received[j]

            # 5 ── Update client model
            client.set_params(aggregated)
