"""
gossip_event_triggered.py — Event-Triggered Asynchronous Gossip for M2.

Each client broadcasts its model update to neighbors ONLY when the change
in its parameters since the last transmission exceeds a threshold delta.

Algorithm (per-client, per-round):
    1.  Do local SGD training.
    2.  Compute change = ||w_i(t) - w_i(t_last_sent)||_2
    3.  If change > delta:  broadcast w_i to all neighbors  (triggered)
        Else:               suppress transmission           (silent)
    4.  Each receiver that got at least one update averages
        received parameters using Metropolis-Hastings weights.
    5.  Clients that received nothing keep their own parameters.

This reduces communication overhead by skipping rounds where the model
has not changed enough to warrant re-synchronisation.
"""

import torch
import numpy as np
from typing import Dict, List, Optional

from client import FLClient
from channel import ImperfectChannel


class EventTriggeredGossipAggregator:
    """
    Event-triggered asynchronous gossip aggregation engine.

    Parameters
    ----------
    clients : list of FLClient
        All participating FL clients.
    adjacency : dict
        node_id -> list of neighbor IDs.
    mixing_matrix : np.ndarray
        Row-stochastic Metropolis-Hastings weight matrix W.
    channel : ImperfectChannel
        Imperfect communication channel (packet loss + async delay).
    delta : float
        Trigger threshold.  A client transmits only when
        ||w(t) - w_last_sent||_2 > delta.
    """

    def __init__(
        self,
        clients: List[FLClient],
        adjacency: Dict[int, List[int]],
        mixing_matrix: np.ndarray,
        channel: ImperfectChannel,
        delta: float = 0.1,
    ):
        self.clients       = {c.client_id: c for c in clients}
        self.adjacency     = adjacency
        self.W             = mixing_matrix
        self.channel       = channel
        self.delta         = delta

        # Last-transmitted parameters per client (initialised to current params)
        self._last_sent: Dict[int, torch.Tensor] = {
            cid: client.get_params().clone()
            for cid, client in self.clients.items()
        }

        # Per-round bookkeeping
        self.trigger_history: List[Dict] = []   # how many triggered each round

    # ── Core API ─────────────────────────────────────────────────────────

    def aggregate_round(self, round_num: int) -> Dict[str, int]:
        """
        Execute one event-triggered gossip round.

        Steps
        -----
        1. Snapshot current parameters.
        2. Decide, per client, whether to transmit (trigger check).
        3. Transmit (through channel) for triggered clients only.
        4. Collect delayed updates from prior rounds.
        5. Aggregate: weighted average for clients that received updates.
        6. Update _last_sent only for clients that transmitted.

        Returns
        -------
        dict
            Round-level trigger statistics: triggered, suppressed.
        """
        # 1 ── Snapshot
        snapshots: Dict[int, torch.Tensor] = {
            cid: client.get_params().clone()
            for cid, client in self.clients.items()
        }

        # 2 ── Trigger check: compare to last-transmitted params
        triggered: Dict[int, bool] = {}
        for cid, params in snapshots.items():
            change = torch.norm(params - self._last_sent[cid]).item()
            triggered[cid] = change > self.delta

        n_triggered  = sum(triggered.values())
        n_suppressed = len(triggered) - n_triggered

        # 3 ── Transmit only for triggered clients
        immediate_deliveries: Dict[int, Dict[int, torch.Tensor]] = {
            cid: {} for cid in self.clients
        }

        for sender_id, should_send in triggered.items():
            if not should_send:
                continue
            for neighbor_id in self.adjacency.get(sender_id, []):
                delivered = self.channel.transmit(
                    sender    = sender_id,
                    receiver  = neighbor_id,
                    params    = snapshots[sender_id],
                    round_num = round_num,
                )
                if delivered:
                    immediate_deliveries[neighbor_id][sender_id] = snapshots[sender_id]

        # 4 ── Collect delayed updates from prior rounds
        delayed = self.channel.collect_delayed(round_num)
        for sender_id, receiver_id, params in delayed:
            if receiver_id in immediate_deliveries:
                immediate_deliveries[receiver_id][sender_id] = params

        # 5 ── Aggregate: weighted average only for clients that received updates
        for cid, client in self.clients.items():
            received  = immediate_deliveries[cid]
            neighbors = self.adjacency.get(cid, [])

            if not neighbors or not received:
                # No neighbors or no updates received — keep own params
                continue

            received_ids = set(received.keys())
            w_received   = sum(self.W[cid][j] for j in received_ids)
            w_self       = self.W[cid][cid]
            w_total      = w_self + w_received

            if w_total < 1e-12:
                continue

            norm_self  = w_self / w_total
            aggregated = norm_self * snapshots[cid]

            for j in received_ids:
                norm_j     = self.W[cid][j] / w_total
                aggregated = aggregated + norm_j * received[j]

            client.set_params(aggregated)

        # 6 ── Update last-sent only for clients that transmitted
        for cid, did_send in triggered.items():
            if did_send:
                self._last_sent[cid] = snapshots[cid].clone()

        stats = {"triggered": n_triggered, "suppressed": n_suppressed}
        self.trigger_history.append(stats)
        return stats

    def reset_last_sent(self) -> None:
        """Re-initialise last-sent to current client parameters (new experiment)."""
        self._last_sent = {
            cid: client.get_params().clone()
            for cid, client in self.clients.items()
        }
        self.trigger_history.clear()
