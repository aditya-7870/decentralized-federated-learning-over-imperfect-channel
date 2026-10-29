"""
channel.py — Imperfect communication channel simulation.

Models two kinds of link impairment:
  1. **Packet loss** — each transmission is independently dropped with
     probability ``packet_loss_rate``.
  2. **Asynchronous delay** — delivered updates are assigned a random
     integer delay ∈ [0, max_delay]; a buffer holds delayed updates and
     releases them in the appropriate future round.

Every transmission event (sent / dropped / delayed / delivered) is logged
so that communication statistics can be computed later.
"""

import numpy as np
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, field


@dataclass
class TransmissionEvent:
    """Record of a single transmission attempt."""
    round_num: int
    sender: int
    receiver: int
    dropped: bool
    delay: int              # 0 = delivered immediately
    delivery_round: int     # round_num + delay
    bytes_size: int         # size of the parameter vector in bytes


class ImperfectChannel:
    """
    Simulates an imperfect communication link between nodes.

    Parameters
    ----------
    packet_loss_rate : float
        Probability of dropping any single update.
    max_async_delay : int
        Maximum random delay in rounds (0 means synchronous delivery).
    seed : int
        Random seed for reproducibility.
    param_bytes : int
        Size of one parameter vector in bytes (used for communication
        cost accounting).
    """

    def __init__(
        self,
        packet_loss_rate: float = 0.0,
        max_async_delay: int = 0,
        seed: int = 42,
        param_bytes: int = 0,
    ):
        self.packet_loss_rate = packet_loss_rate
        self.max_async_delay = max_async_delay
        self.rng = np.random.default_rng(seed)
        self.param_bytes = param_bytes

        # Delay buffer: maps delivery_round → list of (sender, receiver, params)
        self._delay_buffer: Dict[int, List[Tuple[int, int, Any]]] = {}

        # Full transmission log
        self.log: List[TransmissionEvent] = []

    # ── Core API ─────────────────────────────────────────────────────────

    def transmit(
        self,
        sender: int,
        receiver: int,
        params: Any,
        round_num: int,
    ) -> bool:
        """
        Attempt to send *params* from *sender* to *receiver* at *round_num*.

        Returns ``True`` if the update is delivered immediately (delay = 0),
        ``False`` if it was dropped or delayed (delayed updates will appear
        in ``collect_delayed`` in a future round).
        """
        # Packet loss check
        dropped = self.rng.random() < self.packet_loss_rate

        if dropped:
            self.log.append(TransmissionEvent(
                round_num=round_num,
                sender=sender,
                receiver=receiver,
                dropped=True,
                delay=0,
                delivery_round=-1,
                bytes_size=self.param_bytes,
            ))
            return False

        # Determine delay
        delay = 0
        if self.max_async_delay > 0:
            delay = int(self.rng.integers(0, self.max_async_delay + 1))

        delivery_round = round_num + delay

        self.log.append(TransmissionEvent(
            round_num=round_num,
            sender=sender,
            receiver=receiver,
            dropped=False,
            delay=delay,
            delivery_round=delivery_round,
            bytes_size=self.param_bytes,
        ))

        if delay == 0:
            # Immediate delivery
            return True
        else:
            # Store in delay buffer for future delivery
            self._delay_buffer.setdefault(delivery_round, []).append(
                (sender, receiver, params)
            )
            return False

    def collect_delayed(
        self, current_round: int
    ) -> List[Tuple[int, int, Any]]:
        """
        Retrieve all delayed updates scheduled for delivery at *current_round*.

        Returns a list of ``(sender, receiver, params)`` tuples.
        """
        return self._delay_buffer.pop(current_round, [])

    # ── Statistics ───────────────────────────────────────────────────────

    def get_round_stats(self, round_num: int) -> Dict[str, int]:
        """
        Return communication statistics for a specific round.

        Keys: ``attempted``, ``dropped``, ``delivered_immediate``,
        ``delayed``, ``bytes_attempted``, ``bytes_delivered``.
        """
        events = [e for e in self.log if e.round_num == round_num]
        attempted = len(events)
        dropped = sum(1 for e in events if e.dropped)
        delayed = sum(1 for e in events if not e.dropped and e.delay > 0)
        delivered_immediate = attempted - dropped - delayed
        bytes_attempted = sum(e.bytes_size for e in events)
        bytes_delivered = sum(e.bytes_size for e in events if not e.dropped)

        return {
            "attempted": attempted,
            "dropped": dropped,
            "delivered_immediate": delivered_immediate,
            "delayed": delayed,
            "bytes_attempted": bytes_attempted,
            "bytes_delivered": bytes_delivered,
        }

    def get_total_stats(self) -> Dict[str, int]:
        """Aggregate communication statistics across all rounds."""
        total = {
            "attempted": 0,
            "dropped": 0,
            "delivered_immediate": 0,
            "delayed": 0,
            "bytes_attempted": 0,
            "bytes_delivered": 0,
        }
        for e in self.log:
            total["attempted"] += 1
            total["bytes_attempted"] += e.bytes_size
            if e.dropped:
                total["dropped"] += 1
            elif e.delay > 0:
                total["delayed"] += 1
                total["bytes_delivered"] += e.bytes_size
            else:
                total["delivered_immediate"] += 1
                total["bytes_delivered"] += e.bytes_size
        return total

    def reset(self) -> None:
        """Clear all logs and the delay buffer."""
        self.log.clear()
        self._delay_buffer.clear()
