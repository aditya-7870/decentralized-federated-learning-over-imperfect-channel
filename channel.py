"""Discrete-round erasure channel with reproducible directed-link randomness.

Gilbert-Elliott state advances once per link per round, even while its sender
is silent. Delivery means arrival by the observation round, not scheduling.
"""
from dataclasses import dataclass
import numpy as np

@dataclass
class TransmissionEvent:
    round_num: int
    sender: int
    receiver: int
    dropped: bool
    delay: int
    delivery_round: int
    bytes_size: int

class ImperfectChannel:
    def __init__(self, packet_loss_rate=0.0, max_async_delay=0, seed=42,
                 param_bytes=0, loss_model="bernoulli", good_to_bad=0.05,
                 bad_to_good=0.2):
        if not 0 <= packet_loss_rate <= 1 or max_async_delay < 0:
            raise ValueError("Invalid loss probability or delay")
        if loss_model not in ("bernoulli", "gilbert-elliott"):
            raise ValueError("Unknown channel model")
        if not 0 < good_to_bad <= 1 or not 0 < bad_to_good <= 1:
            raise ValueError("Invalid state transition probabilities")
        self.packet_loss_rate = packet_loss_rate
        self.max_async_delay = max_async_delay
        self.seed = seed
        self.param_bytes = param_bytes
        self.loss_model = loss_model
        self.good_to_bad = good_to_bad
        self.bad_to_good = bad_to_good
        self.log = []
        self._delay_buffer = {}
        self.current_round = 0
        self._states = {}

    def _random(self, sender, receiver, rnd, stream):
        return np.random.default_rng(np.random.SeedSequence(
            [self.seed, sender, receiver, rnd, stream]))

    def _is_bad(self, sender, receiver, rnd):
        key = (sender, receiver)
        if key not in self._states:
            stationary = self.good_to_bad / (self.good_to_bad + self.bad_to_good)
            bad = self._random(sender, receiver, 0, 3).random() < stationary
            self._states[key] = (0, bad)
        last, bad = self._states[key]
        for step in range(last + 1, rnd + 1):
            u = self._random(sender, receiver, step, 3).random()
            bad = (u >= self.bad_to_good) if bad else (u < self.good_to_bad)
        self._states[key] = (rnd, bad)
        return bad

    def transmit(self, sender, receiver, params, round_num):
        self.current_round = max(self.current_round, round_num)
        if self.loss_model == "gilbert-elliott":
            # Good state: no erasure; bad state: all packets erased.
            dropped = self._is_bad(sender, receiver, round_num)
        else:
            dropped = self._random(sender, receiver, round_num, 1).random() < self.packet_loss_rate
        delay = 0 if dropped else int(self._random(sender, receiver, round_num, 2).integers(0, self.max_async_delay + 1))
        arrival = -1 if dropped else round_num + delay
        self.log.append(TransmissionEvent(round_num, sender, receiver, bool(dropped), delay, arrival, self.param_bytes))
        if dropped:
            return False
        if delay:
            self._delay_buffer.setdefault(arrival, []).append((sender, receiver, params, round_num))
            return False
        return True

    def collect_delayed(self, current_round):
        self.current_round = max(self.current_round, current_round)
        return self._delay_buffer.pop(current_round, [])

    def get_round_stats(self, round_num):
        events = [e for e in self.log if e.round_num == round_num]
        arrivals = [e for e in self.log if not e.dropped and e.delivery_round == round_num]
        return {
            "attempted": len(events), "dropped": sum(e.dropped for e in events),
            "delivered_immediate": sum(not e.dropped and e.delay == 0 for e in events),
            "delayed": sum(not e.dropped and e.delay > 0 for e in events),
            "delivered": len(arrivals),
            "bytes_attempted": sum(e.bytes_size for e in events),
            "bytes_delivered": sum(e.bytes_size for e in arrivals),
        }

    def get_total_stats(self):
        arrived = [e for e in self.log if not e.dropped and e.delivery_round <= self.current_round]
        return {
            "attempted": len(self.log), "dropped": sum(e.dropped for e in self.log),
            "delivered_immediate": sum(not e.dropped and e.delay == 0 for e in self.log),
            "delayed": sum(not e.dropped and e.delay > 0 for e in self.log),
            "delivered": len(arrived),
            "pending": sum(not e.dropped and e.delivery_round > self.current_round for e in self.log),
            "bytes_attempted": sum(e.bytes_size for e in self.log),
            "bytes_delivered": sum(e.bytes_size for e in arrived),
        }

    def reset(self):
        self.log.clear()
        self._delay_buffer.clear()
        self._states.clear()
        self.current_round = 0
