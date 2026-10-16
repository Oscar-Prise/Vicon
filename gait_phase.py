"""Per-leg gait-phase tracker for the control loop.

The Vicon server sends percent_gc (0-1, time since heel strike / stride time)
at ~30 Hz, but the control loop runs at 100 Hz. Without tracking, ~3 of every
4 loop iterations reuse a stale value, so the torque command is a staircase
and lags the real phase by up to one update interval (plus network latency).

GaitPhaseTracker advances the last received phase at the measured phase rate
between updates, so the lookup uses the phase *now* instead of the phase at the
last packet. A new packet always overrides the extrapolation, so heel-strike
resets from the server are followed exactly.
"""

import numpy as np


class GaitPhaseTracker:
	def __init__(
		self,
		rate_alpha: float = 0.2,
		min_rate: float = 0.3,   # strides/s (~3.3 s stride)
		max_rate: float = 3.0,   # strides/s (~0.33 s stride)
		max_extrapolation_s: float = 0.15,
	):
		self.rate_alpha = rate_alpha
		self.min_rate = min_rate
		self.max_rate = max_rate
		self.max_extrapolation_s = max_extrapolation_s

		self.rate = None        # strides per second (phase units/s)
		self._last_stamp = None  # server timestamp of last packet
		self._raw = 0.0
		self._recv_time = None

	def update(self, raw_gc: float, server_stamp: float, recv_time: float, now: float) -> float:
		"""Return the phase estimate in [0, 1] for time `now`.

		raw_gc       latest percent_gc from the server (0-1)
		server_stamp vicon_timestamp of that packet (used only to detect new packets)
		recv_time    local time.time() when that packet arrived
		now          local time.time() now
		"""
		raw_gc = float(np.clip(raw_gc, 0.0, 1.0))

		if server_stamp != self._last_stamp:  # new packet
			if self._last_stamp is not None and self._recv_time is not None:
				dt = recv_time - self._recv_time
				d = raw_gc - self._raw
				# Use only forward motion that is not a stall at 1.0 or a wrap.
				if dt > 1e-3 and d > 1e-4:
					inst = d / dt
					if self.min_rate <= inst <= self.max_rate:
						self.rate = inst if self.rate is None else (
							(1 - self.rate_alpha) * self.rate + self.rate_alpha * inst
						)
			self._last_stamp = server_stamp
			self._raw = raw_gc
			self._recv_time = recv_time

		if self.rate is None or self._recv_time is None:
			return self._raw

		age = float(np.clip(now - self._recv_time, 0.0, self.max_extrapolation_s))
		# Never extrapolate past the end of the stride: the server signals the wrap.
		return float(min(self._raw + self.rate * age, 1.0))
