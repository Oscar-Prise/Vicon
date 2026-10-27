"""Single-motor CAN check: is the bus up, and does the motor answer?

With test_torque_Nm = 0 the motor stays limp: move it by hand and watch
pos/vel/torque in the terminal and in Teleplot. Set test_torque_Nm to a nonzero
value to command a constant torque (ramped in after torque_start_sec). Works with
one motor on the bus (the other can be unplugged). Ctrl+C stops it and disables
the motor.

Run:  python util_test_singleMotor.py
"""
import signal
import subprocess
import sys
import time

import can
import numpy as np

from utils_teleplot import Teleplot

# =============================================================================
# Configuration — edit before each test
# =============================================================================

# Motor: "ak80-9" (CubeMars) or "robstride02"
motor_type = "robstride02"
can_id = 1
motor_channel = "can0"
bring_up_can = True         # set False if can0 is already up
zero_on_start = True        # current position becomes 0 deg

# Torque command (Nm). 0 = limp. Sign follows the motor's own convention.
test_torque_Nm = 5.0
torque_limit = 5.0          # hard clip on the command (Nm)
torque_start_sec = 2.0      # hold 0 Nm this long before applying torque
torque_ramp_sec = 1.0       # ramp from 0 to test_torque_Nm over this time (0 = step)

control_freq_Hz = 100
duration_sec = 60
print_every_sec = 0.2

# Teleplot (live UDP telemetry)
teleplot_host = "127.0.0.1"
teleplot_port = 47269

# =============================================================================

RAD2DEG = 180.0 / np.pi


def can_link_state(channel):
    """Controller state from `ip` (ERROR-ACTIVE is healthy; BUS-OFF / ERROR-PASSIVE are not)."""
    out = subprocess.run(["ip", "-details", "link", "show", channel],
                         capture_output=True, text=True).stdout
    if not out:
        return "NOT FOUND"
    up = "UP" if ",UP" in out or "<UP" in out else "DOWN"
    state = next((w for w in ("ERROR-ACTIVE", "ERROR-WARNING", "ERROR-PASSIVE", "BUS-OFF", "STOPPED")
                  if w in out), "unknown")
    return f"{up}, {state}"


class CubeMarsSingle:
    def __init__(self, can_id, channel):
        from motors.cubemars import CubeMarsMotorGroup, _Motor
        self._group = CubeMarsMotorGroup(channel=channel)
        self.motor = _Motor(can_id)
        self.channel = channel
        self.bus = self.notifier = None
        self._last_n = 0

    def setup_can(self):
        self._group._setup_can()

    def connect(self):
        self.bus = can.Bus(channel=self.channel, interface="socketcan",
                           receive_own_messages=False, local_loopback=False)
        self.notifier = can.Notifier(self.bus, [self.motor])
        self.motor.bus = self.bus
        if zero_on_start:
            self.motor.zero_encoder()
            time.sleep(0.1)
        self.motor.set_torque(0.0)   # zero-torque MIT frame also starts the reply stream
        time.sleep(0.3)
        if self.motor.n_replies == 0:
            raise RuntimeError("no reply")

    def step(self, torque=0.0):
        """Send `torque` Nm, return (ok, pos_deg, vel_dps, torque, temp)."""
        self.motor.set_torque(torque)
        ok = self.motor.n_replies > self._last_n
        self._last_n = self.motor.n_replies
        m = self.motor
        return ok, m.pos * RAD2DEG, m.vel * RAD2DEG, m.cur, float("nan")

    def disconnect(self):
        if self.bus is None:
            return
        try:
            self.motor.set_torque(0.0)
            time.sleep(0.05)
        finally:
            self.notifier.stop()
            self.bus.shutdown()
            self.bus = None


class RobStrideSingle:
    NAME = "MOTOR"

    def __init__(self, can_id, channel):
        from motors.robstride import RobStrideMotorGroup
        from motors.robstride_dynamics import Motor, RobstrideBus
        self._group = RobStrideMotorGroup(channel=channel)
        self.bus = RobstrideBus(channel, {self.NAME: Motor(id=can_id, model="rs-02")})
        self.offset = 0.0

    def setup_can(self):
        self._group._setup_can()

    def connect(self):
        self.bus.connect()
        self.bus.enable(self.NAME)   # raises if the motor does not answer
        time.sleep(0.2)
        if zero_on_start:
            ok, pos, *_ = self.step()
            if ok:
                self.offset = pos

    def step(self, torque=0.0):
        """Send `torque` Nm, return (ok, pos_deg, vel_dps, torque, temp)."""
        self.bus.write_operation_frame(self.NAME, 0, 0, 0, 0, torque)
        try:
            pos, vel, trq, temp = self.bus.read_operation_frame(self.NAME)
        except (RuntimeError, AssertionError):
            return False, float("nan"), float("nan"), float("nan"), float("nan")
        return True, pos * RAD2DEG - self.offset, vel * RAD2DEG, trq, temp

    def disconnect(self):
        if self.bus.is_connected:
            self.bus.write_operation_frame(self.NAME, 0, 0, 0, 0, 0.0)
            time.sleep(0.05)
            self.bus.disconnect(disable_torque=True)


def safe_disconnect(motor):
    try:
        motor.disconnect()
    except Exception as e:   # e.g. the disable frame gets no reply from an unpowered motor
        print(f"(disconnect: {e})")


def make_motor():
    kind = motor_type.lower()
    if kind == "ak80-9":
        return CubeMarsSingle(can_id, motor_channel)
    if kind == "robstride02":
        return RobStrideSingle(can_id, motor_channel)
    raise ValueError(f"Unknown motor_type {motor_type!r}")


def main():
    motor = make_motor()
    teleplot = Teleplot(teleplot_host, teleplot_port)
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))   # SystemExit runs the cleanup below

    # 1. CAN interface
    if bring_up_can:
        motor.setup_can()
    state = can_link_state(motor_channel)
    print(f"[1/2] CAN link {motor_channel}: {state}")
    if not state.startswith("UP"):
        print("      CAN interface is not up. Check the CAN adapter / transceiver, then retry.")
        return

    # 2. Motor answers?
    print(f"[2/2] Talking to {motor_type} CAN ID {can_id} (0 Nm commanded during connect)...")
    try:
        motor.connect()
    except Exception as e:
        print(f"      No answer from motor ID {can_id}: {e}")
        print("      Check motor power, CAN H/L wiring, termination, and the CAN ID.")
        print(f"      CAN link now: {can_link_state(motor_channel)}")
        safe_disconnect(motor)
        return
    target = float(np.clip(test_torque_Nm, -torque_limit, torque_limit))
    if target != test_torque_Nm:
        print(f"      test_torque_Nm {test_torque_Nm} clipped to {target} (torque_limit {torque_limit})")
    if target == 0.0:
        print("      Motor answered. Move it by hand; Ctrl+C to stop.\n")
    else:
        print(f"      Motor answered. Applying {target:+.2f} Nm after {torque_start_sec:.1f}s "
              f"(ramp {torque_ramp_sec:.1f}s); Ctrl+C to stop.\n")
    print(f"{'t(s)':>6} | {'cmd(Nm)':>8} | {'pos(deg)':>9} {'vel(dps)':>9} {'trq(Nm)':>8} {'temp(C)':>7} | {'rx Hz':>6}")

    t0 = time.time()
    k = 0
    next_print = 0.0
    window_ok, window_n = 0, 0
    try:
        while time.time() - t0 < duration_sec:
            t = time.time() - t0
            if t < torque_start_sec:
                cmd = 0.0
            elif torque_ramp_sec > 0:
                cmd = target * min((t - torque_start_sec) / torque_ramp_sec, 1.0)
            else:
                cmd = target
            ok, pos, vel, trq, temp = motor.step(cmd)
            window_ok += ok
            window_n += 1

            teleplot.sendValue('cmd', cmd)
            teleplot.sendValue('rx_ok', int(ok))
            if ok:
                teleplot.sendValue('pos', pos)
                teleplot.sendValue('vel', vel)
                teleplot.sendValue('trq', trq)
                if not np.isnan(temp):
                    teleplot.sendValue('temp', temp)

            if t >= next_print:
                rx_hz = window_ok / max(window_n, 1) * control_freq_Hz
                teleplot.sendValue('rx_Hz', rx_hz)
                print(f"{t:6.2f} | {cmd:8.2f} | {pos:9.2f} {vel:9.2f} {trq:8.2f} {temp:7.1f} | {rx_hz:6.0f}"
                      + ("" if window_ok else "   <- NO REPLIES"))
                window_ok, window_n = 0, 0
                next_print += print_every_sec

            k += 1
            while (time.time() - t0) < k / control_freq_Hz:
                pass
    except KeyboardInterrupt:
        pass
    finally:
        safe_disconnect(motor)
        teleplot.close()
        print(f"\nDone: {k} loops in {time.time() - t0:.1f}s. CAN link: {can_link_state(motor_channel)}")


if __name__ == "__main__":
    main()
