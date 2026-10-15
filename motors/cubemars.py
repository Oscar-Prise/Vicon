"""CubeMars AK80-9 (TMotor V3, MIT mode) interface for Vicon exoskeleton protocols.

Stripped down from AMP-Hip/actuators (Utils_tmotor_v3.py + Utils_actuator_group.py):
torque-only control, two motors, no reconnect/guard machinery, no signal handlers.
Exposes the same interface as the RobStride group so main.py is unchanged:
connect / disconnect / set_torque(L, R) / update_readings() -> degrees.
"""
import os
import platform
import struct
import time
import numpy as np
import can

RAD2DEG = 180.0 / np.pi
DEG2RAD = np.pi / 180.0

MOTOR_L = "MOTOR_L"
MOTOR_R = "MOTOR_R"

# MIT-mode CAN protocol constants (AK Series Module Product Manual V3.0.1, AK80-9)
MIT_MODE_ID = 8
ORIGIN_SET_ID = 5
REPLY_ID_BASE = 0x2900
P_MIN, P_MAX = -12.56, 12.56   # rad
V_MIN, V_MAX = -65.0, 65.0     # rad/s
T_MIN, T_MAX = -18.0, 18.0     # N-m
KP_MIN, KP_MAX = 0.0, 500.0
KD_MIN, KD_MAX = 0.0, 5.0


def _clamp(x, lo, hi):
    return max(lo, min(hi, x))


def _float_to_uint(x, lo, hi, bits):
    return int(((x - lo) / float(hi - lo)) * ((1 << bits) - 1))


def _mit_torque_message(can_id: int, torque: float) -> can.Message:
    """Pure torque command: pos = vel = kp = kd = 0."""
    p = _float_to_uint(0.0, P_MIN, P_MAX, 16)
    v = _float_to_uint(0.0, V_MIN, V_MAX, 12)
    kp = _float_to_uint(0.0, KP_MIN, KP_MAX, 12)
    kd = _float_to_uint(0.0, KD_MIN, KD_MAX, 12)
    t = _float_to_uint(_clamp(torque, T_MIN, T_MAX), T_MIN, T_MAX, 12)
    data = [
        kp >> 4,
        ((kp & 0xF) << 4) | (kd >> 8),
        kd & 0xFF,
        p >> 8,
        p & 0xFF,
        v >> 4,
        ((v & 0xF) << 4) | (t >> 8),
        t & 0xFF,
    ]
    return can.Message(arbitration_id=MIT_MODE_ID << 8 | can_id, data=data, is_extended_id=True)


def _set_origin_message(can_id: int) -> can.Message:
    return can.Message(arbitration_id=ORIGIN_SET_ID << 8 | can_id, data=[0] * 8, is_extended_id=True)


class _Motor(can.Listener):
    """One AK80-9. Caches the latest reply frame (pos/vel rad, current A)."""

    def __init__(self, can_id: int):
        self.can_id = can_id
        self.bus = None
        self.pos = 0.0   # rad
        self.vel = 0.0   # rad/s
        self.cur = 0.0   # A (reported as torque, as in AMP-Hip)
        self.n_replies = 0

    def on_message_received(self, msg: can.Message) -> None:
        if msg.arbitration_id != REPLY_ID_BASE + self.can_id:
            return
        self.n_replies += 1
        d = msg.data
        self.pos = struct.unpack(">h", bytes(d[0:2]))[0] * 0.1 * DEG2RAD
        self.vel = struct.unpack(">h", bytes(d[2:4]))[0] * 10.0 * DEG2RAD
        self.cur = struct.unpack(">h", bytes(d[4:6]))[0] * 0.01

    def set_torque(self, torque: float) -> None:
        self.bus.send(_mit_torque_message(self.can_id, torque))

    def zero_encoder(self) -> None:
        self.bus.send(_set_origin_message(self.can_id))


class CubeMarsMotorGroup:
    def __init__(
        self,
        can_id_L: int = 1,
        can_id_R: int = 2,
        channel: str = "can0",
        torque_limit: float = 17.0,
        control_freq_Hz: float = 100,
    ):
        self.channel = channel
        self.torque_limit = min(torque_limit, T_MAX)
        self.control_freq_Hz = control_freq_Hz

        self.motors = {MOTOR_L: _Motor(can_id_L), MOTOR_R: _Motor(can_id_R)}
        self.bus = None
        self.notifier = None

    def _setup_can(self) -> None:
        """Bring can0 up at 1 Mbit/s. On Jetson (tegra) the CAN modules are
        reloaded first, otherwise error-filled buffers survive a restart."""
        ip = "sudo /sbin/ip link set can0"
        if "tegra" in platform.uname().release.lower():
            for cmd in ("sudo modprobe can", "sudo modprobe can_raw", "sudo modprobe mttcan",
                        f"{ip} down", "sudo rmmod can_raw", "sudo rmmod can", "sudo rmmod mttcan",
                        "sudo modprobe can", "sudo modprobe can_raw", "sudo modprobe mttcan"):
                os.system(cmd)
        os.system(f"{ip} down")
        rc = os.system(f"{ip} txqueuelen 1000 up type can bitrate 1000000")
        if rc != 0:
            raise RuntimeError(f"Failed to bring up {self.channel} (rc={rc})")

    def connect(self, prompt: bool = True) -> None:
        if prompt:
            input("Stand in the NEUTRAL pose, then press Enter to zero CubeMars AK80-9 motors: ")
        self._setup_can()
        self.bus = can.Bus(channel=self.channel, bustype="socketcan",
                           receive_own_messages=False, local_loopback=False)
        self.notifier = can.Notifier(self.bus, list(self.motors.values()))
        for m in self.motors.values():
            m.bus = self.bus
            m.zero_encoder()   # current position becomes 0 (replaces offset calibration)
        time.sleep(0.1)
        # A zero-torque MIT frame enables the motor and starts its reply stream.
        self.set_torque(0.0, 0.0)
        time.sleep(0.5)
        for name, m in self.motors.items():
            if m.n_replies == 0:
                print(f"WARNING: {name} (CAN ID {m.can_id}) did not reply; its readings will stay 0. "
                      f"Check power, wiring and the motor's CAN ID.")

    def disconnect(self) -> None:
        if self.bus is None:
            return
        try:
            self.set_torque(0.0, 0.0)
            time.sleep(0.1)
        except Exception:
            pass
        self.notifier.stop()
        self.bus.shutdown()
        self.bus = None

    def set_torque(self, torque_l: float, torque_r: float) -> None:
        self.motors[MOTOR_L].set_torque(float(np.clip(torque_l, -self.torque_limit, self.torque_limit)))
        self.motors[MOTOR_R].set_torque(float(np.clip(torque_r, -self.torque_limit, self.torque_limit)))

    def update_readings(self, degrees: bool = True):
        """Latest position, velocity, torque for both motors (no CAN read; the
        notifier thread updates them as reply frames arrive)."""
        L, R = self.motors[MOTOR_L], self.motors[MOTOR_R]
        k = RAD2DEG if degrees else 1.0
        return L.pos * k, L.vel * k, L.cur, R.pos * k, R.vel * k, R.cur
