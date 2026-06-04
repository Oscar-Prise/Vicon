"""RobStride RS-02 motor interface for Vicon exoskeleton protocols."""

import os
import time

import numpy as np

from robstride_dynamics import Motor, RobstrideBus

RAD_TO_DEG = 180.0 / np.pi

MOTOR_LEFT = "motor_left"
MOTOR_RIGHT = "motor_right"


class RobStrideMotorGroup:
    def __init__(
        self,
        can_id_L: int = 1,
        can_id_R: int = 2,
        channel: str = "can0",
        torque_limit: float = 17.0,
        offset_samples: int = 50,
    ):
        self.channel = channel
        self.torque_limit = torque_limit
        self.offset_samples = offset_samples
        self.motors = {
            MOTOR_LEFT: Motor(id=can_id_L, model="rs-02"),
            MOTOR_RIGHT: Motor(id=can_id_R, model="rs-02"),
        }

        self._offsets = {MOTOR_LEFT: 0.0, MOTOR_RIGHT: 0.0}
        self.bus = None

    def _setup_can(self) -> None:
        cmds = [
            f"sudo ip link set {self.channel} down",
            (
                f"sudo ip link set {self.channel} type can bitrate 1000000 "
                "restart-ms 100 berr-reporting off"
            ),
            f"sudo ip link set {self.channel} up",
        ]
        for cmd in cmds:
            rc = os.system(cmd)
            if rc != 0:
                raise RuntimeError(f"Command failed (rc={rc}): {cmd}")

    def connect(self) -> None:
        self._setup_can()
        self.bus = RobstrideBus(self.channel, self.motors)
        self.bus.connect()
        self.bus.enable(MOTOR_LEFT)
        self.bus.enable(MOTOR_RIGHT)
        time.sleep(0.3)
        self._calibrate_offsets()

    def disconnect(self) -> None:
        if self.bus is not None and self.bus.is_connected:
            self.bus.disconnect(disable_torque=True)

    def _calibrate_offsets(self) -> None:
        offsets = {MOTOR_LEFT: [], MOTOR_RIGHT: []}
        for _ in range(self.offset_samples):
            for motor in (MOTOR_LEFT, MOTOR_RIGHT):
                pos, _, _, _ = self.bus.read_operation_frame(motor)
                offsets[motor].append(pos)
            time.sleep(0.01)

        for motor in (MOTOR_LEFT, MOTOR_RIGHT):
            self._offsets[motor] = float(np.mean(offsets[motor]))

        print(
            "RobStride zero offsets (rad): "
            f"left={self._offsets[MOTOR_LEFT]:.4f}, "
            f"right={self._offsets[MOTOR_RIGHT]:.4f}"
        )

    def _apply_position_offset(self, motor: str, position: float) -> float:
        return position - self._offsets[motor]

    def update_readings(self, degrees: bool = True):
        """Read position, velocity, and torque for both motors (2 CAN reads)."""
        pos_l, vel_l, tor_l, _ = self.bus.read_operation_frame(MOTOR_LEFT)
        pos_r, vel_r, tor_r, _ = self.bus.read_operation_frame(MOTOR_RIGHT)

        pos_l = self._apply_position_offset(MOTOR_LEFT, pos_l)
        pos_r = self._apply_position_offset(MOTOR_RIGHT, pos_r)

        if degrees:
            pos_l *= RAD_TO_DEG
            vel_l *= RAD_TO_DEG
            pos_r *= RAD_TO_DEG
            vel_r *= RAD_TO_DEG

        return pos_l, vel_l, tor_l, pos_r, vel_r, tor_r

    def set_torque_lr(self, torque_l: float, torque_r: float) -> None:
        torque_l = float(np.clip(torque_l, -self.torque_limit, self.torque_limit))
        torque_r = float(np.clip(torque_r, -self.torque_limit, self.torque_limit))
        self.bus.write_operation_frame(MOTOR_LEFT, 0, 0, 0, 0, -torque_l)
        self.bus.write_operation_frame(MOTOR_RIGHT, 0, 0, 0, 0, torque_r)
