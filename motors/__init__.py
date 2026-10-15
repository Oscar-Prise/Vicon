"""Motor backends. Each group exposes the same interface:

    connect() / disconnect()
    set_torque(torque_l, torque_r)
    update_readings() -> (pos_l, vel_l, tor_l, pos_r, vel_r, tor_r)  # deg, deg/s, Nm
"""

MOTOR_TYPES = ("ak80-9", "robstride02")


def make_motor_group(motor_type, can_id_L, can_id_R, channel, torque_limit, control_freq_Hz):
    """Build the motor group for `motor_type`. Imports lazily so only the chosen
    backend's dependencies need to be installed."""
    kind = motor_type.lower()
    if kind == "ak80-9":
        from .cubemars import CubeMarsMotorGroup
        return CubeMarsMotorGroup(
            can_id_L=can_id_L, can_id_R=can_id_R, channel=channel,
            torque_limit=torque_limit, control_freq_Hz=control_freq_Hz,
        )
    if kind == "robstride02":
        from .robstride import RobStrideMotorGroup
        return RobStrideMotorGroup(
            can_id_L=can_id_L, can_id_R=can_id_R, channel=channel,
            torque_limit=torque_limit, control_freq_Hz=control_freq_Hz,
        )
    raise ValueError(f"Unknown motor_type {motor_type!r}; choose from {MOTOR_TYPES}")
