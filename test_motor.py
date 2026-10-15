"""Motor bring-up test: connect, command a safe torque, print encoder readings.

Default is ZERO torque, so you can move the leg by hand and watch pos/vel/torque.
No mocap, GPIO or logging involved. Ctrl+C stops it and disables the motors.

Run:  python test_motor.py
"""
import signal
import sys
import time

import numpy as np

from motors import make_motor_group
from t2_spline import HipTorqueProfile
from utils_teleplot import Teleplot

# =============================================================================
# Configuration — edit before each test
# =============================================================================

# Motor: "ak80-9" (CubeMars) or "robstride02"
motor_type = "ak80-9"
can_id_L = 3
can_id_R = 2
motor_channel = "can0"
torque_limit = 5.0          # Nm, hard clip on whatever is commanded
control_freq_Hz = 100

# What to command:
#   "zero"     -> no torque (just read encoders)
#   "constant" -> test_torque_Nm * scale_factor on both motors
#   "spline"   -> sweep a fake gait cycle through the TBE spline (scale/delay apply)
test_mode = "zero"
test_torque_Nm = 0.0        # used by "constant"
scale_factor = 0.0          # 0-1, scales torque (constant and spline)
delay_factor = 0            # gait-cycle % shift of the spline (spline mode only)
body_mass_kg = 74           # spline mode
fake_gait_period_sec = 1.2  # spline mode: time for one 0-100% sweep

# Run length and printing
duration_sec = 30
print_every_sec = 0.1

# Teleplot (live UDP telemetry)
teleplot_host = "127.0.0.1"
teleplot_port = 47269

# =============================================================================

motors = None
teleplot = Teleplot(teleplot_host, teleplot_port)


def shutdown(*_):
    if motors is not None:
        motors.disconnect()
    print("\nMotors disabled. Exiting.")
    sys.exit(0)


def main():
    global motors
    motors = make_motor_group(
        motor_type=motor_type,
        can_id_L=can_id_L,
        can_id_R=can_id_R,
        channel=motor_channel,
        torque_limit=torque_limit,
        control_freq_Hz=control_freq_Hz,
    )
    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    profile = None
    if test_mode == "spline":
        profile = HipTorqueProfile(
            body_mass_kg=body_mass_kg,
            control_freq_Hz=control_freq_Hz,
            scale_factor=scale_factor,
            delay_percent_gc=delay_factor,
        )
    elif test_mode not in ("zero", "constant"):
        raise ValueError(f"Unknown test_mode {test_mode!r}")

    print(f"motor={motor_type} mode={test_mode} limit={torque_limit} Nm "
          f"scale={scale_factor} delay={delay_factor}")
    motors.connect()

    print(f"{'t(s)':>6} | {'posL(deg)':>10} {'velL':>9} {'trqL':>7} | "
          f"{'posR(deg)':>10} {'velR':>9} {'trqR':>7} | {'cmdL':>6} {'cmdR':>6}")

    t0 = time.time()
    k = 0
    next_print = 0.0
    cmdL = cmdR = 0.0
    try:
        while True:
            t = time.time() - t0
            if t >= duration_sec:
                break

            if test_mode == "constant":
                cmdL = cmdR = test_torque_Nm * scale_factor
            elif test_mode == "spline":
                gc_frac = (t % fake_gait_period_sec) / fake_gait_period_sec  # 0-1
                cmdL, cmdR = profile.torque_from_percent_gc_lr(gc_frac, gc_frac)
            else:
                cmdL = cmdR = 0.0

            motors.set_torque(cmdL, cmdR)
            posL, velL, trqL, posR, velR, trqR = motors.update_readings()
            posL, velL = -posL, -velL  # flexion positive on both sides

            teleplot.sendValue('pos_L', posL)
            teleplot.sendValue('pos_R', posR)
            teleplot.sendValue('vel_L', velL)
            teleplot.sendValue('vel_R', velR)
            teleplot.sendValue('trq_L', trqL)
            teleplot.sendValue('trq_R', trqR)
            teleplot.sendValue('cmd_L', cmdL)
            teleplot.sendValue('cmd_R', cmdR)

            if t >= next_print:
                print(f"{t:6.2f} | {posL:10.2f} {velL:9.2f} {trqL:7.2f} | "
                      f"{posR:10.2f} {velR:9.2f} {trqR:7.2f} | {cmdL:6.2f} {cmdR:6.2f}")
                next_print += print_every_sec

            k += 1
            while (time.time() - t0) < k / control_freq_Hz:
                pass
    finally:
        motors.disconnect()
        teleplot.close()
        print(f"Done: {k} loop iterations in {time.time() - t0:.1f}s "
              f"({k / max(time.time() - t0, 1e-9):.0f} Hz)")


if __name__ == "__main__":
    main()
