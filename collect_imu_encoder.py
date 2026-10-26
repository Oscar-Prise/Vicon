"""Passive data collection: Muse thigh IMUs + CubeMars hip encoders. NO controller.

The motors are enabled but only ever commanded 0 Nm, so the device is
transparent while you walk. Press Ctrl+C when you are done; everything
recorded is then written to a CSV in OUTPUT_DIR.

Run:
    python collect_imu_encoder.py                 # name defaults to a timestamp
    python collect_imu_encoder.py --name AB01_walk1
    python collect_imu_encoder.py --imu           # also collect the Muse IMUs (off by default)
    python collect_imu_encoder.py --imu --no-motors   # IMUs only

Startup order (same as AMP-Hip's HipExo):
    1. connect both Muse IMUs over BLE
    2. you stand in the NEUTRAL pose and press Enter
    3. IMU neutral calibration (stand still ~2 s), then motor encoders are zeroed
    4. loop runs until Ctrl+C

Units in the CSV: encoder pos deg, vel deg/s, "cur" = motor current (A, as
reported by the AK80-9); IMU accel m/s^2, gyro deg/s, thigh_angle deg,
quaternion (w,x,y,z), imu_age_* s. Values are raw (no left/right sign flips).
"""
import argparse
import math
import os
import signal
import sys
import time

import numpy as np
import pandas as pd
import yaml

# =============================================================================
# Configuration
# =============================================================================
OUTPUT_DIR = "test_run"
FS_HZ = 100

# Motors (CubeMars AK80-9)
MOTOR_TYPE = "ak80-9"
CAN_ID_L = 3
CAN_ID_R = 2
MOTOR_CHANNEL = "can0"

# Muse IMU driver + rig settings (muse_devices, axes, signs) come from AMP-Hip.
AMP_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "AMP-Hip-main")
MUSE_CONFIG = os.path.join(AMP_DIR, "imu_reference", "cfg", "impedance_rl_hip.yaml")
MUSE_UTILS = os.path.join(AMP_DIR, "imu_reference", "utils")
# =============================================================================

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)


def build_columns(use_imu, use_motors):
    cols = ["time", "host_unix_time"]
    if use_motors:
        cols += ["enc_pos_L", "enc_pos_R", "enc_vel_L", "enc_vel_R", "enc_cur_L", "enc_cur_R"]
    if use_imu:
        for s in ("L", "R"):
            cols += [f"imu_{s}_{k}" for k in ("ax", "ay", "az", "gx", "gy", "gz")]
            cols += [f"imu_{s}_q{k}" for k in "wxyz"]
            cols += [f"imu_{s}_thigh_angle_deg", f"imu_{s}_device_ms", f"imu_{s}_age_s"]
        cols += ["imu_valid"]
    return cols


def main():
    parser = argparse.ArgumentParser(description="Collect IMU + encoder data with no controller.")
    parser.add_argument("--name", default=time.strftime("walk_%Y%m%d_%H%M%S"),
                        help="output file name (without .csv)")
    parser.add_argument("--hz", type=float, default=FS_HZ, help="logging rate (default %(default)s)")
    parser.add_argument("--config", default=MUSE_CONFIG, help="YAML with muse_devices")
    # IMU collection is switched off by default for now; pass --imu to re-enable.
    parser.add_argument("--imu", action="store_true", help="also collect the Muse IMUs (off by default)")
    parser.add_argument("--no-motors", action="store_true", help="do not use the motors")
    args = parser.parse_args()
    use_imu, use_motors = args.imu, not args.no_motors
    if not (use_imu or use_motors):
        parser.error("nothing to collect")

    imus = None
    motors = None
    rows = []   # list of tuples, one per control step

    # Ctrl+C / kill -> KeyboardInterrupt, so the finally block always saves.
    def _raise(sig, frame):
        raise KeyboardInterrupt
    signal.signal(signal.SIGINT, _raise)
    signal.signal(signal.SIGTERM, _raise)

    columns = build_columns(use_imu, use_motors)
    t0 = None
    try:
        # ---- 1. IMUs ----
        if use_imu:
            sys.path.insert(0, MUSE_UTILS)
            from Header_MuseIMU import MuseIMUs
            with open(args.config, "r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f)
            imus = MuseIMUs(cfg)
            imus.start()   # blocks until both streams are up; raises on failure

        # ---- 2/3. neutral pose: IMU calibration, then zero encoders ----
        input("Stand still in the NEUTRAL pose, then press Enter "
              "(calibrates IMUs / zeroes encoders): ")
        if use_imu:
            imus.calibrate_neutral()
        if use_motors:
            from motors import make_motor_group
            motors = make_motor_group(
                motor_type=MOTOR_TYPE, can_id_L=CAN_ID_L, can_id_R=CAN_ID_R,
                channel=MOTOR_CHANNEL, torque_limit=1.0, control_freq_Hz=args.hz,
            )
            motors.connect(prompt=False)   # zeroes the encoders at this pose

        # ---- 4. record ----
        period = 1.0 / args.hz
        print(f"Recording at {args.hz:g} Hz, motors at 0 Nm. Press Ctrl+C to stop and save.")
        t0 = time.perf_counter()
        k = 0
        last_print = t0
        while True:
            k += 1
            target = t0 + k * period
            # sleep most of the wait, spin the last ~1 ms for accuracy
            while True:
                remaining = target - time.perf_counter()
                if remaining <= 0:
                    break
                if remaining > 0.002:
                    time.sleep(remaining - 0.001)

            row = [time.perf_counter() - t0, time.time()]

            if use_motors:
                motors.set_torque(0.0, 0.0)   # keeps motors enabled/replying, no assist
                pL, vL, cL, pR, vR, cR = motors.update_readings()
                row += [pL, pR, vL, vR, cL, cR]

            if use_imu:
                samples, ages, valid = imus.snapshot()
                for side in ("left", "right"):
                    s = samples[side]
                    if s is None:
                        row += [math.nan] * 6 + [math.nan] * 4 + [math.nan, math.nan, math.nan]
                    else:
                        row += list(s.imu) + list(s.quaternion)
                        row += [math.degrees(s.thigh_angle), s.device_time_ms, ages[side]]
                row += [float(valid)]

            rows.append(row)

            now = time.perf_counter()
            if now - last_print >= 1.0:
                last_print = now
                msg = f"{now - t0:7.1f} s  samples {len(rows)}"
                if use_motors:
                    msg += f"  enc L {pL:+7.1f} R {pR:+7.1f} deg"
                if use_imu:
                    msg += f"  imu {'OK' if valid else 'STALE'}"
                print(msg)
    except KeyboardInterrupt:
        print("\nStopping...")
    except Exception as e:
        print(f"\n[ERROR] {e!r}; saving what was recorded so far.")
    finally:
        # Motors first, then IMUs, then save.
        if motors is not None:
            try:
                motors.disconnect()
            except Exception as e:
                print(f"[motors] disconnect error: {e!r}")
        if imus is not None:
            try:
                imus.close()
            except Exception as e:
                print(f"[imu] close error: {e!r}")

        if rows:
            os.makedirs(OUTPUT_DIR, exist_ok=True)
            path = os.path.join(OUTPUT_DIR, f"{args.name}.csv")
            df = pd.DataFrame(np.array(rows, dtype=float), columns=columns)
            df.to_csv(path, index=False)
            dur = df["time"].iloc[-1]
            print(f"Saved {len(df)} samples ({dur:.1f} s, ~{len(df) / max(dur, 1e-9):.1f} Hz) to {path}")
        else:
            print("No samples recorded; nothing saved.")


if __name__ == "__main__":
    main()
