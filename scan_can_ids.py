"""Find which CAN IDs AK80-9 motors answer to.

Sends a ZERO-torque MIT frame to each ID in the range and lists every reply seen
on the bus. Connect ONE motor at a time, motor power on, leg supported.

Run:  python scan_can_ids.py
"""
import os
import time

import can

from motors.cubemars import _mit_torque_message, REPLY_ID_BASE

# =============================================================================
channel = "can0"
id_range = range(1, 128)    # IDs to probe
wait_per_id_sec = 0.05      # time to listen after each probe
bring_up = True             # set False if can0 is already up
# =============================================================================


def main():
    if bring_up:
        os.system(f"sudo /sbin/ip link set {channel} down")
        os.system(f"sudo /sbin/ip link set {channel} txqueuelen 1000 up type can bitrate 1000000")

    bus = can.Bus(channel=channel, bustype="socketcan", receive_own_messages=False)
    seen = {}   # arbitration_id -> (probed id, count)
    try:
        for can_id in id_range:
            bus.send(_mit_torque_message(can_id, 0.0))
            end = time.time() + wait_per_id_sec
            while time.time() < end:
                msg = bus.recv(timeout=max(end - time.time(), 0.001))
                if msg is None:
                    continue
                prev = seen.get(msg.arbitration_id, (can_id, 0))
                seen[msg.arbitration_id] = (prev[0], prev[1] + 1)
    finally:
        bus.shutdown()

    if not seen:
        print("No frames received at all. Check motor power, CAN H/L wiring, bitrate (1 Mbit/s).")
        return
    print("Frames seen (arbitration id -> first probed id, count):")
    for arb, (probed, n) in sorted(seen.items()):
        tag = ""
        if arb >= REPLY_ID_BASE and arb - REPLY_ID_BASE < 256:
            tag = f"   <- AK80-9 reply, motor ID {arb - REPLY_ID_BASE}"
        print(f"  0x{arb:08X}  probed {probed}  x{n}{tag}")


if __name__ == "__main__":
    main()
