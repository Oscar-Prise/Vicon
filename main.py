#FUll protocol script for trapazoid profile
import time, dataclasses, enum, signal, os, atexit, gc, threading
import numpy as np
import pandas as pd
import scipy.signal as sp_signal
from scipy.signal import butter, filtfilt
from Header_Mocap_trigger_protocolTest import Mocap_trigger
from motors import make_motor_group
from utils_gpio import GpioPulse, SyncPulse
from utils_teleplot import Teleplot
from t2_spline import HipTorqueProfile
from gait_phase import GaitPhaseTracker
import csv

# =============================================================================
# Configuration — edit before each trial
# =============================================================================

# Subject / trial naming
# Output files are named: date_subject_exo_exocondition_terrain_torque_delay
# e.g. 10_26_2026_AB01_V2_ExoOn_LG1p2_15Nm_delay-10
subject = 'AB01'
body_mass_kg = 74.5
exo_version = 'V3'
exo_ON = True            # also sets the name's exo condition (ExoOn / ExoOff)
terrain = 'LG1p2'
torque_label = '15Nm'
scale_factor = 0.15      # 0–1, scales spline peak torque (0 = off, 1 = full profile)

# Motors: "ak80-9" (CubeMars) or "robstride02"
motor_type = "ak80-9"
can_id_L = 3
can_id_R = 2

# Output paths
OUTPUT_DIR = 'data'
record_data = False       # False = run the trial without saving CSVs

# Trial timing
trial_start_sec = 1
target_duration_sec = 31
target_time_range = 31
track_gait_phase = True  # extrapolate gait % between ~30 Hz mocap packets (False = use raw packet value)
delay_factor = 5  # gait-cycle % shift along x-axis (positive = peaks occur later)
duration = 0

# Trigger: "mocap" or "typing"
trigger_type = "mocap"

# GPIO sync pulses (for Vicon / external recording)
gpio_pin = 7
gpio_first_pulse_sec = 2.0
gpio_pulse_duration_sec = 0.05

# Motor bus
motor_channel = "can0"
torque_limit = 17.0  # max commanded torque (Nm)
control_freq_Hz = 100

# Teleplot (live UDP telemetry)
teleplot_host = "127.0.0.1"
teleplot_port = 47269

# Vicon / mocap server
mocap_server_ip = "172.24.44.177"
mocap_port = 11
mocap_stale_timeout_sec = 0.5  # no mocap packet for this long -> command 0 Nm

# =============================================================================
# Runtime state (set during execution — do not edit)
# =============================================================================
trial_name = None
mocap_log_csv = None

# data to be saved (changed to lists for efficient appending)
data_to_save = {
    "timestamp": [],
    "mtr_cmd_L": [], "mtr_cmd_R": [], 
    "mtr_pos_L": [], "mtr_pos_R": [], 
    "mtr_vel_L": [], "mtr_vel_R": [],
    "actual_torque_L": [], "actual_torque_R": [],
    "gpio_output": []  # GPIO 
}

# Global variables
mocap_trigger = None  # Will be initialized in __main__
gpio_pulse = None
motors = None
teleplot = None

# Function to save all collected data
def save_data(start_rec_sec=0, trial_time_sec=None):
    global data_to_save

    # Convert lists to NumPy arrays
    data_np = {k: np.array(v) if isinstance(v, list) else v for k, v in data_to_save.items()}

    # Determine minimum length
    min_len = min(len(data_np["timestamp"]), len(data_np["mtr_pos_L"]), len(data_np["mtr_pos_R"]),
                  len(data_np["mtr_vel_L"]), len(data_np["mtr_vel_R"]),
                  len(data_np["actual_torque_L"]), len(data_np["actual_torque_R"]),
                  len(data_np["gpio_output"]) if "gpio_output" in data_np else float('inf'),
                  len(data_np["mtr_cmd_L"]), len(data_np["mtr_cmd_R"]))

    print(f'Total data length collected: {min_len}')
    
    if min_len == 0:
        print("ERROR: No data collected! min_len is 0")
        return
    
    # Calculate start and end indices for slicing
    start_idx = int(start_rec_sec * control_freq_Hz)
    end_idx = min_len
    
    if trial_time_sec:
        end_idx = min(min_len, int((start_rec_sec + trial_time_sec) * control_freq_Hz))
    
    print(f'Slicing data from {start_rec_sec}s to {(start_rec_sec + (trial_time_sec or (min_len/100 - start_rec_sec)))}s')
    print(f'Index range: {start_idx} to {end_idx}')

    # Slice data with start offset
    timestamp_sliced = [t - start_rec_sec for t in data_np["timestamp"][start_idx:end_idx]]
    sliced_data = {k: v[start_idx:end_idx] if k.startswith('mtr') or k.startswith('actual_torque') or k in {'gpio_output', 'perturbation_idx'} else None for k, v in data_np.items()}
    sliced_data['time'] = timestamp_sliced

    # Create DataFrames and save to CSV
    motor_data_keys = ['time', 'mtr_pos_L', 'mtr_pos_R', 'mtr_vel_L', 'mtr_vel_R']
    if 'gpio_output' in sliced_data and sliced_data['gpio_output'] is not None:
        motor_data_keys.append('gpio_output')
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    df_mtr = pd.DataFrame({k: sliced_data[k] for k in motor_data_keys})
    df_mtr.to_csv(f'{OUTPUT_DIR}/{trial_name}_input_motor.csv', index=False)
    print(f'Motor Data saved to {trial_name}_input_motor.csv')
    print('Dimensions:', df_mtr.shape)

    # Save motor command data
    df_torque = pd.DataFrame({k: sliced_data[k] for k in ['time', 'mtr_cmd_L', 'mtr_cmd_R', 'actual_torque_L', 'actual_torque_R', 'gpio_output']})
    df_torque.to_csv(f'{OUTPUT_DIR}/{trial_name}_output_torque.csv', index=False)
    print(f'Torque data saved to {trial_name}_output_torque.csv')
    print('Dimensions:', df_torque.shape)
    
# Signal handler for graceful exit
def exit_signal_handler(sig, frame):
    print("Signal received, initiating shutdown...")    
    
    motors.disconnect()

    if record_data:
        save_data(trial_start_sec, target_duration_sec)
    else:
        print("record_data is False - data not saved")
    if gpio_pulse is not None:
        gpio_pulse.cleanup()
    if teleplot is not None:
        teleplot.close()
    gc.collect()

    print("Exiting program")
    os._exit(0)

def main():
    # include global variables that need to be reassigned inside the main function
    global data_to_save, motors, gpio_pulse

    gpio_pulse = GpioPulse(pin=gpio_pin)
    gpio_pulse.setup()
    pulse_scheduler = SyncPulse(
        gpio_pulse,
        first_at_sec=gpio_first_pulse_sec,
        second_at_sec=target_time_range,
        pulse_duration_sec=gpio_pulse_duration_sec,
    )

    motors = make_motor_group(
        motor_type=motor_type,
        can_id_L=can_id_L,
        can_id_R=can_id_R,
        channel=motor_channel,
        torque_limit=torque_limit,
        control_freq_Hz=control_freq_Hz,
    )
    motors.connect()

    hip_torque_profile = HipTorqueProfile(
        body_mass_kg=body_mass_kg,
        control_freq_Hz=control_freq_Hz,
        scale_factor=scale_factor,
        delay_percent_gc=delay_factor,
    )

    current_pos_L, current_vel_L = 0.0, 0.0
    current_pos_R, current_vel_R = 0.0, 0.0

    # Setting for the exiting process
    atexit.register(lambda: (motors.disconnect(), gpio_pulse.cleanup()))
    signal.signal(signal.SIGINT, exit_signal_handler)

    # Maria
    logging_started = False
    start_time = None
    start_index = 1
    actuation_started = False
    actuation_start_time = None
    pending_param_id = None
    active_param_id = 1
    copRList = []
    copLList = []
    tsentList = []
    trecvList = []
    percent_gcL, percent_gcR = 0.0, 0.0
    gait_tracker_L, gait_tracker_R = GaitPhaseTracker(), GaitPhaseTracker()
    mocap_data_available = False

    # Wait for the trigger to start the trial (Maria)
    if trigger_type == "mocap":
        print("Waiting for start trigger...\n")
        
    elif trigger_type == "typing":
        input_trigger = input("Press Enter to start trial...\n")
        if input_trigger == "":
            print("Trial started")


    if trigger_type == "mocap" and not logging_started:
        print("Waiting for start trigger from server...")
        mocap_trigger.start_logging_event.wait()
        
        # print("Mocap trigger received - starting data logging")
        start_time = time.time()
        logging_started = True
        print(f"Started Vicon time: {start_time}")

        
    elif trigger_type == "typing" and not logging_started:           
        start_time = time.time()
        logging_started = True

    # Main control loop
    trigger = None
    while True:

       
        if not logging_started:
            continue

        # 1. Read the motor encoder values
        # Check if we have received first mocap data
        if trigger_type == "mocap" and mocap_trigger is not None:
            # Trial is over once the mocap server closes the connection: stop
            # the motors (0 Nm), save the data and exit, same as Ctrl+C.
            if mocap_trigger.first_data_received.is_set() and not mocap_trigger.running:
                print("Mocap connection closed - stopping motors and saving data.")
                exit_signal_handler(None, None)

            # Also treat a stalled stream as "no data": command 0 Nm instead of
            # holding the last gait phase.
            mocap_stale = (mocap_trigger.first_data_received.is_set() and
                           (time.time() - mocap_trigger.recv_time) > mocap_stale_timeout_sec)

            if mocap_trigger.first_data_received.is_set() and not mocap_stale:
                time_sent = mocap_trigger.send_time
                time_recv = mocap_trigger.recv_time
                # copR = mocap_trigger.send_copR
                # copL = mocap_trigger.send_copL
                # Frz = mocap_trigger.send_Frz
                # Flz = mocap_trigger.send_Flz
                raw_gcR = mocap_trigger.send_percent_gcR
                raw_gcL = mocap_trigger.send_percent_gcL
                # Cap to one stride: values > 1 (missed/late heel strike) would
                # otherwise wrap around and command torque from the wrong phase.
                if track_gait_phase:
                    # Advance the last packet's phase to "now" (loop is 100 Hz,
                    # packets arrive ~30 Hz) instead of reusing a stale value.
                    now = time.time()
                    percent_gcR = gait_tracker_R.update(raw_gcR, time_sent, time_recv, now)
                    percent_gcL = gait_tracker_L.update(raw_gcL, time_sent, time_recv, now)
                else:
                    percent_gcR = float(np.clip(raw_gcR, 0.0, 1.0))
                    percent_gcL = float(np.clip(raw_gcL, 0.0, 1.0))
                mocap_data_available = True

                # time_needed = time_recv - time_sent
                # copRList.append(copR)
                # copLList.append(copL)
                # tsentList.append(time_sent)
                # trecvList.append(time_recv)

                # with open(mocap_log_csv, "a", newline="", encoding="utf-8") as f:
                #     writer = csv.writer(f)
                #     # writer.writerow([time_sent, time_recv, copR, copL, Frz, Flz])
                #     writer.writerow([time_sent, time_recv, raw_gcR, raw_gcL, percent_gcR, percent_gcL])  # raw (uncapped), then value used for control
            else:
                # Mocap client is running but no data yet - use defaults
                trigger = None
                mocap_data_available = False

        data_to_save['mtr_pos_L'].append(current_pos_L); data_to_save['mtr_pos_R'].append(-current_pos_R)
        data_to_save['mtr_vel_L'].append(current_vel_L); data_to_save['mtr_vel_R'].append(-current_vel_R)

        if exo_ON and mocap_data_available:
            cmdL, cmdR = hip_torque_profile.torque_from_percent_gc_lr(
                percent_gcL, percent_gcR
            )
        else:
            cmdL, cmdR = 0.0, 0.0
            pcmdL, pcmdR = hip_torque_profile.torque_from_percent_gc_lr(
                percent_gcL, percent_gcR
            )
            if not exo_ON:  # avoid 100 Hz console spam when data is just missing/stale
                print(percent_gcL,percent_gcR,pcmdL,pcmdR)

        motors.set_torque(cmdL, cmdR)
        (
            current_pos_L, current_vel_L, current_torque_L,
            current_pos_R, current_vel_R, current_torque_R,
        ) = motors.update_readings() 

        data_to_save['mtr_cmd_L'].append(cmdL)
        data_to_save['mtr_cmd_R'].append(cmdR)
        data_to_save['actual_torque_L'].append(current_torque_L)
        data_to_save['actual_torque_R'].append(-current_torque_R)

        if trigger is not None:
            actuation_started = True
            actuation_start_time = time.time()
            print(trigger)
            trigger = None

        current_time = time.time() - start_time
        pulse_scheduler.update(current_time)
        data_to_save['gpio_output'].append(gpio_pulse.read_state())
        

        # 10. Stream live motor data to Teleplot
        teleplot.sendValue('pos_L', -current_pos_L)  # flexion positive on both sides
        teleplot.sendValue('pos_R', current_pos_R)
        teleplot.sendValue('gc_L', percent_gcL)  # capped 0-1 (what the controller uses)
        teleplot.sendValue('gc_R', percent_gcR)
        if mocap_data_available:
            teleplot.sendValue('gc_L_raw', raw_gcL)  # uncapped, to spot overshoot
            teleplot.sendValue('gc_R_raw', raw_gcR)
        teleplot.sendValue('cmd_L', cmdL)
        teleplot.sendValue('cmd_R', cmdR)

        # Wait for the time to reach the next clock cycle
        if (time.time() - start_time) > (start_index / motors.control_freq_Hz):
            pass
            # print("Loop time exceeded: ", (time.time() - start_time) - (start_index / motors.control_freq_Hz))
        else:
            while (time.time() - start_time) < (start_index / motors.control_freq_Hz):
                pass
        data_to_save['timestamp'].append(time.time()-start_time)
        start_index += 1

if __name__ == '__main__':

    # Garbage collection
    gc.collect()

    exo_condition = 'ExoOn' if exo_ON else 'ExoOff'
    delay_tag = f'delay{delay_factor:g}'.replace('.', 'p')  # e.g. delay-10, delay2p5
    trial_name = (f'{time.strftime("%m_%d_%Y")}_{subject}_{exo_version}_'
                  f'{exo_condition}_{terrain}_{torque_label}_{delay_tag}')
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    # Never overwrite a previous trial with the same settings: append _2, _3, ...
    base_name, repeat = trial_name, 1
    while os.path.exists(f'{OUTPUT_DIR}/{trial_name}_output_torque.csv'):
        repeat += 1
        trial_name = f'{base_name}_{repeat}'
    # mocap_log_csv = f'{OUTPUT_DIR}/{trial_name}_mocap.csv'
    print(f'Trial name: {trial_name}')

    teleplot = Teleplot(teleplot_host, teleplot_port)

    if trigger_type == "mocap":
        mocap_trigger = Mocap_trigger(server_ip=mocap_server_ip, port_number=mocap_port)
        mocap_trigger.start_client()
        mocap_thread = threading.Thread(target=mocap_trigger.stream_data, daemon=True)
        mocap_thread.start()
        print("Mocap client connected and streaming thread started")

    main()