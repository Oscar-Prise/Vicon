#FUll protocol script for trapazoid profile
import time, dataclasses, enum, signal, os, atexit, gc, threading
import numpy as np
import pandas as pd
import scipy.signal as sp_signal
from scipy.signal import butter, filtfilt
from Header_Mocap_trigger_protocolTest import Mocap_trigger
from utils_motors import RobStrideMotorGroup
from utils_gpio import GpioPulse, TrialPulseScheduler
from utils_teleplot import Teleplot
import csv

# =============================================================================
# Configuration — edit before each trial
# =============================================================================

# Trial
subject = 'AB01'
trial_start_sec = 1
target_duration_sec = 31
target_time_range = 31
exo_ON = True
scale_factor_percent = 0
delay_factor = 0
duration = 0
body_mass_kg = 80

# Trigger: "mocap" or "typing"
trigger_type = "typing"

# Output paths
OUTPUT_DIR = 'test_run'
mocap_log_csv = 'output.csv'

# GPIO sync pulses (for Vicon / external recording)
gpio_pin = 7
gpio_first_pulse_sec = 2.0
gpio_pulse_duration_sec = 0.05

# RobStride motors
can_id_L = 1
can_id_R = 2
motor_channel = "can0"
torque_limit = 17.0
offset_samples = 50
control_freq_Hz = 100
frame_length = 95
motor_cmd_L = 1.0   # Nm, used when exo_ON
motor_cmd_R = -1.0  # Nm, used when exo_ON

# Teleplot (live UDP telemetry)
teleplot_host = "127.0.0.1"
teleplot_port = 47269

# Vicon / mocap server
mocap_server_ip = "172.24.44.177"
mocap_port = 11

# =============================================================================
# Runtime state (set during execution — do not edit)
# =============================================================================
trial_num = None
trial_name = None

# data to be saved (changed to lists for efficient appending)
data_to_save = {
    "timestamp": [],
    "mtr_cmd_L": [], "mtr_cmd_R": [],    "mtr_pos_L": [], "mtr_pos_R": [],    "mtr_vel_L": [], "mtr_vel_R": [],
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

    save_data(trial_start_sec, target_duration_sec)
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
    pulse_scheduler = TrialPulseScheduler(
        gpio_pulse,
        first_at_sec=gpio_first_pulse_sec,
        second_at_sec=target_time_range,
        pulse_duration_sec=gpio_pulse_duration_sec,
    )

    motors = RobStrideMotorGroup(
        can_id_L=can_id_L,
        can_id_R=can_id_R,
        channel=motor_channel,
        torque_limit=torque_limit,
        offset_samples=offset_samples,
        control_freq_Hz=control_freq_Hz,
        frame_length=frame_length,
    )
    motors.connect()

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
        # typing 모드는 위에서 이미 처리됨            
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
            if mocap_trigger.first_data_received.is_set():
                copR = mocap_trigger.send_copR
                copL = mocap_trigger.send_copL
                # print(copR)
                time_sent = mocap_trigger.send_time
                time_recv = mocap_trigger.recv_time
                Frz = mocap_trigger.send_Frz
                Flz = mocap_trigger.send_Flz

                # time_needed = time_recv - time_sent
                # copRList.append(copR)
                # copLList.append(copL)
                # tsentList.append(time_sent)
                # trecvList.append(time_recv)

                with open(mocap_log_csv, "a", newline="", encoding="utf-8") as f:
                    writer = csv.writer(f)
                    writer.writerow([time_sent, time_recv, copR, copL, Frz, Flz])

            else:
                # Mocap client is running but no data yet - use defaults
                trigger = None
                mocap_data_available = False
    

        # (
        #     current_pos_L, current_vel_L, current_torque_L,
        #     current_pos_R, current_vel_R, current_torque_R,
        # ) = motors.update_readings()

        data_to_save['mtr_pos_L'].append(current_pos_L); data_to_save['mtr_pos_R'].append(-current_pos_R)
        data_to_save['mtr_vel_L'].append(current_vel_L); data_to_save['mtr_vel_R'].append(-current_vel_R)

        motor_cmd_val_L = motor_cmd_L
        motor_cmd_val_R = motor_cmd_R

        if not exo_ON:
            motor_cmd_val_L, motor_cmd_val_R = 0.0, 0.0

        motors.set_torque(motor_cmd_val_L, motor_cmd_val_R)
        (
            current_pos_L, current_vel_L, current_torque_L,
            current_pos_R, current_vel_R, current_torque_R,
        ) = motors.update_readings() 

        data_to_save['mtr_cmd_L'].append(motor_cmd_val_L)
        data_to_save['mtr_cmd_R'].append(motor_cmd_val_R)
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
        teleplot.sendValue('pos_L', current_pos_L)
        teleplot.sendValue('pos_R', current_pos_R)
        teleplot.sendValue('cmd_L', motor_cmd_val_L)
        teleplot.sendValue('cmd_R', motor_cmd_val_R)

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

    # Prompt for trial number in parent process only (prevents child processes
    # created with multiprocessing 'spawn' from re-running the prompt)
    trial_num = int(input("Enter trial number: "))
    trial_name = f'{subject}_{trial_num}_scale_{scale_factor_percent}'

    teleplot = Teleplot(teleplot_host, teleplot_port)

    if trigger_type == "mocap":
        mocap_trigger = Mocap_trigger(server_ip=mocap_server_ip, port_number=mocap_port)
        mocap_trigger.start_client()
        mocap_thread = threading.Thread(target=mocap_trigger.stream_data, daemon=True)
        mocap_thread.start()
        print("Mocap client connected and streaming thread started")

    main()