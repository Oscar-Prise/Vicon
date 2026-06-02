#FUll protocol script for trapazoid profile
import time, dataclasses, enum, signal, os, atexit, socket, can, gc, threading
import numpy as np
import pandas as pd
import scipy.signal as sp_signal
from scipy.signal import butter, filtfilt
from Header_Mocap_trigger_protocolTest import Mocap_trigger
import Jetson.GPIO as GPIO
from actuator_group import ActuatorGroup
from tmotor_v3 import TMotorV3
import csv
actuation = None

PARAMS_CSV_PATH = os.path.join(os.path.dirname(__file__), "gen_final_paramstrap.csv")

# Trial setting
subject = 'AB01'  # Change this for different subjects
trial_start_sec = 1
target_duration_sec =31
target_time_range = 31
exo_ON = False

# Trigger setting
trigger_type = "mocap"  # "mocap" or "typing"

# Body mass setting
body_mass_kg = 80 # kg

scale_factor_percent = 0
delay_factor = 0
duration = 0
trial_num = None
trial_name = None

# scale_factor = scale_factor_percent/100
# delay_factor = int(delay_factor/10)

# data to be saved (changed to lists for efficient appending)
data_to_save = {
    "timestamp": [],
    "mtr_cmd_L": [], "mtr_cmd_R": [],    "mtr_pos_L": [], "mtr_pos_R": [],    "mtr_vel_L": [], "mtr_vel_R": [],
    "actual_torque_L": [], "actual_torque_R": [],
    "gpio_output": []  # GPIO 
}





def get_single_perturbation_row(params_df, perturbation_id):
    matching_rows = params_df[params_df["Perturbation"].astype(int) == int(perturbation_id)]
    if len(matching_rows) != 1:
        raise ValueError(f"Expected exactly one row with Perturbation={int(perturbation_id)}, found {len(matching_rows)}")
    return matching_rows.iloc[0]


def apply_perturbation_params(exo_obj, perturbation_row):
    exo_obj.scale_factor = 0
    exo_obj.duration = 0
    exo_obj.delay_factor = 0
    exo_obj.side = 'r'


# Global variables
GPIO_PIN = 7  # Define pin number globally
mocap_trigger = None  # Will be initialized in __main__

def send_gpio_pulse_start():
    """Start a GPIO pulse by setting pin HIGH"""
    try:
        GPIO.output(GPIO_PIN, GPIO.HIGH)
        print("GPIO pulse started (HIGH)")
    except Exception as e:
        print(f"Error starting GPIO pulse: {e}")

def send_gpio_pulse_end():
    """End a GPIO pulse by setting pin LOW"""
    try:
        GPIO.output(GPIO_PIN, GPIO.LOW)
        print("GPIO pulse ended (LOW)")
    except Exception as e:
        print(f"Error ending GPIO pulse: {e}")

def get_gpio_output_state():
    """Get current GPIO output pin state (0 or 1)"""
    try:
        return int(GPIO.input(GPIO_PIN))
    except:
        # if GPIO not initialized or error occurs
        return 0

def safe_gpio_cleanup():
    """GPIO """
    try:
        GPIO.cleanup()
        print("GPIO cleaned up successfully")
    except Exception as e:
        print(f"Error during GPIO cleanup: {e}")

trigger_flag = 0

class Exo:
    def __init__(self,):

        self.CAN_id_L = 1 # NEED TO FIRST IDENTIFY THIS using display_motor_data.py
        self.CAN_id_R = 2 # NEED TO FIRST IDENTIFY THIS using display_motor_data.py
        self.mtr_type = "AK80-9"
        self.mtr_version = 3 # either 2 or 3
        self.control_freq_Hz = 100
        self.frame_length = 95  # Window size (in frame)
        # self.torque_limit = 10

        # biotorque parameters
        self.scale_factor = 0
        self.delay_factor = 0 # Number of frames to delay the torque command
        self.duration = 0

        # note: motors zero themselves when actuation.Motors() runs
        _ = input("Press Enter to initialize motors: ")
        assert self.mtr_version in (2,3)
        if self.mtr_version == 2:
            raise NotImplementedError("mtr_version=2 is not supported in this trapezoid-only protocol")
        elif self.mtr_version == 3:
            init_list = [
                TMotorV3(mtr_id, self.mtr_type) for mtr_id in [self.CAN_id_L, self.CAN_id_R]
            ]
            self.mtr_comms = ActuatorGroup(init_list)

        # Specify the CAN interface and channel
        try:
            self.bus = can.Bus(interface='socketcan', channel='can0')  # Replace 'socketcan' and 'can0' with your actual interface and channel
            # print("CAN bus initialized successfully.")
        except Exception as e:
            print(f"Error initializing CAN bus: {e}")
        self.notifier = can.Notifier(self.bus, [])

    def update_readings(self, CAN_id):
        mtr_pos = self.mtr_comms.get_position(CAN_id, degrees=True)
        mtr_vel = self.mtr_comms.get_velocity(CAN_id, degrees=True)
        mtr_torque = self.mtr_comms.get_torque(CAN_id)

        return mtr_pos, mtr_vel, mtr_torque



def trapezoid_profile(time_progress_percent):
    """Return normalized trapezoid amplitude for progress in [0, 100]."""
    if not (0.0 <= time_progress_percent <= 100.0):
        return 0.0

    normalized_x = time_progress_percent / 100.0
    ramp_up_end = 0.20
    ramp_down_start = 0.80

    if normalized_x <= ramp_up_end:
        return normalized_x / ramp_up_end
    if normalized_x <= ramp_down_start:
        return 1.0
    return 1.0 - (normalized_x - ramp_down_start) / (1.0 - ramp_down_start)

# Telemetry function for real-time data visualization
def sendTelemetry(name, value):
    now = time.time() * 1000
    msg = name+":"+str(now)+":"+str(value)+"|g"
    sock.sendto(msg.encode(), teleplotAddr)

def sendBatchTelemetry(data_dict):
    now = time.time() * 1000
    try:
        for name, value in data_dict.items():
            msg = name + ":" + str(now) + ":" + str(value) + "|g"
            sock.sendto(msg.encode(), teleplotAddr)

        return True  # Successfully sent
    except Exception as e:
        print(f"Error in sendBatchTelemetry: {e}")
        return False

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
    start_idx = int(start_rec_sec * 100)  # 100 Hz data collection rate
    end_idx = min_len
    
    if trial_time_sec:
        end_idx = min(min_len, int((start_rec_sec + trial_time_sec) * 100))
    
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
    
    df_mtr = pd.DataFrame({k: sliced_data[k] for k in motor_data_keys})
    df_mtr.to_csv(f'AB07_Validation/{trial_name}_input_motor.csv', index=False)
    print(f'Motor Data saved to {trial_name}_input_motor.csv')
    print('Dimensions:', df_mtr.shape)

    # Save motor command data
    df_torque = pd.DataFrame({k: sliced_data[k] for k in ['time', 'mtr_cmd_L', 'mtr_cmd_R', 'actual_torque_L', 'actual_torque_R', 'gpio_output']})
    df_torque.to_csv(f'AB07_Validation/{trial_name}_output_torque.csv', index=False)
    print(f'Torque data saved to {trial_name}_output_torque.csv')
    print('Dimensions:', df_torque.shape)
    
# Signal handler for graceful exit
def exit_signal_handler(sig, frame):
    print("Signal received, initiating shutdown...")    
    
    # Apply zero torque to the motors
    Exo.mtr_comms.set_torque(Exo.CAN_id_L, 0)
    Exo.mtr_comms.set_torque(Exo.CAN_id_R, 0)

    save_data(trial_start_sec, target_duration_sec)
    cleanup_can(Exo.bus, Exo.notifier)
    safe_gpio_cleanup()  # 안전한 GPIO 정리 함수 사용
    gc.collect()

    print("Exiting program")
    os._exit(0)

# Function to cleanup CAN resources
def cleanup_can(bus, notifier):
    try:
        notifier.stop()
        bus.shutdown()
        print("CAN resources cleaned up successfully")
    except Exception as e:
        print(f"Error during CAN cleanup: {e}")

def main():
    # include global variables that need to be reassigned inside the main function
    global data_to_save, Exo
    # duration = 0.5

    # Initialize GPIO in main process only (not in spawned inference worker)
    GPIO.setmode(GPIO.BOARD)
    GPIO.setup(GPIO_PIN, GPIO.OUT, initial=GPIO.LOW)
    print("GPIO initialized successfully")


    # Initialize the exoskeleton
    Exo = Exo()

    current_pos_L, current_vel_L = 0.0, 0.0
    current_pos_R, current_vel_R = 0.0, 0.0

    # Setting for the exiting process
    atexit.register(lambda: (cleanup_can(Exo.bus, Exo.notifier), safe_gpio_cleanup()))
    signal.signal(signal.SIGINT, exit_signal_handler)

    # Maria
    logging_started = False
    first_pulse_sent = False
    first_pulse_end_time = None
    second_pulse_sent = False
    second_pulse_end_time = None
    start_time = None
    start_index = 1
    actuation_started = False
    actuation_start_time = None
    pending_param_id = None
    active_param_id = 1
    copRList = []
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
    while True:

       
        if not logging_started:
            continue

        # 1. Read the motor encoder values
        # Check if we have received first mocap data
        if mocap_trigger.first_data_received.is_set():
            copR = mocap_trigger.send_copR
            time_sent = mocap_trigger.send_time
            time_recv = mocap_trigger.recv_time
            # print(copR)
            copRList.append(copR)
            tsentList.append(time_sent)
            trecvList.append(time_recv)

            with open("output.csv", "a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow([copR, time_sent, time_recv])

            
        else:
            # Mocap client is running but no data yet - use defaults3
            trigger = None
            mocap_data_available = False
    

        current_pos_L, current_vel_L, current_torque_L = Exo.update_readings(Exo.CAN_id_L)
        current_pos_R, current_vel_R, current_torque_R = Exo.update_readings(Exo.CAN_id_R)

        data_to_save['mtr_pos_L'].append(current_pos_L); data_to_save['mtr_pos_R'].append(-current_pos_R)
        data_to_save['mtr_vel_L'].append(current_vel_L); data_to_save['mtr_vel_R'].append(-current_vel_R)

        # 5. Trapezoid profile generation
        profile_norm = 0.0
        if actuation_started and actuation_start_time is not None and Exo.duration > 0:
            elapsed = time.time() - actuation_start_time
            progress_percent = (elapsed / Exo.duration) * 100.0
            profile_norm = trapezoid_profile(progress_percent)

        motor_cmd_val_L = 0
        motor_cmd_val_R = 0

        if exo_ON == False: motor_cmd_val_L, motor_cmd_val_R = 0.0, 0.0 # use this for Exo off condition

        if trigger is not None:
            actuation_started = True
            actuation_start_time = time.time()
            print(trigger)
            trigger = None

        
       

        # GPIO 
        current_time = time.time() - start_time
        
       
        if current_time >= (2) and not first_pulse_sent:
            send_gpio_pulse_start()
            first_pulse_sent = True
            first_pulse_end_time = current_time + 0.05  
            print("First pulse started 2 seconds after mocap trigger")
        
        
        if first_pulse_sent and first_pulse_end_time and current_time >= first_pulse_end_time:
            send_gpio_pulse_end()
            first_pulse_end_time = None
            print("First pulse ended")
        
   
        if current_time >= (target_time_range) and not second_pulse_sent:
            send_gpio_pulse_start()
            second_pulse_sent = True
            second_pulse_end_time = current_time + 0.05  # 200ms 펄스 지속시간
            print(f'Second pulse started after {current_time} seconds')


        if second_pulse_sent and second_pulse_end_time and current_time >= second_pulse_end_time:
            send_gpio_pulse_end()
            second_pulse_end_time = None
            print("Second pulse ended")
        # if current_time >= 31: ##small buffer, change as needed
        #     break
        # GPIO
        data_to_save['gpio_output'].append(get_gpio_output_state())
        

        # 9. Loop time
        time_0 = time.time()
        # loop_time = time_0 - time_1
        
        # 10. Send telemetry data
        telemetry_data = {
            "gpio_output": get_gpio_output_state(),
        }
        sendBatchTelemetry(telemetry_data)

        # 11. Wait for the time to reach the next clock cycle
        if (time.time() - start_time) > (start_index / Exo.control_freq_Hz):
            pass
            # print("Loop time exceeded: ", (time.time() - start_time) - (start_index / Exo.control_freq_Hz))
        else:
            while (time.time() - start_time) < (start_index / Exo.control_freq_Hz):
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

    # Teleplot setting
    os.system('echo nc -u -w0 127.0.0.1 47269')
    teleplotAddr = ("127.0.0.1", 47269)
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    # Initialize mocap in main process only
    if trigger_type == "mocap":
        mocap_trigger = Mocap_trigger(server_ip="172.24.44.177", port_number=11)
        mocap_trigger.start_client()
        mocap_thread = threading.Thread(target=mocap_trigger.stream_data, daemon=True)
        mocap_thread.start()
        print("Mocap client connected and streaming thread started")

    main()