import time, os, signal, gc, torch
import multiprocessing as mp
import numpy as np
import pandas as pd

from RL_Adapter import RL_Adapter
from Utils_Mocap_Datastream import Mocap_trigger
from Utils_GPIO import GPIO_control
from Utils_Teleplot import Teleplot
from Utils import lowpass_filter, fast_roll, cleanup_can, save_data, cartesian_to_percentage, NumpyCompatUnpickler, causal_filter
from scipy.signal import find_peaks
from scipy.interpolate import CubicHermiteSpline


class Spline_TBE:
    def __init__(self, Exo, trial_name, trigger_type, adaptation_ON, pulse_after_start, trial_dur_sec, adjustment_duration, body_mass_kg, peak_time_):
        self.Exo = Exo
        self.trigger_type = trigger_type
        self.trial_name = trial_name
        self.adaptation_ON = adaptation_ON
        self.body_mass_kg = body_mass_kg
        self.pulse_after_start = pulse_after_start
        self.trial_dur_sec = trial_dur_sec
        self.adjustment_duration = adjustment_duration

        # Initialize Teleplot for telemetry data
        self.teleplot = Teleplot()
        self.GPIO_control = GPIO_control()

        self.RL_adapter = RL_Adapter(self.adaptation_ON)

        # --- Spline Initialization ---
        # self.spline = self.spline_generator(
        #     peak_time = 26.5, rise_time = 18.8, mid_time = 47.1, mid_dur = 1.3, peak_time_2 = 82.2, fall_time = 21.4,
        #     gp_offset = 16, peak_extension_torque = -0.370, peak_flexion_torque = 0.221,
        #     ) # LG, from Supplementary Item of https://doi.org/10.1109/TNSRE.2022.3196665 

        self.spline = self.spline_generator(
            peak_time = 34, rise_time = 18.8, mid_time = 47.1, mid_dur = 1.3, peak_time_2 = 82.2, fall_time = 21.4,
            gp_offset = 16, peak_extension_torque = -0.370, peak_flexion_torque = 0.221,
            ) # LG, from Supplementary Item of https://doi.org/10.1109/TNSRE.2022.3196665        

        if self.trigger_type == 'overground':
            gc_offset = 50
            self.spline = np.r_[self.spline[-(100-gc_offset):], self.spline[:gc_offset]]

    def spline_generator(self, peak_time, rise_time, mid_time, mid_dur, peak_time_2, fall_time,
                     gp_offset, peak_extension_torque, peak_flexion_torque,
                     scale_factor = .5, control_Hz = 100):

        spline_x = np.array([peak_time-rise_time-gp_offset, peak_time-gp_offset, mid_time-mid_dur/2-gp_offset, mid_time+mid_dur/2-gp_offset, peak_time_2-gp_offset, peak_time_2+fall_time-gp_offset, 100+peak_time-rise_time-gp_offset])
        spline_y = np.array([0.0, peak_extension_torque, 0.0, 0.0, peak_flexion_torque, 0.0, 0.0]) * scale_factor
        spline_dydx = np.array([0, 0.0, 0.0, 0, 0.0, 0.0, 0])
        spline_profile = CubicHermiteSpline(spline_x,
                                                spline_y,
                                                spline_dydx, extrapolate='periodic')
        
        spline_profile_arr = spline_profile(np.linspace(0, 99, control_Hz))

        return spline_profile_arr

    def _is_valid_spline_params(self, params):
        spline_x = np.array([
            params['peak_time'] - params['rise_time'] - params['gp_offset'],
            params['peak_time'] - params['gp_offset'],
            params['mid_time'] - params['mid_dur'] / 2 - params['gp_offset'],
            params['mid_time'] + params['mid_dur'] / 2 - params['gp_offset'],
            params['peak_time_2'] - params['gp_offset'],
            params['peak_time_2'] + params['fall_time'] - params['gp_offset'],
            100 + params['peak_time'] - params['rise_time'] - params['gp_offset'],
        ])
        return bool(np.all(np.diff(spline_x) > 0))

    def detect_hs_GRF(self, GRF_data, threshold, min_interval=50):
        # Create binary GRF signal based on threshold
        bi_GRF = np.where(GRF_data < threshold, 0, 1)
        diff_GRF = np.diff(bi_GRF)
        # Find indices where the signal goes from 0 to 1
        hs_idx = np.where(diff_GRF == 1)[0] + 1
        if hs_idx.size == 0:
            return np.array([], dtype=int)
        valid_hs = hs_idx[np.insert(np.diff(hs_idx) >= min_interval, 0, True)]
        return valid_hs
    
    def detect_peak(self, mtr_data, prominence, min_interval=50):
        peak_idx, _ = find_peaks(mtr_data, prominence=prominence)
        if peak_idx.size == 0:
            return np.array([], dtype=int)
        valid_peak = peak_idx[np.insert(np.diff(peak_idx) >= min_interval, 0, True)]
        return valid_peak
    
    def get_hs_idx(self, logging_data, loop_idx, trigger_type, search_window = 600):
        search_start_idx = max(0, loop_idx - search_window)
        recent_data = logging_data[search_start_idx:loop_idx]

        if trigger_type == 'mocap':
            hs_idx = self.detect_hs_GRF(recent_data, 0.1, min_interval=50)
        elif trigger_type == 'overground':
            hs_idx = self.detect_peak(recent_data, 20, min_interval=50)
        hs_idx += (search_start_idx)  # Convert to absolute indices
        return hs_idx

    def get_gait_cycle(self, hs_idx_L, hs_idx_R, loop_idx, trigger_type):
        # TBE - Get stride duration
        stride_dur_L = hs_idx_L[-1] - hs_idx_L[-2]
        stride_dur_R = hs_idx_R[-1] - hs_idx_R[-2]
        # TBE - Get gait cycle
        if trigger_type == 'mocap':
            gc_L = (loop_idx - hs_idx_L[-1]) / stride_dur_L
            gc_R = (loop_idx - hs_idx_R[-1]) / stride_dur_R
            gc_L = int(np.clip(gc_L * self.Exo.control_freq_Hz, 0, 99)) # Convert to integer
            gc_R = int(np.clip(gc_R * self.Exo.control_freq_Hz, 0, 99))
        elif trigger_type == 'overground':
            gc_offset = 50
            gc_L = (gc_L + gc_offset) % 100
            gc_R = (gc_R + gc_offset) % 100
        return gc_L, gc_R

    def trigger_update(self, side, last_used_hs_idx, hs_idx, actions, update_freq_gc=2):
        if len(hs_idx) < (update_freq_gc + 1):
            return actions

        if (last_used_hs_idx not in hs_idx[-(update_freq_gc + 1):]): # Only trigger if the last used peak is not in the recent heel strikes
            # Get the absolute start and end indices for the data slice
            start_idx_abs = hs_idx[-(update_freq_gc + 1)]; end_idx_abs = hs_idx[-1]
            print(f'\n{side}', start_idx_abs, hs_idx[-(update_freq_gc):-1], end_idx_abs)
            mid_peak_idx_rel = hs_idx[-(update_freq_gc):-1] - start_idx_abs # This is relative about start_idx_abs
            
            # Update the last used peak to the end of the current window
            if side == 'L':
                self.last_used_hs_idx_L = hs_idx[-1]
                states = np.vstack((
                    self.data_log['pos_L'][start_idx_abs:end_idx_abs].reshape(1, -1),
                    self.data_log['vel_L'][start_idx_abs:end_idx_abs].reshape(1, -1),
                    self.data_log['imu_L'][start_idx_abs:end_idx_abs].T,  # (6, N)
                    self.data_log['GRF_L'][start_idx_abs:end_idx_abs].reshape(1, -1)
                )) # Shape: (1+1+6+1, N)
            elif side == 'R':
                self.last_used_hs_idx_R = hs_idx[-1]
                states = np.vstack((
                    self.data_log['pos_R'][start_idx_abs:end_idx_abs].reshape(1, -1),
                    self.data_log['vel_R'][start_idx_abs:end_idx_abs].reshape(1, -1),
                    self.data_log['imu_R'][start_idx_abs:end_idx_abs].T,  # (6, N)
                    self.data_log['GRF_R'][start_idx_abs:end_idx_abs].reshape(1, -1)
                )) # Shape: (1+1+6+1, N)

            new_actions = self.RL_adapter.trigger_update(side, states, actions)
            if new_actions is None:
                return actions

            if not self._is_valid_spline_params(new_actions):
                print(f"[{side}] Invalid spline timing order from adapter, keeping previous spline params.")
                return actions

            try:
                self.spline = self.spline_generator(
                    peak_time = new_actions['peak_time'], rise_time = new_actions['rise_time'], mid_time = new_actions['mid_time'], mid_dur = new_actions['mid_dur'], peak_time_2 = new_actions['peak_time_2'], fall_time = new_actions['fall_time'],
                    gp_offset = new_actions['gp_offset'], peak_extension_torque = new_actions['peak_extension_torque'], peak_flexion_torque = new_actions['peak_flexion_torque'],
                )
            except ValueError as e:
                print(f"[{side}] Spline update rejected: {e}")
                return actions
            
            print(new_actions)

            return new_actions

        return actions
            
    def run_loop(self, Exo_ON=False):

        # Setting for the exiting process
        signal.signal(signal.SIGINT, self.exit_signal_handler)

        # Initialize State variables
        pos_L, pos_R = 0.0, 0.0
        vel_L, vel_R = 0.0, 0.0
        imu_L, imu_R, imu_P = np.zeros(6), np.zeros(6), np.zeros(6)
        GRF_L, GRF_R = 0.0, 0.0
        spline_params_L = {'peak_time': 34, 'rise_time': 18.8, 'mid_time': 47.1, 'mid_dur': 1.3, 'peak_time_2': 82.2, 'fall_time': 21.4, 'gp_offset': 16, 'peak_extension_torque': -0.370, 'peak_flexion_torque': 0.221}
        spline_params_R = {'peak_time': 34, 'rise_time': 18.8, 'mid_time': 47.1, 'mid_dur': 1.3, 'peak_time_2': 82.2, 'fall_time': 21.4, 'gp_offset': 16, 'peak_extension_torque': -0.370, 'peak_flexion_torque': 0.221}

        # Initialize update latency trackers
        update_latency_R, update_latency_L = 0.0, 0.0
        update_dur_R, update_dur_L = 0.0, 0.0
        avg_loss = 0
        update_start_idx_L, update_start_idx_R = -1, -1
        self.last_used_hs_idx_L, self.last_used_hs_idx_R = -1, -1

        current_incline = 0  # Default task settings
        prev_incline = current_incline
        prev_speed = None # Initialized to fix UnboundLocalError

        # Initialize data structures to save data
        max_samples = int((self.trial_dur_sec + self.pulse_after_start + 5) * 100) # Extra 5 sec for slowing down
        self.data_log = {
            'timestamp': np.zeros(max_samples),
            'pos_L': np.zeros(max_samples), 'pos_R': np.zeros(max_samples),
            'vel_L': np.zeros(max_samples), 'vel_R': np.zeros(max_samples),
            'imu_L': np.zeros((max_samples, 6)), 'imu_R': np.zeros((max_samples, 6)), 'imu_P': np.zeros((max_samples, 6)),
            'GRF_L': np.zeros(max_samples), 'GRF_R': np.zeros(max_samples),
            'CoP_L': np.zeros(max_samples), 'CoP_R': np.zeros(max_samples),
            'cmd_L': np.zeros(max_samples), 'cmd_R': np.zeros(max_samples),
            'incline': ['']*max_samples, 'speed': ['']*max_samples,
            'gpio_output': np.zeros(max_samples)  # GPIO output state
        }

        self.data_log_adapter = {
            'update_start_idx_L': [], 'update_start_idx_R': [],
            'update_latency_L': [], 'update_latency_R': [],
            'update_dur_L': [], 'update_dur_R': [],
        }

        # Start recording time
        first_pulse_sent = False
        first_pulse_end_time = None
        second_pulse_sent = False
        second_pulse_end_time = None
        loop_idx = 0

        # Wait for the trigger to start the trial
        if self.trigger_type == "mocap":
            print("Wait for the tensorrt to warm up...\n")
            self.mocap_trigger = Mocap_trigger(server_ip="172.24.44.177", port_number=11)
            self.mocap_trigger.start_client()
            self.mocap_trigger.stream_start()
            self.mocap_trigger.wait_for_start_logging()
            print("Mocap trigger received - starting data logging")
            
        elif self.trigger_type == "overground":
            input_trigger = input("Press ENTER to start...\n")
            if input_trigger == "":
                print("Trial started")

        start_time = time.time()

        # Main control loop
        while True:

            # 1. Read the motor encoder values
            pos_L, vel_L = self.Exo.update_readings(self.Exo.CAN_id_L)
            pos_R, vel_R = self.Exo.update_readings(self.Exo.CAN_id_R)
            self.data_log['pos_L'][loop_idx] = pos_L; self.data_log['pos_R'][loop_idx] = pos_R
            self.data_log['vel_L'][loop_idx] = vel_L; self.data_log['vel_R'][loop_idx] = vel_R

            # 2. Read the IMU values
            imu_dict = self.Exo.imus.read_IMUs()
            imu_L, imu_R, imu_P = imu_dict["IMU_THIGH_LEFT"], imu_dict["IMU_THIGH_RIGHT"], imu_dict["IMU_PELVIS"]
            self.data_log['imu_L'][loop_idx, :], self.data_log['imu_R'][loop_idx, :], self.data_log['imu_P'][loop_idx, :] = imu_L, imu_R, imu_P

            # 2.1. Mirror the left data to the right side (Unilateral model input)
            imu_L_reflected, imu_R_reflected = imu_L.copy(), imu_R.copy()
            imu_L_reflected[1] *= -1; imu_L_reflected[3] *= -1; imu_L_reflected[5] *= -1
            imu_R_reflected[1] *= -1; imu_R_reflected[3] *= -1; imu_R_reflected[5] *= -1

            # 3. Read the GRF values & detect heel strikes
            if self.trigger_type == "mocap":
                GRF_L, GRF_R, CoP_L, CoP_R, current_speed = self.mocap_trigger.get_GRF()
                # Use vGRF to get heel strike
                hs_idx_L = self.get_hs_idx(self.data_log['GRF_L'], loop_idx, self.trigger_type)
                hs_idx_R = self.get_hs_idx(self.data_log['GRF_R'], loop_idx, self.trigger_type)

            elif self.trigger_type == "overground":
                GRF_L, GRF_R, current_speed = 0.0, 0.0, 0.0
                # Use extension peak to get toe-off
                hs_idx_L = self.get_hs_idx(-self.data_log['pos_L'], loop_idx, self.trigger_type)
                hs_idx_R = self.get_hs_idx(-self.data_log['pos_R'], loop_idx, self.trigger_type)

            # 3.1 Log the GRF, CoP, incline, and speed data
            if prev_speed is None: prev_speed = current_speed
            self.data_log['GRF_L'][loop_idx] = GRF_L; self.data_log['GRF_R'][loop_idx] = GRF_R
            self.data_log['CoP_L'][loop_idx] = CoP_L; self.data_log['CoP_R'][loop_idx] = CoP_R
            self.data_log['incline'][loop_idx] = current_incline; self.data_log['speed'][loop_idx] = current_speed

            # 4 Get sride duration and gait cycle 
            if (len(hs_idx_L) >= 3) and (len(hs_idx_R) >= 3):
                gc_L, gc_R = self.get_gait_cycle(hs_idx_L, hs_idx_R, loop_idx, self.trigger_type)
                
                # TBE - Get torque command
                cmd_L = self.spline[gc_L] * self.body_mass_kg
                cmd_R = self.spline[gc_R] * self.body_mass_kg

                if first_pulse_sent:
                    spline_params_L = self.trigger_update('L', self.last_used_hs_idx_L, hs_idx_L, actions=spline_params_L, update_freq_gc=2)
                    spline_params_R = self.trigger_update('R', self.last_used_hs_idx_R, hs_idx_R, actions=spline_params_R, update_freq_gc=2)

            else:
                hs_idx_L, hs_idx_R = [0], [0]
                gc_L, gc_R = 0, 0
                cmd_L, cmd_R = 0, 0

            # Calculate gradual torque scaling factor
            gradual_torque_scale = min(1.0, ((loop_idx / self.Exo.control_freq_Hz)) / self.adjustment_duration)
            cmd_L = cmd_L * gradual_torque_scale
            cmd_R = cmd_R * gradual_torque_scale
            
            # Send commands to motors
            self.Exo.command_torque(self.Exo.CAN_id_L, cmd_L, Exo_ON) 
            self.Exo.command_torque(self.Exo.CAN_id_R, cmd_R, Exo_ON)
            self.data_log['cmd_L'][loop_idx], self.data_log['cmd_R'][loop_idx] = cmd_L, cmd_R


            # GPIO pulse logic starting from here
            current_time = time.time() - start_time
            
            # First pulse
            if current_time >= (self.pulse_after_start) and not first_pulse_sent:
                self.GPIO_control.send_gpio_pulse_start()
                first_pulse_sent = True
                first_pulse_end_time = current_time + 0.2  # 200ms pulse duration
            # First pulse end
            if first_pulse_sent and first_pulse_end_time and current_time >= first_pulse_end_time:
                self.GPIO_control.send_gpio_pulse_end()
                first_pulse_end_time = None            
            # Second pulse
            if current_time >= (self.pulse_after_start + self.trial_dur_sec) and not second_pulse_sent:
                self.GPIO_control.send_gpio_pulse_start()
                second_pulse_sent = True
                second_pulse_end_time = current_time + 0.2  # 200ms pulse duration
            # Second pulse end
            if second_pulse_sent and second_pulse_end_time and current_time >= second_pulse_end_time:
                self.GPIO_control.send_gpio_pulse_end()
                second_pulse_end_time = None

                print("Trial finished!")
                self.Exo.command_torque(self.Exo.CAN_id_L, 0)
                self.Exo.command_torque(self.Exo.CAN_id_R, 0)
                save_data(self.data_log, self.trial_name, self.pulse_after_start, self.trial_dur_sec)
                cleanup_can(self.Exo.bus, self.Exo.notifier)
                self.GPIO_control.safe_gpio_cleanup()

                break # Exit the loop after the second pulse ends

            # GPIO output logging
            self.data_log['gpio_output'][loop_idx] = self.GPIO_control.get_gpio_output_state()

            # 9. Loop time
            loop_time_exceeded = (time.time() - start_time) - (loop_idx / self.Exo.control_freq_Hz)

            # 10. Send telemetry data
            telemetry_data = {
                "pos_L": (pos_L, 'pos'),    "pos_R": (pos_R, 'pos'),
                "gc_L": (gc_L, 'gc'),   "gc_R": (gc_R, 'gc'), 
                "cmd_L": (cmd_L, 'cmd'),    "cmd_R": (cmd_R, 'cmd'),
                "hs_L": (hs_idx_L[-1], 'hs'),   "hs_R": (hs_idx_R[-1], 'hs'), "loop_idx": (loop_idx, 'hs'),
                "GRF_L": (GRF_L, 'GRF'),   "GRF_R": (GRF_R, 'GRF'),
                "peak_ext_L": (spline_params_L['peak_extension_torque'], 'torque'), "peak_ext_R": (spline_params_R['peak_extension_torque'], 'torque'),
                "peak_flex_L": (spline_params_L['peak_flexion_torque'], 'torque'), "peak_flex_R": (spline_params_R['peak_flexion_torque'], 'torque'),
                "loop_time_exceeded": (loop_time_exceeded,  None),
            }
            self.teleplot.sendBatchTelemetry(telemetry_data)

            # 11. Wait for the time to reach the next clock cycle
            if (time.time() - start_time) < (loop_idx / self.Exo.control_freq_Hz):
                while (time.time() - start_time) < (loop_idx / self.Exo.control_freq_Hz):
                    pass
            self.data_log['timestamp'][loop_idx] = time.time() - start_time
            loop_idx += 1

    # Signal handler for graceful exit
    def exit_signal_handler(self, sig, frame):
        print("Ctrl + C pressed, shutting down...")

        # save_data(self.data_to_save, self.trial_name, self.pulse_after_start, self.trial_dur_sec)
        cleanup_can(self.Exo.bus, self.Exo.notifier)
        self.GPIO_control.safe_gpio_cleanup()

        gc.collect()
        torch.cuda.empty_cache()

        print("Exiting program")
        os._exit(0)