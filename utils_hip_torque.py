"""Map gait-cycle percentage to hip torque using the TBE spline profile."""

import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import CubicHermiteSpline

# LG, from Supplementary Item of https://doi.org/10.1109/TNSRE.2022.3196665
DEFAULT_SPLINE_PARAMS = {
    "peak_time": 34,
    "rise_time": 18.8,
    "mid_time": 47.1,
    "mid_dur": 1.3,
    "peak_time_2": 82.2,
    "fall_time": 21.4,
    "gp_offset": 16,
    "peak_extension_torque": -0.370,
    "peak_flexion_torque": 0.221,
}


def spline_generator(
    peak_time,
    rise_time,
    mid_time,
    mid_dur,
    peak_time_2,
    fall_time,
    gp_offset,
    peak_extension_torque,
    peak_flexion_torque,
    scale_factor=0.5,
    control_Hz=100,
):
    spline_x = np.array(
        [
            peak_time - rise_time - gp_offset,
            peak_time - gp_offset,
            mid_time - mid_dur / 2 - gp_offset,
            mid_time + mid_dur / 2 - gp_offset,
            peak_time_2 - gp_offset,
            peak_time_2 + fall_time - gp_offset,
            100 + peak_time - rise_time - gp_offset,
        ]
    )
    spline_y = (
        np.array([0.0, peak_extension_torque, 0.0, 0.0, peak_flexion_torque, 0.0, 0.0])
        * scale_factor
    )
    spline_dydx = np.array([0, 0.0, 0.0, 0, 0.0, 0.0, 0])
    spline_profile = CubicHermiteSpline(
        spline_x, spline_y, spline_dydx, extrapolate="periodic"
    )

    spline_profile_arr = spline_profile(np.linspace(0, 99, control_Hz))

    return spline_profile_arr


def percent_gc_to_index(percent_gc: float) -> int:
    """Convert gait-cycle percentage [0, 100] to a spline lookup index [0, 99]."""
    return int(np.clip(percent_gc, 0, 99))


def plot_default_spline(
    body_mass_kg: float = 80.0,
    assistance_scale: float = 1.0,
    control_freq_Hz: float = 100,
    save_path: str | None = None,
    show: bool = True,
) -> plt.Figure:
    """Plot the default TBE hip torque spline over one gait cycle."""
    spline = spline_generator(**DEFAULT_SPLINE_PARAMS, control_Hz=int(control_freq_Hz))
    percent_gc = np.linspace(0, 99, len(spline))
    torque_nm = spline * body_mass_kg * assistance_scale

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(percent_gc, torque_nm, color="tab:blue")
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_xlabel("Gait cycle (%)")
    ax.set_ylabel("Hip torque (Nm)")
    ax.set_title("Default TBE hip torque spline")
    ax.set_xlim(0, 99)
    ax.grid(True)
    fig.tight_layout()

    if show:
        plt.show()

    return fig


class HipTorqueProfile:
    """Look up hip torque (Nm) from Vicon gait-cycle percentage."""

    def __init__(
        self,
        body_mass_kg: float,
        control_freq_Hz: float = 100,
        assistance_scale: float = 1.0,
        spline_params: dict | None = None,
    ):
        self.body_mass_kg = body_mass_kg
        self.control_freq_Hz = control_freq_Hz
        self.assistance_scale = assistance_scale

        params = DEFAULT_SPLINE_PARAMS if spline_params is None else spline_params
        self.spline = spline_generator(
            peak_time=params["peak_time"],
            rise_time=params["rise_time"],
            mid_time=params["mid_time"],
            mid_dur=params["mid_dur"],
            peak_time_2=params["peak_time_2"],
            fall_time=params["fall_time"],
            gp_offset=params["gp_offset"],
            peak_extension_torque=params["peak_extension_torque"],
            peak_flexion_torque=params["peak_flexion_torque"],
            control_Hz=int(control_freq_Hz),
        )

    def torque_from_percent_gc(self, percent_gc: float) -> float:
        """Return hip torque (Nm) for one leg at the given gait-cycle percentage."""
        idx = percent_gc_to_index(percent_gc)
        return self.spline[idx] * self.body_mass_kg * self.assistance_scale

    def torque_from_percent_gc_lr(
        self, percent_gc_l: float, percent_gc_r: float
    ) -> tuple[float, float]:
        """Return (cmd_L, cmd_R) hip torques (Nm) for both legs."""
        return (
            self.torque_from_percent_gc(percent_gc_l),
            self.torque_from_percent_gc(percent_gc_r),
        )


if __name__ == "__main__":
    import argparse
    plot_default_spline()
# cd "c:\CMU\MetaMobility Lab\Personal Code\git_clone\Vicon"
# python utils_hip_torque.py