import numpy as np
from festim_sim import festim_model_standard
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib import cm
from scipy.integrate import cumulative_trapezoid


def test_gauge_range_parameters(
    sample_diameter,
    sample_thickness,
    downstream_volume,
    steady_state_point,
    temperature_max_min,
    pressure_max_min,
    PRF_values,
    gauge_model,
):
    """
    Give a parameter space for given parameters and plot against detectable pressure range

    Args:
        sample_diameter (float): sample diameter in m
        sample_thickness (float): sample thickness in m
        steady_state_point (float): steady state point in decimal
        temperature_max_min (list of floats): max and min temperature in K e.g. [450, 750]
        pressure_max_min (list of floats): max and min pressure in Pa e.g. [1e2, 1e5]
        PRF_values (list of floats): PRF values to test e.g. [1, 10, 100, 1000]
        gauge_model (float): gauge model in Torr
    """

    # run festim simulations
    for pressure_value in pressure_max_min:
        for temp_value in temperature_max_min:
            print(f"Testing case P={pressure_value:.2e}, T={temp_value:.0f}")
            festim_model_standard(
                T=temp_value,
                pressure=pressure_value,
                foldername=f"results/parameter_exploration_alt/P={pressure_value:.2e}/T={temp_value:.0f}",
                regime="diff",
                atol=1e08,
                final_time=1e7,
                sample_thicnkess=sample_thickness,
            )

    def pressure_from_flux(flux, t, T, sample_diameter, downstream_volume):
        """Evaulates the pressure in downstream given the flux, temperature and downstream properties"""

        integrated_flux = cumulative_trapezoid(flux, t, initial=0)
        A = np.pi * (sample_diameter / 2) ** 2  # m^2
        n = integrated_flux * A / (6.022 * 10**23)  # number of hydrogen atoms in mols

        R = 8.314  # J/mol/K

        # Calculate pressure
        P = n * R * T / downstream_volume  # Pa

        return P

    # data processing
    P_data = []
    for PRF_case in PRF_values:
        P_data_per_PRF = []
        for P_value in pressure_max_min:
            P_data_per_pressure = []
            for temp_value in temperature_max_min:
                run_data = np.genfromtxt(
                    f"results/parameter_exploration/P={P_value:.2e}/T={temp_value:.0f}/permeation_standard.csv",
                    delimiter=",",
                    names=True,
                )

                t = run_data["ts"]
                surface_flux = run_data["solute_flux_surface_2_H_m2_s1"] * -1

                # find time to steady state
                time_to_steay_ind = np.where(
                    surface_flux > steady_state_point * surface_flux[-1]
                )[0][0]
                t, surface_flux = (
                    t[:time_to_steay_ind],
                    surface_flux[:time_to_steay_ind],
                )

                P = pressure_from_flux(
                    flux=surface_flux / PRF_case,
                    t=t,
                    T=temp_value,
                    sample_diameter=sample_diameter,
                    downstream_volume=downstream_volume,
                )
                # print(f"{P[-1]:.1e}")
                P_data_per_pressure.append(P[-1])

            P_data_per_PRF.append(P_data_per_pressure)

        P_data.append(P_data_per_PRF)

    # plotting

    norm = LogNorm(vmin=min(PRF_values), vmax=max(PRF_values))
    colorbar = cm.viridis
    colours = [colorbar(norm(i)) for i in PRF_values]

    plt.figure()

    for pressure_data_case, colour in zip(P_data, colours):

        x_values = [
            pressure_max_min[0],
            pressure_max_min[0],
            pressure_max_min[1],
            pressure_max_min[1],
        ]
        y_values = [
            pressure_data_case[0][1],
            pressure_data_case[0][0],
            pressure_data_case[1][0],
            pressure_data_case[1][1],
        ]

        plt.fill(x_values, y_values, alpha=0.7, color=colour)
        plt.annotate(
            f"PRF={PRF_values[colours.index(colour)]}",
            xy=(2e5, pressure_data_case[1][1] * 0.5),
            color=colour,
        )

    # plot gauge range
    gauge_max = gauge_model * 133.3
    # min detectable pressure 0.05% of full scale
    gauge_min = 0.0005 * gauge_max

    x_plot_lims = np.geomspace(1e2, 1e5, num=100)

    # plt.figure()
    plt.fill_between(x_plot_lims, gauge_min, gauge_max, color="grey", alpha=0.5)
    plt.hlines(gauge_min, xmin=x_plot_lims[0], xmax=x_plot_lims[-1], color="grey")
    plt.hlines(gauge_max, xmin=x_plot_lims[0], xmax=x_plot_lims[-1], color="grey")
    plt.annotate(
        f"Baratron {gauge_model} Torr gauge range",
        xy=(x_plot_lims[2], gauge_max * 1.5),
        color="grey",
        ha="left",
    )

    plt.xlabel("Upstream Pressure (Pa)")
    plt.ylabel("Steady flux final downstream pressure (Pa)")
    plt.xscale("log")
    plt.yscale("log")
    plt.xlim(x_plot_lims[0], 1e6)
    ax = plt.gca()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()


sample_diameter = 20e-03  # in m
sample_thickness = 1e-03  # in m
downstream_pipe_diameter = 5e-03  # in m
downstream_pipe_length = 2  # in m

downstream_volume = (
    downstream_pipe_length * np.pi * (downstream_pipe_diameter / 2) ** 2
)  # in m3

test_gauge_range_parameters(
    sample_diameter=sample_diameter,
    sample_thickness=sample_thickness,
    downstream_volume=downstream_volume,
    steady_state_point=0.999,
    temperature_max_min=[450, 750],
    pressure_max_min=[1e2, 1e5],
    PRF_values=[1, 10, 100, 1000],
    gauge_model=1,
)
plt.show()
