import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from matplotlib.colors import Normalize, LogNorm
from scipy.integrate import cumulative_trapezoid

pressure_values = [1e0, 1e1, 1e2, 1e3, 1e4, 1e5]


def plot_surface_flux_transient():

    flux_data = []
    t_data = []
    pressure_values = [1e0, 1e1, 1e2, 1e3, 1e4, 1e5]
    # pressure_values = [1e0, 1e1]

    for case in pressure_values:

        data = np.genfromtxt(
            f"results/testing/P={case:.1e}/permeation_standard.csv",
            delimiter=",",
            names=True,
        )
        t = data["ts"]
        flux = data["solute_flux_surface_2_H_m2_s1"] * -1

        t_data.append(t)
        flux_data.append(flux)

    norm_flux = []
    for i in flux_data:
        norm = np.array(i) / i[-1]
        norm_flux.append(norm)

    plt.figure()
    for t_, flux_, P in zip(t_data, flux_data, pressure_values):
        plt.scatter(t_, flux_, label=f"P={P:.1e}")
    plt.legend()

    plt.figure()
    for t_, flux_, P in zip(t_data, norm_flux, pressure_values):
        plt.scatter(t_, flux_, label=f"P={P:.1e}")
    plt.legend()


def plot_profile():
    data_diff = np.genfromtxt(
        "results/mobile_conc_profile_standard.txt", delimiter=",", names=True
    )
    data_surf = np.genfromtxt(
        "results/mobile_conc_profile_barrier.txt", delimiter=",", names=True
    )

    x_standard = data_standard["x"]
    mobile_standard = data_standard["tsteady"]

    x_barrier = data_barrier["x"]
    mobile_barrier = data_barrier["tsteady"]

    plt.figure()
    plt.plot(x_standard, mobile_standard)
    plt.plot(x_barrier, mobile_barrier)
    plt.yscale("log")


def plot_pressure_testing():

    fluxes = []

    for P_value in pressure_values:
        data = np.genfromtxt(
            f"results/pressure_testing/P={P_value:.0e}/permeation_standard.csv",
            delimiter=",",
            names=True,
        )
        fluxes.append(data["solute_flux_surface_2_H_m2_s1"][-1] * -1)

    plt.figure()
    plt.plot(pressure_values, fluxes)
    plt.yscale("log")
    plt.xscale("log")
    plt.grid()
    plt.xlabel("Upstream pressure (Pa)")
    plt.ylabel(r"Surface flux (H m$^{-2}$ s$^{-1}$)")
    Ps_linear = np.logspace(-7, -1)
    Ps_squared = np.logspace(1, 6)
    plt.plot(Ps_linear, Ps_linear * 1000, linestyle="--", color="gray")
    plt.plot(Ps_squared, Ps_squared**0.5 * 500, linestyle="--", color="gray")


def parameter_exploration_steady():

    test_pressure_values = np.geomspace(1e3, 1e5, num=10)
    test_temp_values = np.linspace(300, 800, num=10)

    data = []

    for temp_value in test_temp_values:
        data_per_temperature_value = []
        for pressure_value in test_pressure_values:
            run_data = np.genfromtxt(
                f"results/parameter_exploration/P={pressure_value:.1e}/T={temp_value:.0f}/permeation_standard.csv",
                delimiter=",",
                names=True,
            )
            data_per_temperature_value.append(
                run_data["solute_flux_surface_2_H_m2_s1"] * -1
            )
        data.append(data_per_temperature_value)

    plt.figure()
    for flux_data, temperature_value in zip(data, test_temp_values):
        plt.plot(test_pressure_values, flux_data, label=f"{temperature_value:.1f} K")

    plt.yscale("log")
    plt.xscale("log")
    plt.xlabel("Upstream pressure (Pa)")
    plt.ylabel(r"Surface flux (H m$^{-2}$ s$^{-1}$)")
    plt.legend()


def pressure_from_flux(flux, t, T):

    sample_diameter = 20e-03
    pipe_diameter = 5e-03
    pipe_length = 2
    downstream_volume = pipe_length * np.pi * (pipe_diameter / 2) ** 2

    integrated_flux = cumulative_trapezoid(flux, t, initial=0)
    A = np.pi * (sample_diameter / 2) ** 2  # m^2
    n = integrated_flux * A / (6.022 * 10**23)  # number of hydrogen atoms in mols

    R = 8.314  # J/mol/K

    # Calculate pressure
    P = n * R * T / downstream_volume  # Pa

    return P


def parameter_exploration_pressure_rise():

    test_temp_values = np.linspace(300, 800, num=10)

    P_data = []
    flux_data = []
    t_data = []

    for temp_value in test_temp_values:
        run_data = np.genfromtxt(
            f"results/parameter_exploration/transient/P=1.0e+03/T={temp_value:.0f}/permeation_standard.csv",
            delimiter=",",
            names=True,
        )

        t = run_data["ts"]
        surface_flux = run_data["solute_flux_surface_2_H_m2_s1"] * -1
        P = pressure_from_flux(flux=surface_flux, t=t, T=temp_value)

        P_data.append(P)
        flux_data.append(surface_flux)
        t_data.append(t)

    norm = Normalize(vmin=min(test_temp_values), vmax=max(test_temp_values))
    colorbar = cm.inferno
    sm = plt.cm.ScalarMappable(cmap=colorbar, norm=norm)
    colours = [colorbar(norm(T)) for T in test_temp_values]

    # # ##### Surface flux

    # plt.figure()
    # for time, flux, T_value, colour in zip(
    #     t_data, flux_data, test_temp_values, colours
    # ):
    #     plt.plot(time, flux, label=f"{T_value}K", color=colour)
    # plt.xlabel("Time (s)")
    # plt.ylabel(r"Surface flux (H m$^{-2}$ s$^{-1}$)")
    # ax = plt.gca()
    # ax.spines["top"].set_visible(False)
    # ax.spines["right"].set_visible(False)

    # plt.colorbar(sm, label=r"Temperature (K)", ax=ax)

    # ##### Normlised surface fluxes

    normalised_fluxes = []
    for case in flux_data:
        normalised_fluxes.append(case / case[-1])

    plt.figure()
    for time, flux, T_value, colour in zip(
        t_data, normalised_fluxes, test_temp_values, colours
    ):
        plt.plot(time, flux, label=f"{T_value}K", color=colour)
    plt.xlabel(r"Time (s)")
    plt.ylabel(r"Normalised Surface flux")
    ax = plt.gca()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.colorbar(sm, label=r"Temperature (K)", ax=ax)

    # # ##### pressure rise ##### #

    # plt.figure()
    # for time, pressure, T_value, colour in zip(
    #     t_data, P_data, test_temp_values, colours
    # ):
    #     plt.plot(time, pressure, label=f"{T_value}K", color=colour)
    # plt.xlabel("Time (s)")
    # plt.ylabel(r"Pressure (Pa)")
    # ax = plt.gca()
    # ax.spines["top"].set_visible(False)
    # ax.spines["right"].set_visible(False)

    # plt.colorbar(sm, label=r"Temperature (K)", ax=ax)


def parameter_exploration_pressure_rise_limits():

    test_temp_values = np.linspace(300, 800, num=10)
    test_pressure_values = np.geomspace(1e3, 1e5, num=10)

    P_data = []

    for pressure_value in test_pressure_values:

        P_data_per_pressure = []

        for temp_value in test_temp_values:
            run_data = np.genfromtxt(
                f"results/parameter_exploration/transient/P={pressure_value:.1e}/T={temp_value:.0f}/permeation_standard.csv",
                delimiter=",",
                names=True,
            )

            t = run_data["ts"]
            surface_flux = run_data["solute_flux_surface_2_H_m2_s1"] * -1
            P = pressure_from_flux(flux=surface_flux, t=t, T=temp_value)

            P_data_per_pressure.append(P[-1])

        P_data.append(P_data_per_pressure)

    ones = np.ones_like(test_pressure_values)
    plot_pressure_values = []

    for P in test_pressure_values:
        plot_pressure_values.append(ones * P)

    # ##### Pressure ranges

    plt.figure()
    for P_u, P_d in zip(plot_pressure_values, P_data):
        plt.plot(P_u, P_d, color="black")
    plt.xlabel("Upstream pressure (Pa)")
    plt.ylabel("Final downstream pressure value (Pa)")
    plt.xscale("log")
    plt.ylim(bottom=0)
    ax = plt.gca()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def surface_flux_transient_800K_varying_pressure():

    test_pressure_values = np.geomspace(1e3, 1e5, num=10)

    flux_data = []
    t_data = []

    for pressure_value in test_pressure_values:
        run_data = np.genfromtxt(
            f"results/parameter_exploration/transient/P={pressure_value:.1e}/T=800/permeation_standard.csv",
            delimiter=",",
            names=True,
        )

        t = run_data["ts"]
        surface_flux = run_data["solute_flux_surface_2_H_m2_s1"] * -1

        flux_data.append(surface_flux)
        t_data.append(t)

    norm = LogNorm(vmin=min(test_pressure_values), vmax=max(test_pressure_values))
    colorbar = cm.viridis
    sm = plt.cm.ScalarMappable(cmap=colorbar, norm=norm)
    colours = [colorbar(norm(P)) for P in test_pressure_values]

    # ##### Surface flux

    plt.figure()
    for time, flux, pressure_value, colour in zip(
        t_data, flux_data, test_pressure_values, colours
    ):
        plt.plot(time, flux, color=colour)
    plt.xlabel("Time (s)")
    plt.ylabel(r"Surface flux (H m$^{-2}$ s$^{-1}$)")
    plt.xlim(0, 400)
    plt.ylim(bottom=0)
    ax = plt.gca()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.colorbar(sm, label=r"Pressure (Pa)", ax=ax)


def evaulate_time_to_steady_state():

    test_temp_values = np.linspace(300, 800, num=10)
    pressure_values = [1e0, 1e1, 1e2, 1e3, 1e4, 1e5]
    time_to_steady = []

    for pressure_value in pressure_values:

        time_to_steady_per_pressure = []

        for temp_value in test_temp_values:
            run_data = np.genfromtxt(
                f"results/parameter_exploration/P={pressure_value:.0e}/T={temp_value:.0f}/permeation_standard.csv",
                delimiter=",",
                names=True,
            )
            t = run_data["ts"]
            surface_flux = run_data["solute_flux_surface_2_H_m2_s1"] * -1
            time_ind = np.where(surface_flux > 0.99 * surface_flux[-1])[0][0]
            print(f"{temp_value}, {pressure_value}")
            time_to_steady_per_pressure.append(t[time_ind])

        time_to_steady.append(time_to_steady_per_pressure)

    norm = LogNorm(vmin=min(pressure_values), vmax=max(pressure_values))
    colorbar = cm.inferno
    sm = plt.cm.ScalarMappable(cmap=colorbar, norm=norm)
    colours = [colorbar(norm(T)) for T in test_temp_values]

    plt.figure()

    for pressure_value, times, colour in zip(pressure_values, time_to_steady, colours):
        plt.plot(test_temp_values, times, color=colour)
    plt.ylabel(r"Time to steady-state (s)")
    plt.xlabel(r"Temperature (K)")
    plt.yscale("log")
    ax = plt.gca()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.colorbar(sm, label=r"Upstream pressure (Pa)", ax=ax)


# plot_surface_flux_transient()
# plot_profile()
plot_pressure_testing()
# parameter_exploration_steady()
# parameter_exploration_pressure_rise()
# parameter_exploration_pressure_rise_limits()
# evaulate_time_to_steady_state()
# surface_flux_transient_800K_varying_pressure()

plt.show()
