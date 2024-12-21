import matplotlib.pyplot as plt
import numpy as np
from festim_sim import pressure_from_flux
from evaluate_pressure_rise import test_temperature_values, test_pressure_values
from matplotlib.colors import LogNorm, Normalize
from matplotlib import cm

# plt.rc("text", usetex=True)
# plt.rc("font", family="serif", size=12)


def plot_pressure_with_upstream_pressure():
    P_data = []
    t_data = []
    surface_flux_data = []

    for P_value in test_pressure_values:
        run_data = np.genfromtxt(
            f"results/parameter_exploration/P={P_value:.2e}/T=700/permeation_standard.csv",
            delimiter=",",
            names=True,
        )

        t = run_data["ts"]
        surface_flux = run_data["solute_flux_surface_2_H_m2_s1"] * -1

        time_to_steay_ind = np.where(surface_flux > 0.999 * surface_flux[-1])[0][0]
        t, surface_flux = t[:time_to_steay_ind], surface_flux[:time_to_steay_ind]

        P = pressure_from_flux(flux=surface_flux, t=t, T=700)

        P_data.append(P)
        t_data.append(t)
        surface_flux_data.append(surface_flux)

    norm = LogNorm(vmin=min(test_pressure_values), vmax=max(test_pressure_values))
    colorbar = cm.viridis
    sm = plt.cm.ScalarMappable(cmap=colorbar, norm=norm)
    colours = [colorbar(norm(i)) for i in test_pressure_values]

    fig, axs = plt.subplots(2, 1, sharex=True, figsize=(6, 8))

    # Plot on the first axis
    for colour, flux_values, t_values in zip(colours, surface_flux_data, t_data):
        axs[0].plot(t_values, flux_values, color=colour)
    axs[0].set_ylabel(r"Permeation flux (H m$^{-2}$ s$^{-1}$)")
    axs[0].set_ylim(bottom=1e15)
    axs[0].set_yscale("log")
    axs[0].spines["top"].set_visible(False)
    axs[0].spines["right"].set_visible(False)
    axs[0].set_title(r"T=700 K")

    # Plot on the second axis
    for colour, P_values, t_values in zip(colours, P_data, t_data):
        axs[1].plot(t_values, P_values, color=colour)
    axs[1].set_ylabel(r"Downstream pressure (Pa)")
    axs[1].set_xlabel(r"Time (s)")
    # axs[1].set_xscale("log")
    # axs[1].set_ylim(bottom=1e12)

    axs[1].set_yscale("log")
    axs[1].set_ylim(1e-01, 1e3)
    axs[1].set_xlim(left=0)
    # axs[1].set_ylim(bottom=0)
    axs[1].spines["top"].set_visible(False)
    axs[1].spines["right"].set_visible(False)

    fig.colorbar(sm, label=r"Upstream pressure (Pa)", ax=axs, shrink=0.75)


def plot_pressure_with_temperature():
    P_data = []
    t_data = []
    surface_flux_data = []

    for T_value in test_temperature_values:
        run_data = np.genfromtxt(
            f"results/parameter_exploration/P=1.00e+05/T={T_value:.0f}/permeation_standard.csv",
            delimiter=",",
            names=True,
        )

        t = run_data["ts"]
        surface_flux = run_data["solute_flux_surface_2_H_m2_s1"] * -1

        time_to_steay_ind = np.where(surface_flux > 0.999 * surface_flux[-1])[0][0]
        t, surface_flux = t[:time_to_steay_ind], surface_flux[:time_to_steay_ind]

        P = pressure_from_flux(flux=surface_flux, t=t, T=T_value)

        P_data.append(P)
        t_data.append(t)
        surface_flux_data.append(surface_flux)

    norm = Normalize(
        vmin=min(test_temperature_values), vmax=max(test_temperature_values)
    )
    colorbar = cm.inferno
    sm = plt.cm.ScalarMappable(cmap=colorbar, norm=norm)
    colours = [colorbar(norm(i)) for i in test_temperature_values]

    normalised_fluxes = []
    for fluxes in surface_flux_data:
        normalised_fluxes.append(fluxes / fluxes[-1])

    fig, axs = plt.subplots(2, 1, sharex=True, figsize=(6, 8))

    # Plot on the first axis
    for colour, flux_values, t_values in zip(colours, surface_flux_data, t_data):
        axs[0].plot(t_values, flux_values, color=colour)
    axs[0].set_ylabel(r"Permeation flux (H m$^{-2}$ s$^{-1}$)")
    axs[0].set_ylim(bottom=1e12)
    axs[0].set_yscale("log")
    axs[0].spines["top"].set_visible(False)
    axs[0].spines["right"].set_visible(False)
    axs[0].set_title(r"P$_{\mathrm{up}}$=10$^{5}$ Pa")

    # Plot on the second axis
    for colour, P_values, t_values in zip(colours, P_data, t_data):
        axs[1].plot(t_values, P_values, color=colour)
    axs[1].set_ylabel(r"Downstream pressure (Pa)")
    axs[1].set_xlabel(r"Time (s)")
    axs[1].set_xscale("log")
    axs[1].set_xlim(left=1e1)
    axs[1].set_yscale("log")
    axs[1].set_ylim(1e-01, 1e3)
    axs[1].spines["top"].set_visible(False)
    axs[1].spines["right"].set_visible(False)

    fig.colorbar(sm, label=r"Temperature (K)", ax=axs, shrink=0.75)


def plot_pressure_and_temp():
    P_data = []
    t_data = []

    for P_value in test_pressure_values:
        P_data_per_pressure = []
        t_data_per_pressure = []
        for temp_value in test_temperature_values:
            run_data = np.genfromtxt(
                f"results/parameter_exploration/P={P_value:.2e}/T={temp_value:.0f}/permeation_standard.csv",
                delimiter=",",
                names=True,
            )

            t = run_data["ts"]
            surface_flux = run_data["solute_flux_surface_2_H_m2_s1"] * -1

            time_to_steay_ind = np.where(surface_flux > 0.999 * surface_flux[-1])[0][0]
            t, surface_flux = t[:time_to_steay_ind], surface_flux[:time_to_steay_ind]

            P = pressure_from_flux(flux=surface_flux, t=t, T=temp_value)

            P_data_per_pressure.append(P)
            t_data_per_pressure.append(t)

        P_data.append(P_data_per_pressure)
        t_data.append(t_data_per_pressure)

    norm_T = Normalize(
        vmin=min(test_temperature_values), vmax=max(test_temperature_values)
    )
    colorbar_T = cm.inferno
    sm_T = plt.cm.ScalarMappable(cmap=colorbar_T, norm=norm_T)
    colours_T = [colorbar_T(norm_T(T)) for T in test_temperature_values]

    norm_P = LogNorm(vmin=min(test_pressure_values), vmax=max(test_pressure_values))
    colorbar_P = cm.viridis
    sm_P = plt.cm.ScalarMappable(cmap=colorbar_P, norm=norm_P)
    colours_P = [colorbar_P(norm_P(P)) for P in test_pressure_values]

    fig, axs = plt.subplots(2, 1, sharex=True, figsize=(6, 8))

    # Plot on the first axis
    for P_values, t_values in zip(P_data, t_data):
        for pressure, time, colour in zip(P_values, t_values, colours_T):
            axs[0].plot(time, pressure, color=colour)
    axs[0].set_ylabel(r"Downstream pressure (Pa)")
    axs[0].set_ylim(1e-01, 1e3)
    axs[0].set_xlim(left=5e01)
    axs[0].set_yscale("log")
    axs[0].set_xscale("log")
    axs[0].spines["top"].set_visible(False)
    axs[0].spines["right"].set_visible(False)
    # axs[0].colorbar(sm_T, label=r"Temperature (K)")

    # Plot on the second axis
    for P_values, t_values, colour in zip(P_data, t_data, colours_P):
        for pressure, time in zip(P_values, t_values):
            axs[1].plot(time, pressure, color=colour)
    axs[1].set_ylabel(r"Downstream pressure (Pa)")
    axs[1].set_xlabel(r"Time (s)")
    axs[1].set_ylim(1e-01, 1e3)
    axs[1].set_xlim(left=5e01)
    axs[1].set_yscale("log")
    axs[1].set_xscale("log")
    axs[1].spines["top"].set_visible(False)
    axs[1].spines["right"].set_visible(False)
    # axs[1].colorbar(sm_P, label=r"Pressure (Pa)")

    fig.colorbar(sm_T, label=r"Temperature (K)", ax=axs[0])
    fig.colorbar(sm_P, label=r"Pressure (Pa)", ax=axs[1])


def plot_pressure_ranges():
    P_data = []
    t_data = []

    for P_value in test_pressure_values:
        P_data_per_pressure = []
        for temp_value in test_temperature_values:
            run_data = np.genfromtxt(
                f"results/parameter_exploration/P={P_value:.2e}/T={temp_value:.0f}/permeation_standard.csv",
                delimiter=",",
                names=True,
            )

            t = run_data["ts"]
            surface_flux = run_data["solute_flux_surface_2_H_m2_s1"] * -1

            time_to_steay_ind = np.where(surface_flux > 0.999 * surface_flux[-1])[0][0]
            t, surface_flux = t[:time_to_steay_ind], surface_flux[:time_to_steay_ind]

            P = pressure_from_flux(flux=surface_flux, t=t, T=temp_value)

            P_data_per_pressure.append(P[-1])

        P_data.append(P_data_per_pressure)

    norm = Normalize(
        vmin=min(test_temperature_values), vmax=max(test_temperature_values)
    )
    colorbar = cm.inferno
    sm = plt.cm.ScalarMappable(cmap=colorbar, norm=norm)
    colours = [colorbar(norm(i)) for i in test_temperature_values]

    plt.figure()
    for P_up_values, P_down_values in zip(test_pressure_values, P_data):
        x = np.ones_like(P_down_values) * P_up_values
        for x_value, P_value, colour in zip(x, P_down_values, colours):
            plt.scatter(x_value, P_value, color=colour)
        # plt.plot(x, P_down_values, color="black")

    plt.xlabel("Upstream Pressure (Pa)")
    plt.ylabel("Steady flux final downstream pressure (Pa)")
    plt.xscale("log")
    plt.yscale("log")
    plt.ylim(1e-01, 1e3)
    # plt.xlim(left=1e1)
    ax = plt.gca()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.colorbar(sm, label=r"Temperature (K)", ax=ax)


def plot_pressure_ranges_with_PRF():
    P_data = []
    P_data_PRF_10 = []
    P_data_PRF_100 = []
    P_data_PRF_1000 = []

    for P_value in test_pressure_values:
        P_data_per_pressure = []
        P_data_PRF_10_per_pressure = []
        P_data_PRF_100_per_pressure = []
        P_data_PRF_1000_per_pressure = []

        for temp_value in test_temperature_values:
            run_data = np.genfromtxt(
                f"results/parameter_exploration/P={P_value:.2e}/T={temp_value:.0f}/permeation_standard.csv",
                delimiter=",",
                names=True,
            )

            t = run_data["ts"]
            surface_flux = run_data["solute_flux_surface_2_H_m2_s1"] * -1

            time_to_steay_ind = np.where(surface_flux > 0.999 * surface_flux[-1])[0][0]
            t, surface_flux = t[:time_to_steay_ind], surface_flux[:time_to_steay_ind]

            P = pressure_from_flux(flux=surface_flux, t=t, T=temp_value)

            P_PRF_10 = pressure_from_flux(flux=surface_flux / 10, t=t, T=temp_value)
            P_PRF_100 = pressure_from_flux(flux=surface_flux / 100, t=t, T=temp_value)
            P_PRF_1000 = pressure_from_flux(flux=surface_flux / 1000, t=t, T=temp_value)

            P_data_per_pressure.append(P[-1])
            P_data_PRF_10_per_pressure.append(P_PRF_10[-1])
            P_data_PRF_100_per_pressure.append(P_PRF_100[-1])
            P_data_PRF_1000_per_pressure.append(P_PRF_1000[-1])

        P_data.append(P_data_per_pressure)
        P_data_PRF_10.append(P_data_PRF_10_per_pressure)
        P_data_PRF_100.append(P_data_PRF_100_per_pressure)
        P_data_PRF_1000.append(P_data_PRF_1000_per_pressure)

    PRF_values = [1, 10, 100, 1000]
    norm = LogNorm(vmin=min(PRF_values), vmax=max(PRF_values))
    colorbar = cm.viridis
    colours = [colorbar(norm(i)) for i in PRF_values]

    # pressure gauge range 1 Torr
    Torr_model = 1
    gauge_max = Torr_model * 133.3
    # min detectable pressure 0.05% of full scale
    gauge_min = 0.0005 * gauge_max

    x_min, x_max = 6e1, 1e6
    x_values = np.geomspace(x_min, 1e5, num=100)

    plt.figure()
    plt.fill_between(x_values, gauge_min, gauge_max, color="grey", alpha=0.5)
    plt.hlines(gauge_min, xmin=x_values[0], xmax=x_values[-1], color="grey")
    plt.hlines(gauge_max, xmin=x_values[0], xmax=x_values[-1], color="grey")
    plt.annotate(
        f"Baratron {Torr_model} Torr gauge range",
        xy=(x_values[2], gauge_max * 1.5),
        color="grey",
        ha="left",
    )

    # colours = ["black", "red", "blue", "green"]
    for colour, data in zip(
        colours, [P_data, P_data_PRF_10, P_data_PRF_100, P_data_PRF_1000]
    ):
        for P_up_values, P_down_values in zip(test_pressure_values, data):
            x = np.ones_like(P_down_values) * P_up_values
            plt.plot(x, P_down_values, color=colour)
        plt.annotate(
            f"PRF={PRF_values[colours.index(colour)]}",
            xy=(2e5, data[-1][-1] * 0.5),
            color=colour,
        )

    plt.xlabel("Upstream Pressure (Pa)")
    plt.ylabel("Steady flux final downstream pressure (Pa)")
    plt.xscale("log")
    plt.yscale("log")
    # plt.ylim(1e-01, 1e3)
    plt.xlim(x_min, x_max)
    ax = plt.gca()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()

    # plt.colorbar(sm, label=r"Temperature (K)", ax=ax)


if __name__ == "__main__":
    # plot_pressure_with_upstream_pressure()
    plot_pressure_with_temperature()
    # plot_pressure_and_temp()
    # plot_pressure_ranges()
    # plot_pressure_ranges_with_PRF()

    plt.show()
