import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib import cm
import numpy as np
from scipy.integrate import cumulative_trapezoid

from pipe_losses import temperature_values, pressure_values


def hydrogen_atoms_to_mass(atom_count: float) -> float:
    """
    Converts the number of hydrogen atoms to mass in grams.

    Args:
        atom_count: Number of hydrogen atoms.

    returns:
        float: Mass in grams.
    """
    atomic_mass_h = 1.00784 / (6.022e23)  # grams per hydrogen atom
    return atom_count * atomic_mass_h


def label_lines(lines, xvals, colours):
    for line, xval, colour in zip(lines, xvals, colours):
        xdata, ydata = line.get_xdata(), line.get_ydata()
        yval = np.interp(xval, xdata, ydata)  # Interpolate y-value at xval
        label = line.get_label()

        plt.annotate(
            label,
            xy=(xval, yval),
            xytext=(0, 0),
            textcoords="offset points",
            fontsize=12,
            va="center",
            ha="center",
            bbox=dict(facecolor="white", edgecolor="none"),
            color=colour,
        )


def plot_inventory_and_surface_flux(pressure_value: float, temperature_value: float):
    data = np.genfromtxt(
        f"results/item_1_1a/{temperature_value:.1f}K/{pressure_value:.2e}Pa/derived_quantities.csv",
        delimiter=",",
        names=True,
    )
    ts = data["ts"]
    inventory = data["Total_solute_volume_1_H_m1"]
    surface_fluxes = data["solute_flux_surface_2_H_m1_s1"] * -1

    one_year = 3600 * 24 * 365
    ts = np.array(ts) / one_year

    pipe_length = 1000  # m

    inventory = np.array(inventory) * pipe_length
    inventory = hydrogen_atoms_to_mass(inventory)

    surface_fluxes = np.array(surface_fluxes) * pipe_length  # to H/s/km
    surface_fluxes = hydrogen_atoms_to_mass(surface_fluxes) * one_year  # to g/km/yr

    plt.figure()
    plt.plot(ts, inventory, color="black")
    plt.xlabel("Time (yr)")
    plt.ylabel("Mobile hydrogen in wall (g/km)")
    plt.ylim(bottom=0)
    plt.xlim(0, 40)
    ax = plt.gca()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()

    plt.figure()
    plt.fill_between(ts, surface_fluxes, color="black")
    plt.xlabel("Time (yr)")
    plt.ylabel("Instantaneous Hydrogen permeation losses to wall (g/km/yr)")
    plt.xlim(0, 40)
    plt.ylim(bottom=0)
    ax = plt.gca()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()

    plt.figure()
    total_permeation_loss = cumulative_trapezoid(surface_fluxes, x=ts, initial=0)
    plt.plot(ts, total_permeation_loss, color="black")
    plt.xlabel("Time (yr)")
    plt.ylabel("Cumulative hydrogen permeation losses to wall (g/km)")
    plt.xlim(0, 40)
    plt.ylim(bottom=0)
    ax = plt.gca()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()


def plot_mobile_conc_field_over_time(pressure_value: float, temperature_value: float):
    data = np.genfromtxt(
        f"results/item_1_1a/{temperature_value:.1f}K/{pressure_value:.2e}Pa/mobile_concentration_field.txt",
        delimiter=",",
        names=True,
    )
    # print(data.dtype.names)
    # quit()
    x = data["x"]
    x = (x - x[0]) * 1000  # Convert meters to mm and moves left to zero

    c_data = [
        data["t315e07s"],
        data["t158e08s"],
        data["t315e08s"],
        data["t631e08s"],
        data["t126e09s"],
    ]
    c_data_g = []
    for data in c_data:
        c_data_g.append(hydrogen_atoms_to_mass(data))

    # Custom x-positions for labels (closer to the start)
    label_positions = [
        x[500],
        x[1300],
        x[2000],
        x[3000],
        x[4500],
    ]  # Adjust these indices as needed

    year_values = [1, 5, 10, 20, 40]

    norm = Normalize(vmin=min(year_values), vmax=max(year_values))
    colorbar = cm.viridis
    colours = [colorbar(norm(i)) for i in year_values]

    plt.figure()
    lines = []
    for c, label, colour in zip(c_data_g, ["1y", "5y", "10y", "20y", "40y"], colours):
        (line,) = plt.plot(x, c, label=label, color=colour)
        lines.append(line)
    plt.xlim(x[0], x[-1])
    plt.ylim(bottom=0)
    plt.xlabel("Pipe wall (mm)")
    plt.ylabel("Mobile H Concentration (g/m)")
    ax = plt.gca()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    label_lines(lines, label_positions, colours)
    plt.tight_layout()


if __name__ == "__main__":
    # plot_standard()
    # plot_item_1_1a()
    # plot_inventories_by_pressure(pressure_value=pressure_values[-1])
    # plot_inventories_by_temperature()

    plot_inventory_and_surface_flux(
        pressure_value=pressure_values[-1], temperature_value=temperature_values[-1]
    )
    # plot_mobile_conc_field_over_time(
    #     pressure_value=pressure_values[-1], temperature_value=temperature_values[-1]
    # )

    plt.show()
