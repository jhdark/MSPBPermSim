import matplotlib.pyplot as plt
import numpy as np
from evaluate_time_to_steady import test_temperature_values

plt.rc("text", usetex=True)
plt.rc("font", family="serif", size=12)

time_to_steady = []

for T_value in test_temperature_values:
    run_data = np.genfromtxt(
        f"results/time_to_ss/T={T_value:.1f}K/permeation_standard.csv",
        delimiter=",",
        names=True,
    )
    t = run_data["ts"]
    surface_flux = run_data["solute_flux_surface_2_H_m2_s1"] * -1
    time_ind = np.where(surface_flux > 0.999 * surface_flux[-1])[0][0]
    time_to_steady.append(t[time_ind])

plt.figure()

plt.plot(test_temperature_values, time_to_steady, color="black")
plt.ylabel(r"Time to steady-state (s)")
plt.xlabel(r"Temperature (K)")
plt.yscale("log")
plt.xlim(400, 750)
hour = 3600

for value in [1, 6, 12, 24, 48]:
    plt.hlines(
        y=hour * value,
        xmin=400,
        xmax=750,
        color="grey",
        alpha=0.5,
        linestyle="dashed",
    )
    plt.annotate(f"{value}h", xy=[730, hour * value * 1.2], color="grey", alpha=0.5)

ax = plt.gca()
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

plt.show()
