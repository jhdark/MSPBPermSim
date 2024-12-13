import numpy as np
import festim as F
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

from festim_sim import (
    substrate_D_0,
    substrate_E_D,
    substrate_S_0,
    substrate_E_S,
    substrate_Kd_0,
    substrate_E_Kd,
)

k_b = F.k_B


def W_number(K_d, thickness, pressure, diffusivity, solubility):
    return (K_d * (pressure**0.5) * thickness) / (diffusivity * solubility)


default_P = 1e4  # Pa
default_e = 974e-6  # m
default_T = 300 + 273.15  # K


def W_testing(P=default_P, e=default_e, T=default_T):

    return W_number(
        K_d=substrate_Kd_0 * np.exp(-substrate_E_Kd / (k_b * T)),
        thickness=e,
        pressure=P,
        diffusivity=substrate_D_0 * np.exp(-substrate_E_D / (k_b * T)),
        solubility=substrate_S_0 * np.exp(-substrate_E_S / (k_b * T)),
    )


P_testing = np.geomspace(
    1e2, 1e05, num=100
)  # Pa  range taken from https://doi.org/10.1016/j.nme.2021.101062 Section 2.5
e_testing = np.geomspace(
    2e-4, 1e-3, num=100
)  # m minimum value taken from https://doi.org/10.1016/j.nme.2021.101062 Section 2.5
T_testing = np.linspace(200, 400, num=100) + 273.15  # K

e_ticks = np.linspace(2e-4, 1e-3, num=9)
e_plot = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

plt.rc("text", usetex=True)
plt.rc("font", family="serif", size=12)

plt.figure()
plt.title(f"T = {default_T} K, e = {default_e:.1e} m")
W_test_P = W_testing(P=P_testing)
plt.plot(P_testing, W_test_P, color="black")
plt.xscale("log")
plt.ylabel("Permeation number, W")
plt.xlabel("Upstream pressure (Pa)")
plt.xlim(min(P_testing), max(P_testing))
plt.ylim(bottom=0)
ax = plt.gca()
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
plt.tight_layout()

plt.figure()
plt.title(f"T = {default_T} K, P = {default_P:.1e} Pa")
W_test_e = W_testing(e=e_testing)
plt.plot(e_testing, W_test_e, color="black")
plt.ylabel("Permeation number, W")
plt.xlabel("Sample thickness (m)")
plt.xlim(min(e_testing), max(e_testing))
plt.xticks(ticks=e_ticks, labels=e_plot)
plt.ylim(bottom=0)
ax = plt.gca()
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
plt.tight_layout()

plt.figure()
plt.title(f"P = {default_P:.1e} Pa, e = {default_e:.1e} m")
W_test_T = W_testing(T=T_testing)
plt.plot(T_testing, W_test_T, color="black")
plt.ylabel("Permeation number, W")
plt.xlabel("Temperature (K)")
plt.xlim(min(T_testing), max(T_testing))
plt.ylim(bottom=0)
ax = plt.gca()
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
plt.tight_layout()

X, Y = np.meshgrid(P_testing, e_testing)
Z = W_testing(P=X, e=Y)

plt.figure(figsize=(8, 6))
plt.title(f"T = {default_T} K")

# option for diverging colourbar
# norm = TwoSlopeNorm(vmin=1 / np.max(Z), vcenter=1, vmax=np.max(Z))
# contour = plt.contourf(X, Y, Z, levels=1000, cmap="coolwarm", norm=norm)

contour = plt.contourf(X, Y, Z, levels=1000, cmap="viridis")
ax = plt.gca()
CS = ax.contour(X, Y, Z, levels=[1, 4, 16], colors="white")
ax.clabel(CS, CS.levels, fontsize=10)
cbar = plt.colorbar(contour, format="%.1f")
cbar.set_label("Permeation number, W")
plt.xlabel("Upstream pressure (Pa)")
plt.ylabel("Sample thickness (mm)")
plt.xscale("log")
plt.yticks(ticks=e_ticks, labels=e_plot)
# plt.yscale("log")

plt.show()
