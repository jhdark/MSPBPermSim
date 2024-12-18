import numpy as np
import festim as F
import matplotlib.pyplot as plt

plt.rc("text", usetex=True)
plt.rc("font", family="serif", size=12)


def compute_permeation_flux(L, P_up, K_r_, K_d, D):
    my_model = F.Simulation(log_level=40)

    my_model.mesh = F.MeshFromVertices(vertices=np.linspace(0, L, num=10))

    my_model.materials = F.Material(id=1, D_0=D, E_D=0, S_0=1, E_S=0)

    my_model.boundary_conditions = [
        F.DissociationFlux(Kd_0=K_d, E_Kd=0, P=P_up, surfaces=1),
        F.RecombinationFlux(Kr_0=K_r_, E_Kr=0, order=2, surfaces=[1, 2]),
    ]

    my_model.T = 300  # ignored here

    my_model.settings = F.Settings(
        absolute_tolerance=1e-7,
        relative_tolerance=1e-10,
        transient=True,
        final_time=100,
    )

    my_model.dt = F.Stepsize(0.01, stepsize_change_ratio=1.1)

    permeation_flux = F.HydrogenFlux(surface=2)
    surface_concentration = F.TotalSurface(field="solute", surface=1)
    surface_concentration_L = F.TotalSurface(field="solute", surface=2)
    my_model.exports = [
        F.DerivedQuantities(
            [permeation_flux, surface_concentration, surface_concentration_L],
            show_units=True,
            filename=f"results/regime_testing/P={P_up:.0e}.csv",
        ),
        # F.TXTExport(field=0, filename=f"results/regime_testing/P={P_up:.2e}.txt"),
    ]

    my_model.initialise()
    my_model.run()


def run_festim_scripts(pressures):

    for P_up in pressures:
        print(f" ---- P = {P_up:.2e} ----")
        compute_permeation_flux(e, P_up, K_r, K_d, D)


K_r = 10
K_d = 100
K_s = (K_d / K_r) ** 0.5
e = 1
D = 2


def W_to_pressure(W):
    return ((W * D * K_s) / (K_d * e)) ** 2


def pressure_to_W(P):
    return (K_d * (P**0.5) * e) / (D * K_s)


W_test_values = np.geomspace(1e-2, 1e02, num=11)

pressure_values = W_to_pressure(W=W_test_values)

# run_festim_scripts(pressure_values)


fluxes = []

for P in pressure_values:
    data = np.genfromtxt(
        f"results/regime_testing/P={P:.0e}.csv", delimiter=",", names=True
    )
    fluxes.append(data["solute_flux_surface_2_H_m2_s1"])


steady_state_fluxes = np.abs([flux.data[-1] for flux in fluxes])

normalised_fluxes = steady_state_fluxes / steady_state_fluxes[5]

plt.plot(W_test_values, normalised_fluxes, marker="o")
plt.xscale("log")
plt.yscale("log")
plt.xlabel("Permeation number, W")
plt.ylabel("Normalised Steady state permeation flux")


Ps_linear = W_test_values[:5]
W_linear = W_to_pressure(W=Ps_linear)
Ps_squared = W_test_values[6:]
W_squared = W_to_pressure(W=Ps_squared)

plt.plot(Ps_linear, W_linear * 1000, linestyle="--", color="gray")
plt.plot(Ps_squared, W_squared**0.5 * 100, linestyle="--", color="gray")

plt.annotate(r"$\propto W \propto \sqrt{P}$", (3.5e0, 1.5e2), color="gray")
plt.annotate(r"$\propto W^{2} \propto P$", (1.75e-2, 4e-2), color="gray")

ax = plt.gca()
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

plt.tight_layout()

plt.show()
