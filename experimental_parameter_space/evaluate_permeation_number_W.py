import numpy as np
import h_transport_materials as htm
import festim as F
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

# obtain material properties
# inconel
substrate_D = htm.diffusivities.filter(material="inconel_600").filter(
    author="kishimoto"
)[0]
substrate_D_0 = substrate_D.pre_exp.magnitude
substrate_E_D = substrate_D.act_energy.magnitude

substrate_S = htm.solubilities.filter(material="inconel_600").filter(
    author="kishimoto"
)[0]
substrate_S_0 = substrate_S.pre_exp.magnitude
substrate_E_S = substrate_S.act_energy.magnitude

substrate_recomb = htm.recombination_coeffs.filter(material="inconel_600").filter(
    author="rota"
)[0]
substrate_Kr_0 = substrate_recomb.pre_exp.magnitude
substrate_E_Kr = substrate_recomb.act_energy.magnitude

substrate_diss = htm.dissociation_coeffs.filter(material="inconel_600").filter(
    author="rota"
)[0]
substrate_Kd_0 = substrate_diss.pre_exp.magnitude
substrate_E_Kd = substrate_diss.act_energy.magnitude

k_b = F.k_B
T = 600
kd = substrate_Kd_0 * np.exp(substrate_E_Kd / (k_b * T))
ks = substrate_S_0 * np.exp(substrate_E_S / (k_b * T))
D = substrate_D_0 * np.exp(substrate_E_D / (k_b * T))
e = 5e-04
P = 100


def W_testing(P=100, e=5e-04, T=600):

    T = 600
    kd = substrate_Kd_0 * np.exp(substrate_E_Kd / (k_b * T))
    ks = substrate_S_0 * np.exp(substrate_E_S / (k_b * T))
    D = substrate_D_0 * np.exp(substrate_E_D / (k_b * T))

    return (kd * (P**0.5) * e) / (D * ks)


P_testing = np.geomspace(1e-08, 1e08, num=100)
e_testing = np.geomspace(1e-05, 1, num=100)
T_testing = np.linspace(300, 900, num=100)

W_test_P = W_testing(P=P_testing)
W_test_e = W_testing(e_testing)
W_test_T = W_testing(T_testing)

plt.figure()
plt.plot(P_testing, W_test_P, color="black")
plt.xscale("log")
plt.yscale("log")
plt.ylabel("W")
plt.xlabel("Upstream pressure (Pa)")
plt.xlim(min(P_testing), max(P_testing))
ax = plt.gca()
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

plt.figure()
plt.plot(e_testing, W_test_e, color="black")
plt.xscale("log")
plt.yscale("log")
plt.ylabel("W")
plt.xlabel("Sample thickness (m)")
plt.xlim(min(e_testing), max(e_testing))
ax = plt.gca()
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

plt.figure()
plt.plot(T_testing, W_test_T, color="black")
plt.ylabel("W")
plt.xlabel("Temperature (K)")
plt.xlim(min(T_testing), max(T_testing))
ax = plt.gca()
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

X, Y = np.meshgrid(P_testing, e_testing)
Z = W_testing(X, Y)

plt.figure(figsize=(8, 6))

contour = plt.contourf(
    X,
    Y,
    Z,
    levels=1000,
    cmap="viridis",
    norm=LogNorm(vmin=np.min(Z), vmax=np.max(Z)),
)
cbar = plt.colorbar(contour)
cbar.set_label("W value")

plt.xlabel("Upstream pressure (Pa)")
plt.ylabel("Sample thickness (m)")
plt.xscale("log")
plt.yscale("log")

plt.show()
