import festim as F

model_barrier = F.Simulation()

barrier_thick = 1e-6
substrate_thick = 3e-3

barrier_left = F.Material(
    id=1,
    D_0=1e-8,
    E_D=0.39,
    S_0=1e22,
    E_S=1.04,
    borders=[0, barrier_thick]
    )

tungsten = F.Material(
    id=2,
    D_0=4.1e-7,
    E_D=0.39,
    S_0=1.87e24,
    E_S=1.04,
    borders=[barrier_thick, substrate_thick + barrier_thick]
    )

barrier_right = F.Material(
    id=3,
    D_0=1e-8,
    E_D=0.39,
    S_0=1e22,
    E_S=1.04,
    borders=[substrate_thick + barrier_thick, substrate_thick + 2 * barrier_thick]
    )

model_barrier.materials = [barrier_left, tungsten, barrier_right]

import numpy as np

vertices_left = np.linspace(0, barrier_thick, num=50)

vertices_mid = np.linspace(
    barrier_thick, substrate_thick + barrier_thick, num=50)

vertices_right = np.linspace(substrate_thick + barrier_thick,
                             substrate_thick + 2*barrier_thick, num=50)

vertices = np.concatenate([vertices_left, vertices_mid, vertices_right])

model_barrier.mesh = F.MeshFromVertices(vertices)

model_barrier.T = 600

left_bc = F.SievertsBC(
    surfaces=1,
    S_0=barrier_left.S_0,
    E_S=barrier_left.E_S,
    pressure=100
    )

right_bc = F.DirichletBC(
    field="solute",
    surfaces=2,
    value=0
    )

model_barrier.boundary_conditions = [left_bc, right_bc]

folder = 'task04'

derived_quantities_with_barrier = F.DerivedQuantities([F.HydrogenFlux(surface=2)], show_units=True)

txt_export = F.TXTExport(
    field='solute',
    filename=folder + '/mobile.txt',
    times=[100, 17000, 8e5],
    )

model_barrier.exports = [derived_quantities_with_barrier, txt_export]

model_barrier.settings = F.Settings(
    absolute_tolerance=1e0,
    relative_tolerance=1e-09,
    final_time=8e5,
    chemical_pot=True,
)


model_barrier.dt = F.Stepsize(
    initial_value=5,
    stepsize_change_ratio=1.1,
    milestones=[100, 17000, 8e5]
)
model_barrier.initialise()
model_barrier.run()

import matplotlib.pyplot as plt

fig, axs = plt.subplots(nrows=1, ncols=3, sharey=True)
data = np.genfromtxt(folder + f"/mobile.txt", skip_header=1, delimiter=",")
data = data[data[:, 0].argsort()]  # make sure data is sorted

xlim_left = (0, barrier_thick * 1.2)
xlim_mid = (None, None)
xlim_right = (substrate_thick, substrate_thick + 2 * barrier_thick)

for ax, xlim in zip(
    axs,
    [xlim_left, xlim_mid, xlim_right],
):
    plt.sca(ax)
    for i, time in enumerate(txt_export.times):
        plt.plot(data[:, 0], data[:, i + 1], label=f"{time:.0f} s")

    plt.xlabel("Depth (m)")
    plt.xlim(*xlim)

axs[0].set_yscale("log")
axs[0].set_ylim(bottom=1e12)
axs[0].set_ylabel("Mobile H concentration (H/m$^3$)")
axs[0].set_title("zoom left")
axs[2].set_title("zoom right")
axs[0].legend(reverse=True)
plt.show()