import festim as F
import numpy as np
from scipy.integrate import cumulative_trapezoid
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib import cm

test_temperature_values = np.linspace(450, 750, num=7)
test_pressure_values = np.geomspace(1e2, 1e5, num=10)
sample_thickness = 1e-03


def festim_model_standard(
    T,
    pressure,
    foldername,
    regime="diff",
    final_time=1e7,
    atol=1e-08,
    sample_thickness=1e-03,
):
    """Run a standard festim model for permeation of hydrogen through a sample

    Args:
        T (float): temperature in K
        pressure (float): pressure in Pa
        foldername (str): foldername to save the results
        regime (str, optional): permeation regime, either diffusion, "diff" or surface "surf" limited.
            Defaults to "diff".
        final_time (float, optional): final time for the simulation. Defaults to 1e7.
        atol (float, optional): absolute tolerance for the simulation. Defaults to 1e-08.
        sample_thicnkess (float, optional): thickness of the sample in m. Defaults to 1e-03.
    """

    import h_transport_materials as htm

    # obtain material properties
    # inconel
    substrate_D = htm.diffusivities.filter(material="inconel_625")[0]
    substrate_D_0 = substrate_D.pre_exp.magnitude
    substrate_E_D = substrate_D.act_energy.magnitude

    substrate_recomb = htm.recombination_coeffs.filter(material="inconel_625")[1]
    substrate_Kr_0 = substrate_recomb.pre_exp.magnitude
    substrate_E_Kr = substrate_recomb.act_energy.magnitude

    substrate_diss = htm.dissociation_coeffs.filter(material="inconel_625")[1]
    substrate_Kd_0 = substrate_diss.pre_exp.magnitude
    substrate_E_Kd = substrate_diss.act_energy.magnitude

    substrate_S = htm.Solubility(
        S_0=(substrate_diss.pre_exp / substrate_recomb.pre_exp) ** 0.5,
        E_S=(0.5 * (substrate_diss.act_energy - substrate_recomb.act_energy)),
    )
    substrate_S_0 = substrate_S.pre_exp.magnitude
    substrate_E_S = substrate_S.act_energy.magnitude

    # build festim model
    model_standard = F.Simulation(log_level=40)

    # define mesh
    model_standard.mesh = F.MeshFromVertices(
        vertices=np.linspace(0, sample_thickness, num=500)
    )

    # define material
    substrate_standard = F.Material(
        id=1, D_0=substrate_D_0, E_D=substrate_E_D, S_0=substrate_S_0, E_S=substrate_E_S
    )
    model_standard.materials = F.Materials([substrate_standard])

    # define temperature
    model_standard.T = T

    # define boundary conditions
    if regime == "diff":
        model_standard.boundary_conditions = [
            F.SievertsBC(
                pressure=pressure,
                S_0=substrate_standard.S_0,
                E_S=substrate_standard.E_S,
                surfaces=[1],
            ),
            F.DirichletBC(value=0, surfaces=[2], field="solute"),
        ]
    elif regime == "surf":
        model_standard.boundary_conditions = [
            F.RecombinationFlux(
                Kr_0=substrate_Kr_0, E_Kr=substrate_E_Kr, order=2, surfaces=[1, 2]
            ),
            F.DissociationFlux(
                Kd_0=substrate_Kd_0, E_Kd=substrate_E_Kd, P=pressure, surfaces=[1]
            ),
        ]
    else:
        ValueError(f"permeation regime {regime} not recognised, should be diff or surf")

    # define settings
    model_standard.settings = F.Settings(
        absolute_tolerance=atol,
        relative_tolerance=1e-10,
        maximum_iterations=100,
        transient=True,
        final_time=final_time,
        linear_solver="mumps",
    )

    # define time step
    model_standard.dt = F.Stepsize(
        initial_value=0.1, stepsize_change_ratio=1.05, dt_min=1e-03
    )

    # define exports
    outflux = F.HydrogenFlux(surface=2)
    my_derived_quantites = F.DerivedQuantities(
        [outflux],
        filename=f"{foldername}/permeation_standard.csv",
        show_units=True,
    )
    model_standard.exports = F.Exports(
        [
            my_derived_quantites,
            # F.XDMFExport(
            #     field="solute", checkpoint=False, mode=1, folder=f"{foldername}/"
            # ),
            # F.TXTExport(
            #     field="solute",
            #     filename=f"{foldername}/mobile_conc_profile_standard.txt",
            # ),
        ]
    )

    model_standard.initialise()
    model_standard.run()


def pressure_from_flux(
    flux, t, T, sample_diameter, downstream_pipe_diameter, downstream_pipe_length
):
    """Evaulates the pressure in downstream given the flux, temperature and downstream properties

    Args:
        flux (np.array): flux of hydrogen in H/m^2/s
        t (np.array): time in s
        T (float): temperature in K
        sample_diameter (float): diameter of the sample in m
        downstream_pipe_diameter (float): diameter of the downstream pipe in m
        downstream_pipe_length (float): length of the downstream pipe in m
    """

    # calculate downstream volume
    downstream_volume = (
        downstream_pipe_length * np.pi * (downstream_pipe_diameter / 2) ** 2
    )

    integrated_flux = cumulative_trapezoid(flux, t, initial=0)
    A = np.pi * (sample_diameter / 2) ** 2  # m^2
    n = integrated_flux * A / (6.022 * 10**23)  # number of hydrogen atoms in mols

    R = 8.314  # J/mol/K

    # Calculate pressure
    P = n * R * T / downstream_volume  # Pa

    return P


def festim_model_case_700K_1e5Pa(sample_thickness):
    """Run a standard festim model for permeation of hydrogen through a sample at 700K and 1e5 Pa"""

    festim_model_standard(
        sample_thickness=sample_thickness,
        T=700,
        pressure=1e5,
        foldername="results/",
        regime="diff",
        atol=1e8,
        final_time=5000,
    )


def test_gauge_range_parameters(
    sample_diameter,
    sample_thickness,
    downstream_pipe_diameter,
    downstream_pipe_length,
    steady_state_point,
    sample_temperature_max_min,
    upstream_pressure_max_min,
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
        sample_temperature_max_min (list of floats): max and min pressure in Pa e.g. [1e2, 1e5]
        PRF_values (list of floats): PRF values to test e.g. [1, 10, 100, 1000]
        gauge_model (float): gauge model in Torr
    """

    # run festim simulations
    for P_up in upstream_pressure_max_min:
        for T in sample_temperature_max_min:
            print(f"Testing case P={P_up:.2e}, T={T:.0f}")
            festim_model_standard(
                T=T,
                pressure=P_up,
                foldername=f"results/parameter_exploration_alt/P={P_up:.2e}/T={T:.0f}",
                regime="diff",
                atol=1e08,
                final_time=1e7,
                sample_thickness=sample_thickness,
            )

    # data processing
    P_down_data = []
    for PRF_case in PRF_values:
        P_down_data_per_PRF = []
        for P_up in upstream_pressure_max_min:
            P_down_data_per_P_up = []
            for T in sample_temperature_max_min:
                run_data = np.genfromtxt(
                    f"results/parameter_exploration/P={P_up:.2e}/T={T:.0f}/permeation_standard.csv",
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
                    T=T,
                    sample_diameter=sample_diameter,
                    downstream_pipe_diameter=downstream_pipe_diameter,
                    downstream_pipe_length=downstream_pipe_length,
                )

                P_down_data_per_P_up.append(P[-1])

            P_down_data_per_PRF.append(P_down_data_per_P_up)

        P_down_data.append(P_down_data_per_PRF)

    # plotting

    norm = LogNorm(vmin=min(PRF_values), vmax=max(PRF_values))
    colorbar = cm.viridis
    colours = [colorbar(norm(i)) for i in PRF_values]

    plt.figure()

    n = 0
    for P_down, colour in zip(P_down_data, colours):
        n += 1

        x_values = [
            upstream_pressure_max_min[0],
            upstream_pressure_max_min[0],
            upstream_pressure_max_min[1],
            upstream_pressure_max_min[1],
        ]
        y_values = [
            P_down[0][1],
            P_down[0][0],
            P_down[1][0],
            P_down[1][1],
        ]

        plt.fill(x_values, y_values, alpha=0.7, color=colour)
        plt.annotate(
            f"PRF={PRF_values[colours.index(colour)]}",
            xy=(2e5, P_down[1][1] * 0.5),
            color=colour,
        )

        if n == 1:
            plt.scatter(x_values, y_values, color=colour)

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


if __name__ == "__main__":

    for P_up in test_pressure_values:
        for T in test_temperature_values:
            print(f"Testing case P={P_up:.2e}, T={T:.0f}")
            festim_model_standard(
                T=T,
                pressure=P_up,
                foldername=f"results/parameter_exploration/P={P_up:.2e}/T={T:.0f}",
                regime="diff",
                atol=1e08,
                final_time=1e7,
                sample_thickness=sample_thickness,
            )
