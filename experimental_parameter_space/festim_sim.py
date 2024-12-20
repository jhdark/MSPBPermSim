import festim as F
import numpy as np
from scipy.integrate import cumulative_trapezoid


def festim_model_standard(
    T, pressure, foldername, regime="diff", final_time=1e7, atol=1e-08
):

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
    model_standard.mesh = F.MeshFromVertices(vertices=np.linspace(0, 1e-03, num=500))

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


if __name__ == "__main__":
    festim_model_standard(
        T=700,
        pressure=1e5,
        foldername="results/",
        regime="diff",
        atol=1e8,
    )
