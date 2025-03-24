import festim as F
import numpy as np


def deg_to_kelvin(deg: float) -> float:
    """converts temperature value from degrees C to kelvin (K)

    Args:
        deg: value of temperature in degrees C

    Returns:
        float: value of temperature in kelvin (K)
    """
    return deg + 273.15


def bar_to_pa(bar: float) -> float:
    """converts pressure value from bar to pascals (Pa)

    Args:
        bar: value of pressure in bar

    Returns:
        float: value of pressure in pascals (Pa)
    """
    return bar * 1e5


def festim_sim(
    wall_thickness, pipe_pressure, temperature, results_foldername="results"
):
    """Run a 1-d festim simulation of hydrogen permeation in the wall of a pipe

    Args:
        wall_thickness (float): Wall thickness of the pipe in meters
        pipe_pressure (float): Pressure inside the pipe in Pa
    """

    my_model = F.Simulation()
    pipe_outer_radius = 323.9e-03 / 2
    wall_thickness = 14.3e-03
    my_model.mesh = F.MeshFromVertices(
        vertices=np.linspace(
            pipe_outer_radius - wall_thickness, pipe_outer_radius, 10000
        ),
        type="cylindrical",
    )
    my_model.materials = F.Materials(
        [F.Material(D_0=5.97e-07, E_D=0.52, S_0=2.85e23, E_S=0.14, id=1)]
    )

    my_model.boundary_conditions = [
        F.RecombinationFlux(Kr_0=4.85e-27, E_Kr=0.53, order=2, surfaces=[1, 2]),
        F.DissociationFlux(Kd_0=8.61e18, E_Kd=0.66, P=pipe_pressure, surfaces=[1]),
    ]

    my_model.T = temperature

    month = 3600 * 24 * 30
    year = 3600 * 24 * 365
    five_years = 5 * year
    ten_years = 10 * year
    twenty_years = 20 * year
    forty_years = 40 * year
    my_model.settings = F.Settings(
        transient=True,
        absolute_tolerance=1e10,
        relative_tolerance=1e-10,
        final_time=year * 40,
    )
    my_model.dt = F.Stepsize(
        initial_value=1,
        stepsize_change_ratio=1.01,
        milestones=[month, year, five_years, ten_years, twenty_years, forty_years],
    )

    my_derived_quantities = F.DerivedQuantities(
        [
            # F.HydrogenFlux(surface=2),
            # F.TotalVolume(field="solute", volume=1),
            F.TotalVolumeCylindrical(field="solute", volume=1),
            F.SurfaceFluxCylindrical(field="solute", surface=2),
        ],
        filename=f"{results_foldername}/derived_quantities.csv",
        nb_iterations_between_exports=1,
        show_units=True,
    )
    my_model.exports = F.Exports(
        [
            my_derived_quantities,
            # F.XDMFExport(field="solute", mode=1, filename=f"{results_foldername}/mobile_concentration.xdmf", checkpoint=False),
            F.TXTExport(
                field="solute",
                filename=f"{results_foldername}/mobile_concentration_field.txt",
                times=my_model.dt.milestones,
            ),
        ]
    )

    my_model.initialise()
    my_model.run()


# temperature values from 10 to 60 degrees C
temperature_values = np.linspace(10, 60, 5)
temperature_values = deg_to_kelvin(temperature_values)

# pressure values from 35 to 90 bar
pressure_values = np.linspace(35, 90, 5)
pressure_values = bar_to_pa(pressure_values)

if __name__ == "__main__":
    pressure_max = bar_to_pa(90)
    T_max = deg_to_kelvin(60)
    foldername = f"results/item_1_1a/{T_max:.1f}K/{pressure_max:.2e}Pa"
    festim_sim(
        wall_thickness=14.3e-03,
        pipe_pressure=pressure_max,
        temperature=T_max,
        results_foldername=foldername,
    )

    # item 1/1A
    # wall_thickness = 14.3e-03 #m
    # # for temperature in temperature_values:
    # temperature = temperature_values[-1]
    # for pipe_pressure in pressure_values:
    #     foldername = f"results/item_1_1a/{temperature:.1f}K/{pipe_pressure:.2e}Pa"
    #     festim_sim(wall_thickness, pipe_pressure, temperature, results_foldername=foldername)

    # # item 1B/1C
    # wall_thickness = 15.9e-03 #m
    # for temperature in temperature_values:
    #     for pipe_pressure in pressure_values:
    #         foldername = f"results/item_1b_1c/{temperature:.1f}K/{pipe_pressure:.2e}Pa"
    #         festim_sim(wall_thickness, pipe_pressure, temperature, results_foldername=foldername)
