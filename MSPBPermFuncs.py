import festim as F
import fenics as f
import numpy as np
import h_transport_materials as htm
import matplotlib.pyplot as plt

def AssignMaterials(my_model, barrier_diffusivity, barrier_solubility, substrate_diffusivity, substrate_solubility):
    """
    takes in the model and htm properties groups and applies them to the model
    """

    # Assigning Materials
    D_0 = barrier_diffusivity.pre_exp.magnitude
    E_D = barrier_diffusivity.act_energy.magnitude

    S_0 = barrier_solubility.pre_exp.magnitude
    E_S = barrier_solubility.act_energy.magnitude

    # source: Zhou 2024
    barrier = F.Material(
        id=1,
        D_0=D_0, E_D=E_D,
        S_0=S_0, E_S=E_S
    )

    D_0 = substrate_diffusivity.pre_exp.magnitude
    E_D = substrate_diffusivity.act_energy.magnitude

    S_0 = substrate_solubility.pre_exp.magnitude
    E_S = substrate_solubility.act_energy.magnitude

    # D_0, E_d source: "Reiter 1996"
    steel_mat = F.Material(
        id=2,
        D_0=D_0, E_D=E_D,
        S_0=S_0, E_S=E_S
    )
    my_model.materials = F.Materials([barrier, steel_mat])

def MeshModel(my_model, t_barrier, t_substrate):
    # Meshing model
    vertices_left = np.linspace(0, t_barrier, num=3)
    vertices_mid = np.linspace(t_barrier,
        t_barrier + t_substrate, num=10)
    vertices = np.concatenate([vertices_left, vertices_mid])

    my_model.mesh = F.MeshFromVertices(vertices)

def ApplyBCs(my_model, T, P_up, barrier_solubility):
    # Temperature of substrate
    my_model.T = T

    left_bc = F.SievertsBC(
        surfaces=1,
        S_0 = barrier_solubility.pre_exp.magnitude,
        E_S = barrier_solubility.act_energy.magnitude,
        pressure=P_up
    )

    right_bc = F.DirichletBC(
        field="solute",
        surfaces=2,
        value=0
    )

    my_model.boundary_conditions = [left_bc, right_bc]

def RunModel(my_model, final_time, folder):
    my_model.settings = F.Settings(
        absolute_tolerance=1e6,
        relative_tolerance=1e-10,
        final_time=final_time  # s
        )

    my_model.dt = F.Stepsize(initial_value=100,
                            stepsize_change_ratio = 1.1)

    
    derived_quantities = F.DerivedQuantities([F.HydrogenFlux(surface=2)], show_units=True)
    txt_export = F.TXTExport(
        field='solute',
        filename=folder + '/mobile.txt',
    )

    xdmf_export = F.XDMFExport(field="solute", folder=folder, checkpoint=False)

    my_model.exports = [derived_quantities, txt_export, xdmf_export]
    my_model.initialise()

    my_model.run()

    return derived_quantities

