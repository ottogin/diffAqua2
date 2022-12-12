from pathlib import Path
import sys
import time
import math
import random
import copy
from collections import deque
from tqdm import trange, tqdm

import scipy
import scipy.optimize
import numpy as np

import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

import trimesh
import pyvista as pv
import imageio

from diffpd.fem import DeformableHex, HydrodynamicsStateForceHex
from diffpd.sim import Sim
from diffpd.nn import OpenFoldController
from diffpd import transforms
from diffpd.mesh import MeshHex

seed = 42
# pv.start_xvfb()

# Mesh parameters.
length = 20
dx = 1. / length

# Hydrodynamics parameters.
rho = 1e1
v_water = [0, 0, 0]   # Velocity of the water.
# Cd_points = (angle, coeff) pairs where angle is normalized to [0, 1].
Cd_points = np.array([
    [0.0, 0.05],
    [0.4, 0.05],
    [0.7, 1.85],
    [1.0, 2.05],
]) # * 1.0
# Ct_points = (angle, coeff) pairs where angle is normalized to [-1, 1].
Ct_points = np.array([
    [-1, -0.8],
    [-0.3, -0.5],
    [0.3, 0.1],
    [1, 2.5],
])

# FEM parameters.
youngs_modulus = 1e6
poissons_ratio = 0.45
average_density = 1e1
dt = 3.33e-3
method = 'pd_eigen'
options = {
    'max_pd_iter': 2000, 'max_ls_iter': 10, 'abs_tol': 1e-4, 'rel_tol': 1e-3,
    'verbose': 0, 'thread_ct': 64, 'use_bfgs': 1, 'bfgs_history_size': 10
}


def simulate(voxels, num_frames=20):
    ## Inint simulator

    shape = voxels.shape
    rest_mesh = MeshHex.load(voxels.clone().detach().numpy(), dx=dx)
    transform = []
    transform.append(transforms.AddStateForce(
        'hydrodynamics', [rho] + v_water + Cd_points.ravel().tolist() + Ct_points.ravel().tolist() + rest_mesh.boundary.ravel().tolist()))

    actuator_scale = 0.04
    actuator_height = int(shape[2] * actuator_scale)
    actuator_width =  int(shape[1] * actuator_scale)

    all_muscles = []
    shared_muscles = []
    for z in range(int(shape[2] / 2) - actuator_height, int(shape[2] / 2) + actuator_height):
        muscle_pair = []
        for y in range(int(shape[1] / 2) - actuator_width, int(shape[1] / 2) + actuator_width):
            indices = rest_mesh.cell_indices[int(0.45 * shape[0]):int(0.5 * shape[0]), y, z].tolist()
            transform.append(transforms.AddActuationEnergy(1e6, [1.0, 0.0, 0.0], indices))
            muscle_pair.append(indices)

            # print(indices)
        shared_muscles.append(muscle_pair)
    all_muscles.append(shared_muscles)

    transform = transforms.Compose(transform)

    deformable = DeformableHex(
        rest_mesh, density=average_density, dt=dt, method=method, options=options,)
    deformable = transform(deformable)

    dofs = deformable.dofs()
    act_dofs = deformable.act_dofs()

    q0 = torch.as_tensor(rest_mesh.vertices).view(-1).clone().detach().to(torch.float64)
    v0 = torch.zeros(dofs).detach().to(torch.float64)

    head_indices = rest_mesh.node_indices[0, 0, 0]

    sim = Sim(deformable)
    sim.add_default_pd_energies(['corotated'], youngs_modulus, poissons_ratio)

    voxel_mesh = q0.clone()
    voxel_mesh.requires_grad = True
    voxel_mesh = voxel_mesh.to(torch.float64)

    controller = OpenFoldController(
        deformable, all_muscles,
        num_steps=num_frames,
        segment_len=1,
        init_period=16.0,
        init_magnitude=128.0).to(torch.float64)

    a = None
    q, v = q0, v0

    target_dir = torch.Tensor([-1.0, 0.0, 0.0]).to(torch.float64)
    forward_loss = 0.0
    
    qs, vs = [], []
    for a in controller():
        q, v = sim(q, v, a, shape=voxel_mesh)

        v_zero = torch.zeros_like(v).view(-1, 3)
        v_zero[:, :-1] = v.view(-1, 3)[:, :-1]
        v_zero = v_zero.view(-1)

        v_center = v_zero.view(-1, 3).mean(dim=0)

        dot = torch.dot(v_center, target_dir)
        forward_loss += -dot

        qs.append(q)
        vs.append(v)

    # for idx in range(20):
    #     print(torch.mean(vs[idx].reshape(-1, 3)[:, 0]).item(),
    #           torch.mean(vs[idx].reshape(-1, 3)[:, 1]).item(),
    #           torch.mean(vs[idx].reshape(-1, 3)[:, 2]).item()
    #         )
    # speed = torch.mean((qs[-1] - q0).reshape(-1, 3)[:, 0])#torch.mean(torch.concatenate(vs).reshape(-1, 3))
    
    return forward_loss, voxel_mesh
        
