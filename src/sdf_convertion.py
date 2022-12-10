import logging
import numpy as np
import skimage.measure
import time
import torch
import pdb

import sys
sys.path.insert(1, "external/MeshSDF")
from lib.utils import *


def convert_sdf_samples_to_mesh(
    pytorch_3d_sdf_tensor,
    voxel_grid_origin,
    voxel_size,
    offset=None,
    scale=None,
):
    """
    Convert sdf samples to vertices,faces

    :param pytorch_3d_sdf_tensor: a torch.FloatTensor of shape (n,n,n)
    :voxel_grid_origin: a list of three floats: the bottom, left, down origin of the voxel grid
    :voxel_size: float, the size of the voxels

    This function is adapted from https://github.com/facebookresearch/DeepSDF
    """

    numpy_3d_sdf_tensor = pytorch_3d_sdf_tensor.numpy()

    verts, faces, normals, values = skimage.measure.marching_cubes(
        numpy_3d_sdf_tensor, level=0.0, spacing=voxel_size
    )

    # transform from voxel coordinates to camera coordinates
    # note x and y are flipped in the output of marching_cubes
    mesh_points = np.zeros_like(verts)
    mesh_points[:, 0] = voxel_grid_origin[0] + verts[:, 0]
    mesh_points[:, 1] = voxel_grid_origin[1] + verts[:, 1]
    mesh_points[:, 2] = voxel_grid_origin[2] + verts[:, 2]

    # apply additional offset and scale
    if scale is not None:
        mesh_points = mesh_points / scale
    if offset is not None:
        mesh_points = mesh_points - offset

    # return mesh
    return mesh_points, faces


def predict_sdf_grid(decoder, latent_vec, voxel_size, N=[64, 64, 64], max_batch=32 ** 3, offset=None, scale=None, voxel_origin = [-1., -.5, -.5]):
    start = time.time()

    decoder.eval()
    
    num_samples =  N[0] * N[1] * N[2]
    overall_index = torch.arange(0, num_samples, 1, out=torch.LongTensor())
    samples = torch.zeros(num_samples, 4)

    # transform first 3 columns
    # to be the x, y, z index
    samples[:, 2] = overall_index % N[2]
    samples[:, 1] = (overall_index.long() / N[2]) % N[1]
    samples[:, 0] = ((overall_index.long() / N[2]) / N[1]) % N[0]

    # transform first 3 columns
    # to be the x, y, z coordinate
    samples[:, 0] = (samples[:, 0] * voxel_size[0]) + voxel_origin[0]
    samples[:, 1] = (samples[:, 1] * voxel_size[1]) + voxel_origin[1]
    samples[:, 2] = (samples[:, 2] * voxel_size[2]) + voxel_origin[2]

    samples.requires_grad = False

    head = 0

    while head < num_samples:
        sample_subset = samples[head : min(head + max_batch, num_samples), 0:3].to(latent_vec.device)

        samples[head : min(head + max_batch, num_samples), 3] = (
            decode_sdf(decoder, latent_vec, sample_subset)
            .squeeze(1)
            .detach()
            .cpu()
        )
        head += max_batch

    sdf_values = samples[:, 3]
    sdf_values = sdf_values.reshape(*N)

    return sdf_values

def reconstruct_trimesh(decoder, latent_vec, N=[64, 32, 32], max_batch=32 ** 3, 
                        offset=None, scale=None, 
                        voxel_origin=[-1., -.5, -.5],
                        bbox_size=[2., 1., 1.]):
        
        N = np.array(N)
        # NOTE: the voxel_origin is actually the (bottom, left, down) corner, not the middle 
        voxel_size = np.array(bbox_size) / (N - 1)  
        sdf_values =\
            predict_sdf_grid(decoder, latent_vec, voxel_size, N, max_batch, offset, scale, voxel_origin)

        verts, faces = convert_sdf_samples_to_mesh(
            sdf_values.data.cpu(),
            voxel_origin,
            voxel_size,
            offset,
            scale,
        )

        return verts, faces

def reconstruct_voxels(decoder, latent_vec, N=[64, 32, 32], max_batch=32 ** 3, 
                        offset=None, scale=None, 
                        voxel_origin=[-1., -.5, -.5],
                        bbox_size=[2., 1., 1.]):
    
        N = np.array(N)
        voxel_size = np.array(bbox_size) / (N - 1)  
        sdf_values =\
            predict_sdf_grid(decoder, latent_vec, voxel_size, N, max_batch, offset, scale, voxel_origin)

        return (sdf_values < 0.).type(torch.FloatTensor)
