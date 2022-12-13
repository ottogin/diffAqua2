import os
import torch
import time
import json
import argparse

from src.sdf_convertion import reconstruct_voxels
from src.simulation import simulate

import sys
sys.path.insert(1, "external/MeshSDF")
from lib.models.decoder import DeepSDF
from chamferdist import ChamferDistance
from diffpd.mesh import MeshHex


DEVICE = "cuda:0"

def run_optimisation(save_path, start_latent_num=14, lr=1e-4, num_iters=100, dx=1./20, dx_sdf=1./32, do_target_shape_optimisation=False, N=[64, 32, 32]):
    print(f"Start optimisation {save_path} from shape #{start_latent_num} with lr={lr} and {num_iters} iterations", N)

    if not os.path.exists(save_path):
        os.mkdir(save_path)
    save_path = os.path.join(save_path, "summary.json")

    summary = {
        "params": {
            "start_latent_num": start_latent_num,
            "lr": lr,
            "num_iters": num_iters
        },
        "running_time": time.time(),
        "run": {
            "metrics": [],
            "times": [],
            "latents": [],
        }
    }

    # Load the model
    experiment_dir = "runs/wolfish_e256/"

    specs = json.load(open(os.path.join(experiment_dir, "specs.json")))
    train_mapping = json.load(open(specs["TrainSplit"]))
    data_mapping = json.load(open("/".join(specs["TrainSplit"].split("/")[:-1] + ["mapping.json"])))

    # Load the model
    decoder = torch.nn.DataParallel(DeepSDF(specs["CodeLength"],  **specs["NetworkSpecs"]), device_ids=[DEVICE])
    saved_model_state = torch.load(
        os.path.join(experiment_dir, "ModelParameters", "latest.pth"), map_location=DEVICE
    )
    decoder.load_state_dict(saved_model_state["model_state_dict"])
    decoder = decoder.eval()

    # Load latent codes
    orig_latents = torch.load(os.path.join(experiment_dir, "LatentCodes/latest.pth"), map_location=DEVICE)["latent_codes"]["weight"]

    latent = torch.clone(orig_latents[start_latent_num]).requires_grad_(True)
    optimizer = torch.optim.Adam([latent], lr=lr)

    start_time = time.time()

    if do_target_shape_optimisation:
        chamferDist = ChamferDistance()
        target_shape = reconstruct_voxels(decoder, torch.clone(orig_latents[41]), N=N)
        target_shape = MeshHex.load(target_shape.clone().detach().numpy(), dx=dx)
        target_shape = torch.as_tensor(target_shape.vertices).view(-1).clone().detach().to(torch.float32)
    
    for it in range(num_iters):
        optimizer.zero_grad()

        # Forward
        voxels = reconstruct_voxels(decoder, latent, N=N)
        if do_target_shape_optimisation:
            voxel_mesh = MeshHex.load(voxels.clone().detach().numpy(), dx=dx)
            voxel_mesh = torch.as_tensor(voxel_mesh.vertices).view(-1).clone().detach().to(torch.float32).requires_grad_(True)
            speed = -chamferDist(voxel_mesh.reshape(1, -1, 3), target_shape.reshape(1, -1, 3))\
                    -chamferDist(target_shape.reshape(1, -1, 3), voxel_mesh.reshape(1, -1, 3))
        else:
            speed, voxel_mesh = simulate(voxels)

        loss = speed
        loss.backward()

        # Backward through MeshSDF - don't forget to convert the 
        # Modify the dL/dx_i to be in the DeepSDF coordinate system
        dL_dx_i = dx / dx_sdf * voxel_mesh.grad.reshape(-1, 3).type(torch.FloatTensor)
        dL_dx_i = dL_dx_i.to(DEVICE)
        # use vertices to compute full backward pass
        optimizer.zero_grad()

        # Convert voxel_mesh to DeepSDF cooridnate system !!!!!!!!
        voxel_mesh = voxel_mesh.reshape(-1, 3) * dx_sdf / dx + torch.tensor([[-1, -0.5, -0.5]], dtype=voxel_mesh.dtype, device=voxel_mesh.device)
        xyz = voxel_mesh.clone().detach().type(torch.FloatTensor)
        xyz = xyz.to(DEVICE).requires_grad_(True)
        latent_inputs = latent.expand(xyz.shape[0], -1)

        #first compute normals
        pred_sdf = decoder(latent_inputs, xyz)
        # for df, grad in zip(pred_sdf, dL_dx_i):
        #     print(df.item(), grad.item())

        loss_normals = torch.sum(pred_sdf)
        loss_normals.backward(retain_graph = True)
        # normalization to take into account for the fact sdf is not perfect...

        normals = xyz.grad/torch.norm(xyz.grad, 2, 1).unsqueeze(-1)
        # now assemble inflow derivative
        optimizer.zero_grad()

        # Drop points inside the mesh - gradients there are not reliable
        filt = (pred_sdf[:, 0] <= dx_sdf / 0.00000001) & (pred_sdf[:, 0] >= -dx_sdf / 0.000000001)
        dL_ds_i = -torch.matmul(dL_dx_i[filt].unsqueeze(1), normals[filt].unsqueeze(-1)).squeeze(-1)
        # refer to Equation (4) in the main paper
        loss_backward = torch.sum(dL_ds_i * pred_sdf[filt])
        loss_backward.backward()
        # and update params
        optimizer.step()

        time_elapsed = time.time() - start_time

        summary["run"]["times"].append(time_elapsed)
        summary["run"]["metrics"].append(speed.item())
        summary["run"]["latents"].append(latent.cpu().detach().numpy().tolist())
        print(f"#{it:4d} : Fish Speed: {speed:.5f}       Time Elapsed: {time_elapsed:.2f}", end='\r')
        
        with(open(save_path, "w")) as f:
            json.dump(summary, f)

    return summary


if __name__ == "__main__":    
    base_dir = "optimisation_runs"
    
    parser = argparse.ArgumentParser(prog = 'Fish Optimiser', description = 'Optimise fish perfomance with DeepSDF')
    parser.add_argument('name', action='store', type=str, help='The text to parse.')
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--nits', default=100, type=int)
    parser.add_argument('--startshape', default=14, type=int)

    args = parser.parse_args()

    run_optimisation(os.path.join(base_dir, args.name), start_latent_num=args.startshape, lr=args.lr, num_iters=args.nits)