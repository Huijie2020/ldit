# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
A minimal training script for DiT using PyTorch DDP.
"""
import torch
# the first flag below was False when we tested this script but True makes A100 training a lot faster:
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision.datasets import ImageFolder
from torchvision import transforms
import numpy as np
from collections import OrderedDict
from PIL import Image
from copy import deepcopy
from glob import glob
from time import time
import argparse
import logging
import os

from models import DiT_models
from diffusion import create_diffusion
from diffusers.models import AutoencoderKL

import utils.render
from utils.lidar import LiDARUtility, get_hdl64e_linear_ray_angles
from diffusion import encoding
from torch import nn
import torch.nn.functional as F
import datasets as ds
import einops
# from torch.utils.tensorboard import SummaryWriter
# from tensorboardX import SummaryWriter
################################################################################
#                             Training Helper Functions                         #
#################################################################################

@torch.no_grad()
def update_ema(ema_model, model, decay=0.9999):
    """
    Step the EMA model towards the current model.
    """
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())

    for name, param in model_params.items():
        # TODO: Consider applying only to params that require_grad to avoid small numerical changes of pos_embed
        ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)


def requires_grad(model, flag=True):
    """
    Set requires_grad flag for all parameters in a model.
    """
    for p in model.parameters():
        p.requires_grad = flag


def cleanup():
    """
    End DDP training.
    """
    dist.destroy_process_group()


def create_logger(logging_dir):
    """
    Create a logger that writes to a log file and stdout.
    """
    if dist.get_rank() == 0:  # real logger
        logging.basicConfig(
            level=logging.INFO,
            format='[\033[34m%(asctime)s\033[0m] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S',
            handlers=[logging.StreamHandler(), logging.FileHandler(f"{logging_dir}/log.txt")]
        )
        logger = logging.getLogger(__name__)
    else:  # dummy logger (does nothing)
        logger = logging.getLogger(__name__)
        logger.addHandler(logging.NullHandler())
    return logger


# def center_crop_arr(pil_image, image_size):
#     """
#     Center cropping implementation from ADM.
#     https://github.com/openai/guided-diffusion/blob/8fb3ad9197f16bbc40620447b2742e13458d2831/guided_diffusion/image_datasets.py#L126
#     """
#     while min(*pil_image.size) >= 2 * image_size:
#         pil_image = pil_image.resize(
#             tuple(x // 2 for x in pil_image.size), resample=Image.BOX
#         )
#
#     scale = image_size / min(*pil_image.size)
#     pil_image = pil_image.resize(
#         tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
#     )
#
#     arr = np.array(pil_image)
#     crop_y = (arr.shape[0] - image_size) // 2
#     crop_x = (arr.shape[1] - image_size) // 2
#     return Image.fromarray(arr[crop_y: crop_y + image_size, crop_x: crop_x + image_size])


#################################################################################
#                                  Training Loop                                #
#################################################################################

def main(args):
    """
    Trains a new DiT model.
    """
    assert torch.cuda.is_available(), "Training currently requires at least one GPU."

    # Setup DDP:
    dist.init_process_group("nccl")
    assert args.global_batch_size % dist.get_world_size() == 0, f"Batch size must be divisible by world size."
    rank = dist.get_rank()
    device = rank % torch.cuda.device_count()
    seed = args.global_seed * dist.get_world_size() + rank
    torch.manual_seed(seed)
    torch.cuda.set_device(device)
    print(f"Starting rank={rank}, seed={seed}, world_size={dist.get_world_size()}.")

    # Setup an experiment folder:
    if rank == 0:
        os.makedirs(args.results_dir, exist_ok=True)  # Make results folder (holds all experiment subfolders)
        experiment_index = len(glob(f"{args.results_dir}/*"))
        model_string_name = args.model.replace("/", "-")  # e.g., DiT-XL/2 --> DiT-XL-2 (for naming folders)
        experiment_dir = f"{args.results_dir}/{experiment_index:03d}-{model_string_name}"  # Create an experiment folder
        checkpoint_dir = f"{experiment_dir}/checkpoints"  # Stores saved model checkpoints
        os.makedirs(checkpoint_dir, exist_ok=True)
        tensorboard_dir = f"{experiment_dir}/tensorboard"  # Stores saved model checkpoints
        os.makedirs(tensorboard_dir, exist_ok=True)
        logger = create_logger(experiment_dir)
        logger.info(f"Experiment directory created at {experiment_dir}")
        # tb_writer = SummaryWriter(tensorboard_dir)
    else:
        logger = create_logger(None)

    channels = [
        1 if args.train_depth else 0,
        1 if args.train_reflectance else 0,
    ]

    in_channels = sum(channels)

    # set lidar projection
    if "spherical" in args.lidar_projection:
        print("set HDL-64E linear ray angles")
        coords = get_hdl64e_linear_ray_angles(*args.image_size)
    elif "unfolding" in args.lidar_projection:
        print("set dataset ray angles")
        _coords = torch.load(f"data/{args.dataset}/unfolding_angles.pth")
        coords = F.interpolate(_coords, size=args.image_size, mode="nearest-exact")
    else:
        raise ValueError(f"Unknown: {args.lidar_projection}")

    # spatial coords embedding
    coords_embedding = None
    if args.model_coords_embedding == "spherical_harmonics":
        coords_embedding = encoding.SphericalHarmonics(levels=5)
        in_channels += coords_embedding.extra_ch
    elif args.model_coords_embedding == "polar_coordinates":
        coords_embedding = nn.Identity()
        in_channels += coords.shape[1]
    elif args.model_coords_embedding == "fourier_features":
        coords_embedding = encoding.FourierFeatures(args.image_size)
        in_channels += coords_embedding.extra_ch

    # Utility
    lidar_utils = LiDARUtility(
        resolution=args.image_size,
        image_format=args.image_format,
        min_depth=args.min_depth,
        max_depth=args.max_depth,
        ray_angles=coords,
    )
    lidar_utils.to(device)

    def preprocess(batch):
        x = []
        if args.train_depth:
            x += [lidar_utils.convert_depth(batch["depth"])]
        if args.train_reflectance:
            x += [batch["reflectance"]]
        x = torch.cat(x, dim=1)
        x = lidar_utils.normalize(x)
        x = F.interpolate(
            x.to(device),
            size=args.image_size,
            mode="nearest-exact",
        )
        return x

    def split_channels(image: torch.Tensor):
        depth, rflct = torch.split(image, channels, dim=1)
        return depth, rflct

    # @torch.inference_mode()
    # def log_images(image, tag: str = "name", global_step: int = 0):
    #     image = lidar_utils.denormalize(image)
    #     out = dict()
    #     depth, rflct = split_channels(image)
    #     if depth.numel() > 0:
    #         out[f"{tag}/depth"] = utils.render.colorize(depth)
    #         metric = lidar_utils.revert_depth(depth)
    #         mask = (metric > lidar_utils.min_depth) & (metric < lidar_utils.max_depth)
    #         out[f"{tag}/depth/orig"] = utils.render.colorize(
    #             metric / lidar_utils.max_depth
    #         )
    #         xyz = lidar_utils.to_xyz(metric) / lidar_utils.max_depth * mask
    #         normal = -utils.render.estimate_surface_normal(xyz)
    #         normal = lidar_utils.denormalize(normal)
    #         bev = utils.render.render_point_clouds(
    #             points=einops.rearrange(xyz, "B C H W -> B (H W) C"),
    #             colors=einops.rearrange(normal, "B C H W -> B (H W) C"),
    #             t=torch.tensor([0, 0, 1.0]).to(xyz),
    #         )
    #         out[f"{tag}/bev"] = bev.mul(255).clamp(0, 255).byte()
    #     if rflct.numel() > 0:
    #         out[f"{tag}/reflectance"] = utils.render.colorize(rflct, cm.plasma)
    #     if mask.numel() > 0:
    #         out[f"{tag}/mask"] = utils.render.colorize(mask, cm.binary_r)
    #     tracker.log_images(out, step=global_step)

    # Create model:
    assert args.image_size[0] % 8 == 0 and args.image_size[1] % 8 == 0, "Image size must be divisible by 8 (for the VAE encoder)."
    latent_size = (args.image_size[0] // 8, args.image_size[1] // 8)
    model = DiT_models[args.model](
        input_size=latent_size,
        num_classes=args.num_classes
    )
    # Note that parameter initialization is done within the DiT constructor
    ema = deepcopy(model).to(device)  # Create an EMA of the model for use after training
    requires_grad(ema, False)
    model = DDP(model.to(device), device_ids=[rank])
    diffusion = create_diffusion(timestep_respacing="")  # default: 1000 steps, linear noise schedule
    vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-{args.vae}").to(device)
    logger.info(f"DiT Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Setup optimizer (we used default Adam betas=(0.9, 0.999) and a constant learning rate of 1e-4 in our paper):
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0)

    # Setup data:
    # transform = transforms.Compose([
    #     transforms.Lambda(lambda pil_image: center_crop_arr(pil_image, args.image_size)),
    #     transforms.RandomHorizontalFlip(),
    #     transforms.ToTensor(),
    #     transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
    # ])
    # dataset = ImageFolder(args.data_path, transform=transform)
    dataset = ds.load_dataset(
        path=f"data/{args.dataset}",
        name=args.lidar_projection,
        split=ds.Split.TRAIN,
        num_proc=args.num_workers,
        trust_remote_code=True,
    ).with_format("torch")
    sampler = DistributedSampler(
        dataset,
        num_replicas=dist.get_world_size(),
        rank=rank,
        shuffle=True,
        seed=args.global_seed
    )
    loader = DataLoader(
        dataset,
        batch_size=int(args.global_batch_size // dist.get_world_size()),
        shuffle=False,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True
    )
    # logger.info(f"Dataset contains {len(dataset):,} images ({args.data_path})")

    # Prepare models for training:
    update_ema(ema, model.module, decay=0)  # Ensure EMA is initialized with synced weights
    model.train()  # important! This enables embedding dropout for classifier-free guidance
    ema.eval()  # EMA model should always be in eval mode

    # Variables for monitoring/logging purposes:
    train_steps = 0
    log_steps = 0
    running_loss = 0
    start_time = time()

    logger.info(f"Training for {args.epochs} epochs...")
    for epoch in range(args.epochs):
        sampler.set_epoch(epoch)
        logger.info(f"Beginning epoch {epoch}...")
        for x in loader:
            x = preprocess(x)
            y = torch.zeros(x.shape[0]).to(torch.int64)
            x = x.to(device)
            y = y.to(device)

            # spatial embedding
            cemb = encoding.positional_encoding_polar_1channel(coords)
            cemb = cemb.repeat_interleave(x.shape[0], dim=0) # [6, 1, 64, 1024]
            cemb = cemb.to(device)
            x = torch.cat([x, cemb], dim=1) # [6, 3, 64, 1024]

            with torch.no_grad():
                # Map input images to latent space + normalize latents:
                x = vae.encode(x).latent_dist.sample().mul_(0.18215)
            t = torch.randint(0, diffusion.num_timesteps, (x.shape[0],), device=device)
            model_kwargs = dict(y=y)
            loss_dict = diffusion.training_losses(model, x, t, model_kwargs)
            loss = loss_dict["loss"].mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            update_ema(ema, model.module)

            # Log loss values:
            running_loss += loss.item()
            log_steps += 1
            train_steps += 1
            # tb_writer.add_scalar('Train', loss.item(), train_steps)
            if train_steps % args.log_every == 0:
                # Measure training speed:
                torch.cuda.synchronize()
                end_time = time()
                steps_per_sec = log_steps / (end_time - start_time)
                # Reduce loss history over all processes:
                avg_loss = torch.tensor(running_loss / log_steps, device=device)
                dist.all_reduce(avg_loss, op=dist.ReduceOp.SUM)
                avg_loss = avg_loss.item() / dist.get_world_size()
                logger.info(f"(step={train_steps:07d}) Train Loss: {avg_loss:.4f}, Train Steps/Sec: {steps_per_sec:.2f}")
                # Reset monitoring variables:
                running_loss = 0
                log_steps = 0
                start_time = time()

            # Save DiT checkpoint:
            if train_steps % args.ckpt_every == 0 and train_steps > 0:
                if rank == 0:
                    checkpoint = {
                        "model": model.module.state_dict(),
                        "ema": ema.state_dict(),
                        "opt": opt.state_dict(),
                        "args": args
                    }
                    checkpoint_path = f"{checkpoint_dir}/{train_steps:07d}.pt"
                    torch.save(checkpoint, checkpoint_path)
                    logger.info(f"Saved checkpoint to {checkpoint_path}")
                dist.barrier()

    model.eval()  # important! This disables randomized embedding dropout
    # do any sampling/FID calculation/etc. with ema (or model) in eval mode ...

    logger.info("Done!")
    cleanup()


if __name__ == "__main__":
    # Default args here will train DiT-XL/2 with the hyperparameters we used in our paper (except training iters).
    parser = argparse.ArgumentParser()
    # parser.add_argument("--data-path", type=str, required=True, default="./data/kitti_360")
    parser.add_argument("--dataset", type=str, default='kitti_360')
    parser.add_argument("--results-dir", type=str, default="results")
    parser.add_argument("--model", type=str, choices=list(DiT_models.keys()), default="DiT-XL/2")
    parser.add_argument("--image-format", type=str, default="log_depth")
    parser.add_argument("--image-size", type=tuple, default=(64, 1024))
    parser.add_argument("--lidar-projection", type=str, choices=["unfolding-2048", "spherical-2048", "unfolding-1024", "spherical-1024"], default="spherical-1024")
    parser.add_argument("--model-coords-embedding", type=str, choices=["spherical_harmonics", "polar_coordinates", "fourier_features"], default="polar_coordinates")
    parser.add_argument("--train-depth", type=bool, default=True)
    parser.add_argument("--train-reflectance", type=bool, default=True)
    parser.add_argument("--train-mask", type=bool, default=False)
    parser.add_argument("--min-depth", type=float, default=1.45)
    parser.add_argument("--max-depth", type=float, default=80.0)
    parser.add_argument("--num-classes", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--global-batch-size", type=int, default=100)
    parser.add_argument("--global-seed", type=int, default=0)
    parser.add_argument("--vae", type=str, choices=["ema", "mse"], default="ema")  # Choice doesn't affect training
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--ckpt-every", type=int, default=50_000)
    args = parser.parse_args()
    main(args)
