from cores.dataset import GaussianDataset
from cores.unet import Unet1D
from cores.model import GaussianDiffusion1D
from cores.trainer import Trainer1D
import torch
import argparse


def main(dataset_dir: str, dim: int, seq_length: int,
         train_batch_size, gradient_accumulate_every: int,
         train_lr: float, train_num_steps: int,
         results_folder: str):
    dataset = GaussianDataset(dataset_dir)
    unet1d = Unet1D(dim, channels=14)
    gaussian_diffusion = GaussianDiffusion1D(unet1d, seq_length=seq_length)
    trainer = Trainer1D(
        gaussian_diffusion,
        dataset,
        train_batch_size=train_batch_size,
        gradient_accumulate_every=gradient_accumulate_every,
        train_lr=train_lr,
        train_num_steps=train_num_steps,
        results_folder=results_folder,
    )

    trainer.train()

    pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_dir', type=str, default='./gs_dataset')
    parser.add_argument('--dim', type=int, default=32)
    parser.add_argument('--seq_length', type=int, default=2 ** 16)
    parser.add_argument('--train_batch_size', type=int, default=1)
    parser.add_argument('--gradient_accumulate_every', type=int, default=1)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--num_steps', type=int, default=100000)
    parser.add_argument('--results_folder', type=str, default='./results')
    args = parser.parse_args()
    main(
        args.dataset_dir,
        dim=args.dim,
        train_batch_size=args.train_batch_size,
        gradient_accumulate_every=args.gradient_accumulate_every,
        train_lr=args.lr,
        seq_length=args.seq_length,
        train_num_steps=args.num_steps,
        results_folder=args.results_folder
    )
