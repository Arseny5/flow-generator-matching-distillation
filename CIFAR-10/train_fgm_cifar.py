#!/usr/bin/env python
import os
import sys
import random
import copy
import torch
import wandb  # For logging with Weights & Biases
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from tqdm import tqdm, trange
import matplotlib.pyplot as plt
import gc
import torchdiffeq
from torchdiffeq import odeint
from torchvision.utils import make_grid, save_image
from torchvision.transforms import ToPILImage
from torchvision import datasets, transforms
from torchcfm.models.unet.unet import UNetModelWrapper

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def init_wandb():
    wandb.init(
        project="Flow-Generator-Matching",
        name="FGM-distillation-run",
        config={
            "lr_online": 2e-5,
            "lr_generator": 2e-5,
            "batch_size": 10,
            "epochs": 20000
        }
    )

def warmup_lr(step, warmup=5000):
    return min(step, warmup) / warmup

class HiddenPrints:
    """Класс для скрытия вывода (опционально)"""
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')
    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout

def load_data(batch_size=128, num_workers=4, dataset_name="CIFAR10"):
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
    ])
    if dataset_name == "MNIST":
        dataset = datasets.MNIST(root='./data', train=True, download=True, transform=ToTensor())
    elif dataset_name == "CIFAR10":
        dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    else:
        raise ValueError("Unsupported dataset. Please choose 'MNIST' or 'CIFAR10'.")
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=True)
    return data_loader

# Model loading: Teacher, Student, Generator
def load_models():
    checkpoint = torch.load("pretrain_flow_matching/checkpoints/cfm_cifar10_weights_step_400000.pt", map_location=device)
    state_dict_ema = checkpoint["ema_model"]

    ema_model = UNetModelWrapper(
        dim=(3, 32, 32),
        num_res_blocks=2,
        num_channels=128,
        channel_mult=[1, 2, 2, 2],
        num_heads=4,
        num_head_channels=64,
        attention_resolutions="16",
        dropout=0.1,
    ).to(device)
    ema_model.load_state_dict(state_dict_ema)
    Pretrained_v_t = copy.deepcopy(ema_model).eval().to(device)
    for p in Pretrained_v_t.parameters():
        p.requires_grad = False

    Online_v_t = UNetModelWrapper(
        dim=(3, 32, 32),
        num_res_blocks=2,
        num_channels=128,
        channel_mult=[1, 2, 2, 2],
        num_heads=4,
        num_head_channels=64,
        attention_resolutions="16",
        dropout=0.1,
    ).to(device)
    Online_v_t.load_state_dict(Pretrained_v_t.state_dict())

    Generator = UNetModelWrapper(
        dim=(3, 32, 32),
        num_res_blocks=1,
        num_channels=64,
        channel_mult=[1, 2, 2],
        num_heads=4,
        num_head_channels=32,
        attention_resolutions="16",
        dropout=0.0,
    ).to(device)

    model_size_gen = sum(p.numel() for p in Generator.parameters())
    print("Generator model params: %.2f M" % (model_size_gen / 1024 / 1024))
    model_size_pre = sum(p.numel() for p in Pretrained_v_t.parameters())
    print("Original FM model params: %.2f M" % (model_size_pre / 1024 / 1024))
    model_size_online = sum(p.numel() for p in Online_v_t.parameters())
    print("Online FM model params: %.2f M" % (model_size_online / 1024 / 1024))

    return Pretrained_v_t, Online_v_t, Generator

class UnconditionalFlowLoss:
    def __init__(self, sigma_min: float = 0.001):
        super().__init__()
        self.sigma_min = sigma_min
        self.epsilon = 1e-5
        self.device = device

    def psi_t(self, x: torch.Tensor, x_1: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        while t.dim() < x.dim():
            t = t[..., None]
        return (1 - (1 - self.sigma_min) * t) * x + t * x_1

    def loss_flow_matching(self, Online_v_t: nn.Module, Pretrained_v_t: nn.Module, Generator: nn.Module, batch_size: int = 10) -> torch.Tensor:
        # Генерируем t для всего батча (смешанное значение)
        t = (torch.rand(1, device=self.device) + torch.arange(10 * batch_size, device=self.device) / (10 * batch_size)) % (1 - self.epsilon)
        t = t[:, None]
        t_steps = torch.linspace(0, 1, 2, device=self.device)
        z = torch.randn(10 * batch_size, 3, 32, 32, device=self.device)

        with torch.no_grad():
            traj = torchdiffeq.odeint(
                lambda _t, _x: Generator(_t, _x),
                z,
                t_steps,
                atol=1e-4,
                rtol=1e-4,
                method="euler",
            )
        x_0 = traj[-1]
        psi_t_val = self.psi_t(z, x_0, t)
        d_psi = x_0 - (1 - self.sigma_min) * z
        online_train = Online_v_t(t[:, 0], psi_t_val)
        loss_flow = torch.mean((online_train - d_psi) ** 2)
        torch.cuda.empty_cache()
        return loss_flow

    def loss_generator_matching(self, Online_v_t: nn.Module, Pretrained_v_t: nn.Module, Generator: nn.Module, batch_size: int = 10):
        t = (torch.rand(1, device=self.device) + torch.arange(10 * batch_size, device=self.device) / (10 * batch_size)) % (1 - self.epsilon)
        t = t[:, None]
        t_steps = torch.linspace(0, 1, 2, device=self.device)
        z = torch.randn(10 * batch_size, 3, 32, 32, device=self.device)

        traj = torchdiffeq.odeint(
            lambda _t, _x: Generator(_t, _x),
            z,
            t_steps,
            atol=1e-4,
            rtol=1e-4,
            method="euler",
        )
        x_0 = traj[-1]
        psi_t_val = self.psi_t(z, x_0, t)
        d_psi = x_0 - (1 - self.sigma_min) * z

        pre_train = Pretrained_v_t(t[:, 0], psi_t_val)
        online_train = Online_v_t(t[:, 0], psi_t_val)

        l1 = torch.mean((pre_train - online_train) ** 2)
        l2 = torch.mean(2 * ((pre_train - online_train) * (online_train - d_psi)))
        torch.cuda.empty_cache()
        return l1 + l2, x_0

def train_fgm(Pretrained_v_t, Online_v_t, Generator, data_loader):
    wandb.watch([Online_v_t, Generator], log="all", log_freq=100)

    optimizer_online = torch.optim.Adam(Online_v_t.parameters(), lr=2e-5, betas=(0.0, 0.999))
    optimizer_gen    = torch.optim.Adam(Generator.parameters(), lr=2e-5, betas=(0.0, 0.999))
    scaler_online = GradScaler()
    scaler_gen    = GradScaler()
    
    flow_loss = UnconditionalFlowLoss(sigma_min=0.0)

    n_epochs = 20000
    log_interval = 50
    save_interval = 1000

    for epoch in range(n_epochs):
        # (a) Freeze Generator, train Online_v_t (Flow Matching)
        for p in Generator.parameters():
            p.requires_grad = False
        for p in Online_v_t.parameters():
            p.requires_grad = True

        Generator.eval()
        Online_v_t.train()

        optimizer_online.zero_grad()
        with autocast():
            loss_flow = flow_loss.loss_flow_matching(
                Online_v_t=Online_v_t,
                Pretrained_v_t=Pretrained_v_t,
                Generator=Generator,
                batch_size=data_loader.batch_size
            )
        scaler_online.scale(loss_flow).backward()
        scaler_online.step(optimizer_online)
        scaler_online.update()
        loss_FM = loss_flow.item()

        # (b) Freeze Online_v_t, train Generator (Generator Matching)
        for p in Generator.parameters():
            p.requires_grad = True
        for p in Online_v_t.parameters():
            p.requires_grad = False

        Generator.train()
        Online_v_t.eval()

        optimizer_gen.zero_grad()
        with autocast():
            loss_gen, image_gen = flow_loss.loss_generator_matching(
                Online_v_t=Online_v_t,
                Pretrained_v_t=Pretrained_v_t,
                Generator=Generator,
                batch_size=data_loader.batch_size
            )
        scaler_gen.scale(loss_gen).backward()
        scaler_gen.step(optimizer_gen)
        scaler_gen.update()
        loss_GM = loss_gen.item()

        wandb.log({
            "Online_FM_Loss": loss_FM,
            "Generator_Matching_Loss": loss_GM,
            "step": epoch
        })

        if epoch % log_interval == 0:
            print(f"Iter {epoch:05d} | Online FM Loss: {loss_FM:.6f} | Generator Matching Loss: {loss_GM:.6f}")

            Generator.eval()
            with torch.no_grad():
                # 1) Pretrained ODE final output (for reference)
                z = torch.randn(10, 3, 32, 32, device=device)
                t_steps = torch.linspace(0, 1, 100, device=device)
                traj = torchdiffeq.odeint(
                    lambda _t, _x: Pretrained_v_t(_t, _x),
                    z,
                    t_steps,
                    rtol=1e-5,
                    atol=1e-5,
                    method="euler",
                )
                ref_final = traj[-1].clamp(-1, 1)
                grid_gen = make_grid(ref_final, nrow=5, value_range=(-1, 1))
                grid_gen = (grid_gen + 1) / 2.0

                # 2) Generator single-step samples
                gen_one_step = z + Generator(torch.zeros(10, device=device), z)
                grid_gen_step = make_grid(gen_one_step.clamp(-1, 1), nrow=5, value_range=(-1, 1))
                grid_gen_step = (grid_gen_step + 1) / 2.0

                # 3) Online model single-step samples
                online_one_step = z + Online_v_t(torch.zeros(10, device=device), z)
                grid_online = make_grid(online_one_step.clamp(-1, 1), nrow=5, value_range=(-1, 1))
                grid_online = (grid_online + 1) / 2.0

            wandb.log({
                "Pretrained_ODE_final": wandb.Image(grid_gen.permute(1, 2, 0).cpu().numpy()),
                "Generator_single_step": wandb.Image(grid_gen_step.permute(1, 2, 0).cpu().numpy()),
                "Online_single_step": wandb.Image(grid_online.permute(1, 2, 0).cpu().numpy()),
                "step": epoch
            })

            plt.figure(figsize=(6,6))
            plt.imshow(grid_gen.permute(1, 2, 0).cpu().numpy())
            plt.title("Pretrained ODE Final")
            plt.axis("off")
            plt.show()

            plt.figure(figsize=(6,6))
            plt.imshow(grid_gen_step.permute(1, 2, 0).cpu().numpy())
            plt.title("Generator Single-Step")
            plt.axis("off")
            plt.show()

            plt.figure(figsize=(6,6))
            plt.imshow(grid_online.permute(1, 2, 0).cpu().numpy())
            plt.title("Online Model Single-Step")
            plt.axis("off")
            plt.show()

            Generator.train()

        if epoch % save_interval == 0 and epoch != 0:
            ds_name = "celeba"
            os.makedirs(f"./checkpoints/{ds_name}/", exist_ok=True)
            torch.save(Pretrained_v_t.state_dict(), f"./checkpoints/{ds_name}/Pretrained_v_t.pt")
            torch.save(Online_v_t.state_dict(),     f"./checkpoints/{ds_name}/Online_v_t.pt")
            torch.save(Generator.state_dict(),      f"./checkpoints/{ds_name}/Generator.pt")

    wandb.finish()

def main():
    seed_everything(42)
    # Load dataset (here using CIFAR10 as an example; replace with CelebA if available)
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
    ])
    train_ds = datasets.CIFAR10(root='./data', download=True, transform=transform)
    data_loader = DataLoader(train_ds, batch_size=10, shuffle=True, num_workers=NUM_WORKERS, drop_last=True)

    Pretrained_v_t, Online_v_t, Generator = load_models()

    # Run training
    train_fgm(Pretrained_v_t, Online_v_t, Generator, data_loader)

if __name__ == "__main__":
    main()
