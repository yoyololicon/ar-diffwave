import math
from random import uniform
from torch import optim
import torch
from torch import nn
from torch.nn import functional as F
import pytorch_lightning as pl
from torchaudio.transforms import MelSpectrogram
from audio_diffusion_pytorch import UNet1d
from demucs import Demucs

from transformers import PositionalEncoding
from utils import gamma2as, gamma2logas


class UNet(pl.LightningModule):
    def __init__(self,
                 channels,
                 depth,
                 rescale,
                 kernel_size,
                 stride,
                 attention_layers,
                 **kwargs) -> None:
        super().__init__()

        self.save_hyperparameters(ignore=list(kwargs.keys()))

        self.unet = Demucs(channels=channels, depth=depth, rescale=rescale,
                           kernel_size=kernel_size, stride=stride, attention_layers=attention_layers)
        # self.unet = UNet1d(
        #     in_channels=1,
        #     patch_size=8,
        #     channels=64,
        #     multipliers=[1, 2, 4, 8, 8, 8],
        #     factors=[4, 4, 4, 2, 2],
        #     attentions=[False, False, False, True, True],
        #     num_blocks=[2, 2, 2, 2, 2],
        #     attention_heads=8,
        #     attention_features=64,
        #     attention_multiplier=2,
        #     resnet_groups=8,
        #     kernel_multiplier_downsample=2,
        #     kernel_sizes_init=[1, 3, 7],
        #     use_nearest_upsample=False,
        #     use_skip_scale=True,
        #     use_attention_bottleneck=True,
        #     use_learned_time_embedding=True,
        # )

        # construct diffusion parameters
        self.gamma0 = torch.nn.Parameter(torch.tensor(-17.))
        self.gamma1 = torch.nn.Parameter(torch.tensor(0.))
        self.prior_logvar = torch.nn.Parameter(torch.tensor(0.))

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("LitModel")
        parser.add_argument('--channels', type=int, default=64)
        parser.add_argument('--depth', type=int, default=5)
        parser.add_argument('--rescale', type=float, default=0.1)
        parser.add_argument('--kernel_size', type=int, default=8)
        parser.add_argument('--stride', type=int, default=4)
        parser.add_argument('--attention_layers', type=int, default=4)
        return parent_parser

    def get_gamma(self, t):
        return self.gamma0 + (self.gamma1 - self.gamma0) * t

    def forward(self, audio, diffusion_step, *args, **kwargs):
        x = audio.unsqueeze(1)
        return self.unet(x, diffusion_step, *args, **kwargs).squeeze(1)

    def training_step(self, batch, batch_idx):
        x: torch.Tensor = batch
        N = x.shape[0]

        noise = torch.randn_like(x)

        t = (uniform(0, 1) + torch.arange(N, device=self.device)) / N
        gamma = self.get_gamma(t)

        alpha_t, var_t = gamma2as(gamma)
        z_t = alpha_t[:, None] * x + var_t.sqrt()[:, None] * noise
        noise_hat = self(z_t, t)

        d_gamma_t = self.gamma1 - self.gamma0
        loss_T, prior_kld, ll = self.diffusion_loss(
            d_gamma_t, x, noise, noise_hat)
        loss = loss_T + prior_kld - ll

        values = {
            'loss': loss,
            'loss_T': loss_T,
            'prior_kld': prior_kld,
            'll': ll,
            'gamma0': self.gamma0,
            'gamma1': self.gamma1,
            'prior_logvar': self.prior_logvar,
        }
        self.log_dict(values, prog_bar=False, sync_dist=True)
        return loss

    def diffusion_loss(self, d_gamma_t, x, noise, noise_hat):
        log_alpha_1, log_var_1 = gamma2logas(self.gamma1)

        x_flat = x.reshape(-1)
        x_dot = x_flat @ x_flat / x_flat.numel()

        # prior loss KL(q(z_1| x) || p(z_1))
        prior_loss = 0.5 * ((log_var_1 - self.prior_logvar).exp() + x_dot *
                            torch.exp(log_alpha_1 * 2 - self.prior_logvar) - 1 - log_var_1 + self.prior_logvar)

        # recon loss E[-log p(x | z_0)]
        ll = -0.5 * (self.gamma0 + 1 + math.log(2 * math.pi))

        # diffusion loss
        diff = noise - noise_hat
        diff = diff.view(-1)
        loss_T = 0.5 * d_gamma_t * diff @ diff / diff.numel()
        return loss_T, prior_loss, ll

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=0.0002)
