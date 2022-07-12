import math
from random import uniform
from torch import optim
import torch
from torch import nn
from torch.nn import functional as F
import pytorch_lightning as pl
from torchaudio.transforms import MelSpectrogram

from transformers import PositionalEncoding
from utils import gamma2as, gamma2logas


@torch.jit.script
def silu(x):
    return x * torch.sigmoid(x)


@torch.jit.script
def gru(x):
    a, b = x.chunk(2, 1)
    return a.tanh() * b.sigmoid()


class ResidualBlock(nn.Module):
    last_layer: bool
    channels: int

    def __init__(self, residual_channels, dilation, last_layer=False):
        super().__init__()
        self.dilated_conv = nn.Conv1d(
            residual_channels, 2 * residual_channels, 3, padding=dilation, dilation=dilation)

        self.output_projection = nn.Conv1d(
            residual_channels, residual_channels if last_layer else residual_channels * 2, 1)

        self.last_layer = last_layer
        self.channels = residual_channels

    def forward(self, x, c):
        y = self.output_projection(gru(self.dilated_conv(x) + c))
        res = y[:, :self.channels]
        skip = y[:, self.channels:]
        if self.last_layer:
            return None, res
        res = (res + x) * 0.70710678118654757
        return res, skip


class DiffusionEmbedding(nn.Module):
    def __init__(self, max_steps):
        super().__init__()
        self.register_buffer('embedding', self._build_embedding(
            max_steps + 1), persistent=False)
        self.projection1 = nn.Linear(128, 512)
        self.projection2 = nn.Linear(512, 512)

    def forward(self, diffusion_step: torch.Tensor):
        if not diffusion_step.is_floating_point():
            x = self.embedding[diffusion_step]
        else:
            x = self._lerp_embedding(diffusion_step)
        x = self.projection1(x)
        x = silu(x)
        x = self.projection2(x)
        x = silu(x)
        return x

    def _lerp_embedding(self, t):
        t = t * (self.embedding.shape[0] - 1)
        low_idx = torch.floor(t).long()
        high_idx = torch.ceil(t).long()
        low = self.embedding[low_idx]
        high = self.embedding[high_idx]
        return low + (high - low) * (t - low_idx)[:, None]

    def _build_embedding(self, max_steps):
        steps = torch.arange(max_steps).unsqueeze(1)  # [T,1]
        dims = torch.arange(64).unsqueeze(0)          # [1,64]
        table = steps * 10.0**(dims * 4.0 / 63.0)     # [T,64]
        table = torch.view_as_real(torch.exp(1j * table)).view(max_steps, -1)
        return table


class ARDiffWave(pl.LightningModule):
    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("LitModel")
        parser.add_argument("--sr", type=int, default=16000)
        parser.add_argument("--diff_channels", type=int, default=64)
        parser.add_argument("--diff_layers", type=int, default=30)
        parser.add_argument("--cycle_length", type=int, default=10)
        parser.add_argument("--hop_length", type=int, default=256)
        parser.add_argument("--n_fft", type=int, default=1024)
        parser.add_argument("--d_model", type=int, default=128)
        parser.add_argument("--nhead", type=int, default=4)
        parser.add_argument("--trsfmr_layers", type=int, default=4)
        parser.add_argument("--dim_feedforward", type=int, default=512)
        parser.add_argument("--dropout", type=float, default=0.1)
        return parent_parser

    def __init__(self,
                 sr,
                 diff_channels,
                 diff_layers,
                 cycle_length,
                 n_fft,
                 hop_length,
                 d_model,
                 nhead,
                 dim_feedforward,
                 dropout,
                 trsfmr_layers,
                 **kwargs) -> None:
        super().__init__()

        self.save_hyperparameters(ignore=list(kwargs.keys()))

        diff_channels = self.hparams.diff_channels
        diff_layers = self.hparams.diff_layers
        cycle_length = self.hparams.cycle_length
        n_fft = self.hparams.n_fft
        hop_length = self.hparams.hop_length
        d_model = self.hparams.d_model
        nhead = self.hparams.nhead
        dim_feedforward = self.hparams.dim_feedforward
        dropout = self.hparams.dropout
        trsfmr_layers = self.hparams.trsfmr_layers
        sr = self.hparams.sr

        # construct te diffwave part
        self.input_projection = nn.Sequential(
            nn.Conv1d(1, diff_channels, 1),
            nn.ReLU(inplace=True)
        )
        self.output_projection = nn.Sequential(
            nn.Conv1d(diff_channels, diff_channels, 1),
            nn.ReLU(inplace=True),
            nn.Conv1d(diff_channels, 1, 1)
        )

        dilations = [2 ** (i % cycle_length) for i in range(diff_layers)]
        self.residual_layers = nn.ModuleList([
            ResidualBlock(diff_channels, d) for d in dilations[:-1]
        ])
        self.residual_layers.append(ResidualBlock(
            diff_channels, dilations[-1], last_layer=True))

        self.diffusion_embedding = DiffusionEmbedding(1000)
        self.diffusion_projection = nn.Linear(
            512, diff_channels * diff_layers, bias=False)
        self.conditioner = nn.Conv1d(
            d_model, diff_channels * 2 * diff_layers, 1, bias=False)

        # construct the melspec part
        self.feature_frontend = MelSpectrogram(
            sample_rate=sr, n_fft=n_fft, hop_length=hop_length,
            n_mels=80, normalized=True
        )

        self.emb_projection = nn.Linear(80, d_model, bias=False)
        self.emb_dropout = nn.Dropout(dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer, trsfmr_layers)

        self.pe = PositionalEncoding(d_model)
        self.feature_upsampler = nn.Upsample(
            scale_factor=hop_length, mode='nearest')
        self.hop_length = hop_length

        # construct diffusion parameters
        self.gamma0 = torch.nn.Parameter(torch.tensor(-23.))
        self.gamma1 = torch.nn.Parameter(torch.tensor(3.6))

    def get_gamma(self, t):
        return self.gamma0 + (self.gamma1 - self.gamma0) * t

    def get_feature(self, x):
        return self.feature_frontend(x).add_(1e-6).log10_().mul_(10).add_(60).relu_() / 80

    def diffusion_forward(self, audio, feature, diffusion_step):
        x = audio.unsqueeze(1)
        x = self.input_projection(x)

        diffusion_embs = self.diffusion_embedding(diffusion_step)
        diffusion_embs = self.diffusion_projection(
            diffusion_embs).unsqueeze(2).chunk(len(self.residual_layers), 1)
        condition = self.conditioner(feature).chunk(
            len(self.residual_layers), 1)
        skip = torch.zeros_like(x)
        for i, layer in enumerate(self.residual_layers):
            res, s = layer(x + diffusion_embs[i], condition[i])
            skip += s
            if res is not None:
                x = res

        x = skip / math.sqrt(len(self.residual_layers))
        x = self.output_projection(x).squeeze(1)
        return x

    def forward(self, z_t, x_prev, diffusion_step):
        assert z_t.size() == x_prev.size()
        cats = torch.cat([x_prev, z_t], dim=1)
        feats = self.get_feature(cats).transpose(1, 2)
        embs = self.emb_dropout(self.emb_projection(
            feats)) + self.pe(feats.size(1))
        h = self.encoder(embs)
        valid_length = z_t.size(1) // self.hop_length + 1
        h = h[:, -valid_length:, :].transpose(1, 2)

        upsampled_h = self.feature_upsampler(h)[..., -z_t.shape[1]:]
        return self.diffusion_forward(z_t, upsampled_h, diffusion_step)

    def training_step(self, batch, batch_idx):
        x: torch.Tensor = batch
        N = x.shape[0]
        x_prev, x = x.chunk(2, 1)

        noise = torch.randn_like(x)

        t = (uniform(0, 1) + torch.arange(N, device=self.device)) / N
        gamma = self.get_gamma(t)

        alpha_t, var_t = gamma2as(gamma)
        z_t = alpha_t[:, None] * x + var_t.sqrt()[:, None] * noise
        noise_hat = self(z_t, x_prev, t)

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
        }
        self.log_dict(values, prog_bar=False, sync_dist=True)
        return loss

    def diffusion_loss(self, d_gamma_t, x, noise, noise_hat):
        log_alpha_1, log_var_1 = gamma2logas(self.gamma1)

        x_flat = x.reshape(-1)
        x_dot = x_flat @ x_flat / x_flat.numel()

        # prior loss KL(q(z_1| x) || p(z_1))
        prior_loss = 0.5 * (log_var_1.exp() + x_dot *
                            torch.exp(log_alpha_1 * 2) - 1 - log_var_1)

        # recon loss E[-log p(x | z_0)]
        ll = -0.5 * (self.gamma0 + 1 + math.log(2 * math.pi))

        # diffusion loss
        diff = noise - noise_hat
        diff = diff.view(-1)
        loss_T = 0.5 * d_gamma_t * diff @ diff / diff.numel()
        return loss_T, prior_loss, ll

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=0.0002)


if __name__ == '__main__':
    from torchinfo import summary
    model = ARDiffWave(16000, 64, 30, 10, 1024, 256, 128,
                       4, 512, 0.1, trsfmr_layers=8).cuda()

    x = torch.randn(4, 16000).cuda()
    z = torch.randn(4, 16000).cuda()
    t = torch.rand(4).cuda()

    h = model.get_feature(x)
    print(h.max(), h.min())
    exit()

    summary(model, input_data=(x, z, t), device='cuda',
            col_names=("input_size", "output_size", "num_params",
                       "mult_adds"),
            depth=2,
            row_settings=("depth", "var_names"))
    model = model.to_torchscript()
    # y = model(z, x, t)
    # print(y.shape)
