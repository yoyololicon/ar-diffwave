import torch
from argparse import ArgumentParser
import torchaudio
from tqdm import tqdm

from model import ARDiffWave
from unet import UNet
from utils import gamma2logas

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("ckpt_path", type=str)
    parser.add_argument("--file", type=str, default="test.wav")
    parser.add_argument("-T", type=int, default=100)
    args = parser.parse_args()

    model = UNet.load_from_checkpoint(args.ckpt_path, map_location="cpu")
    state = torch.load(args.ckpt_path, map_location="cpu")['WavDataModule']
    context = state['segment']
    sr = 22050

    diffusion_steps = torch.linspace(0, 1, args.T)
    with torch.no_grad():
        gamma = model.get_gamma(diffusion_steps)
    diffusion_steps = diffusion_steps.cuda()
    gamma = gamma.cuda()
    scale = model.prior_logvar.exp().sqrt().item()

    # jitted = model.to_torchscript().cuda()
    jitted = model.cuda()
    jitted.eval()

    log_alpha, log_var = gamma2logas(gamma)
    alpha = log_alpha.exp()
    var = log_var.exp()
    alpha_st = torch.exp(log_alpha[:-1] - log_alpha[1:])
    c = -torch.expm1(gamma[:-1] - gamma[1:])
    c.relu_()

    with torch.no_grad():
        z_t = gamma.new_empty((1, context * 4)).normal_(std=scale)
        for t in range(args.T - 1, 0, -1):
            s = t - 1
            noise_hat = jitted(z_t, diffusion_steps[t:t+1], context // 1024)
            mu = (z_t - var[t].sqrt() * c[s] * noise_hat) * alpha_st[s]
            z_t = mu
            z_t += (var[s] * c[s]).sqrt() * torch.randn_like(z_t)

        noise_hat = jitted(z_t, diffusion_steps[:1])
        pred = (z_t - var[0].sqrt() * noise_hat) * torch.exp(-log_alpha[0])
    torchaudio.save(args.file, pred.cpu(), sr)
