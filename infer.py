import torch
from argparse import ArgumentParser
import torchaudio
from tqdm import tqdm

from model import ARDiffWave
from utils import gamma2logas

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("ckpt_path", type=str)
    parser.add_argument("--duration", type=int, default=3)
    parser.add_argument("--file", type=str, default="test.wav")
    parser.add_argument("-T", type=int, default=100)
    args = parser.parse_args()

    model = ARDiffWave.load_from_checkpoint(args.ckpt_path, map_location="cpu")
    state = torch.load(args.ckpt_path, map_location="cpu")['WavDataModule']
    context = state['segment'] // 2
    sr = model.hparams.sr
    output_length = int(sr * args.duration)
    total_steps = output_length // context

    diffusion_steps = torch.linspace(0, 1, args.T)
    with torch.no_grad():
        gamma = model.get_gamma(diffusion_steps)
    diffusion_steps = diffusion_steps.cuda()
    gamma = gamma.cuda()

    jitted = model.to_torchscript().cuda()
    jitted.eval()

    log_alpha, log_var = gamma2logas(gamma)
    alpha = log_alpha.exp()
    var = log_var.exp()
    alpha_st = torch.exp(log_alpha[:-1] - log_alpha[1:])
    c = -torch.expm1(gamma[:-1] - gamma[1:])
    c.relu_()

    outputs = []
    prev_chunk = gamma.new_zeros(1, context)
    with torch.no_grad():
        for i in tqdm(range(total_steps)):
            z_t = torch.randn_like(prev_chunk)
            for t in range(args.T - 1, 0, -1):
                s = t - 1
                noise_hat = jitted(z_t, prev_chunk, diffusion_steps[t:t+1])
                mu = (z_t - var[t].sqrt() * c[s] * noise_hat) * alpha_st[s]
                z_t = mu
                z_t += (var[s] * c[s]).sqrt() * torch.randn_like(z_t)

            noise_hat = jitted(z_t, prev_chunk, diffusion_steps[:1])
            pred = (z_t - var[0].sqrt() * noise_hat) * torch.exp(-log_alpha[0])
            outputs.append(pred)
            prev_chunk = pred

    pred = torch.cat(outputs, dim=1).cpu()
    torchaudio.save(args.file, pred, sr)
