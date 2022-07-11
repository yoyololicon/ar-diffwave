import torch
import torch.nn.functional as F


def gamma2as(g: torch.Tensor):
    var = g.sigmoid()
    return (1 - var).sqrt(), var


def gamma2snr(g: torch.Tensor) -> torch.Tensor:
    return torch.exp(-g)


def snr2as(snr: torch.Tensor):
    snr_p1 = snr + 1
    return torch.sqrt(snr / snr_p1), snr_p1.reciprocal()


def gamma2logas(g: torch.Tensor):
    log_var = -F.softplus(-g)
    return 0.5 * (-g + log_var), log_var
