import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelSummary, LearningRateMonitor
from pytorch_lightning.strategies.ddp import DDPStrategy
from argparse import ArgumentParser
from model import ARDiffWave
from unet import UNet
from data import WavDataModule


def main(args):
    args_dict = vars(args)

    pl.seed_everything(2434)

    gpus = torch.cuda.device_count()

    if args.ckpt_path:
        lit_model = UNet.load_from_checkpoint(args.ckpt_path)
    else:
        lit_model = UNet(**args_dict)

    if args.ckpt_path:
        state_dict = torch.load(args.ckpt_path, map_location="cpu")[
            'WavDataModule']
        wav_data = WavDataModule(**state_dict)
    else:
        wav_data = WavDataModule(**args_dict)

    callbacks = [
        ModelSummary(max_depth=2),
        # LearningRateMonitor()
    ]

    trainer = pl.Trainer.from_argparse_args(args,
                                            callbacks=callbacks, log_every_n_steps=1,
                                            benchmark=True, detect_anomaly=True, gpus=gpus,
                                            strategy=DDPStrategy(find_unused_parameters=False) if gpus > 1 else None)
    trainer.fit(lit_model, wav_data, ckpt_path=args.ckpt_path)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser = pl.Trainer.add_argparse_args(parser)
    parser = UNet.add_model_specific_args(parser)
    parser = WavDataModule.add_data_specific_args(parser)
    parser.add_argument("--ckpt-path", type=str)

    torch.set_float32_matmul_precision("high")

    args = parser.parse_args()
    main(args)
