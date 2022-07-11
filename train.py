import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelSummary, LearningRateMonitor
from pytorch_lightning.plugins import DDPPlugin
from argparse import ArgumentParser
from model import ARDiffWave
from data import WavDataModule


def main(args):
    args_dict = vars(args)

    pl.seed_everything(2434)

    gpus = torch.cuda.device_count()

    lit_model = ARDiffWave(**args_dict)
    wav_data = WavDataModule(**args_dict)

    callbacks = [
        ModelSummary(max_depth=2),
        # LearningRateMonitor()
    ]

    trainer = pl.Trainer(
        callbacks=callbacks, log_every_n_steps=1,
        benchmark=True, detect_anomaly=True, gpus=gpus,
        strategy=DDPPlugin(find_unused_parameters=False) if gpus > 1 else None)
    trainer.fit(lit_model, wav_data, ckpt_path=args.ckpt_path)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser = pl.Trainer.add_argparse_args(parser)
    parser = ARDiffWave.add_model_specific_args(parser)
    parser = WavDataModule.add_data_specific_args(parser)
    parser.add_argument("--ckpt-path", type=str)

    args = parser.parse_args()
    main(args)
