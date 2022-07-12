import pytorch_lightning as pl
from torch.utils.data import DataLoader
import datasets


class WavDataModule(pl.LightningDataModule):
    @staticmethod
    def add_data_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("DataModule")
        parser.add_argument("data_dir", type=str)
        parser.add_argument("--batch-size", type=int, default=4)
        parser.add_argument("--size", type=int, default=4000)
        parser.add_argument("--segment", type=int, default=16000)
        parser.add_argument("--deterministic-data", action="store_true")
        return parent_parser

    def __init__(self, batch_size=4, data_dir: str = "", size: int = 4000, segment: int = 16000, deterministic_data: bool = False,
                 train_transforms=None, val_transforms=None, test_transforms=None, dims=None, **kwargs):
        super().__init__(train_transforms, val_transforms, test_transforms, dims)
        self.data_dir = data_dir
        self.size = size
        self.segment = segment
        self.deterministic = deterministic_data
        self.batch_size = batch_size

    def setup(self, stage=None):
        self.train_dataset = datasets.RandomWAVDataset(
            self.data_dir, self.size, self.segment, self.deterministic)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size,
                          pin_memory=True, num_workers=8, prefetch_factor=4)

    def state_dict(self):
        return {
            "data_dir": self.data_dir,
            "size": self.size,
            "segment": self.segment,
            "deterministic_data": self.deterministic,
        }

    def load_state_dict(self, state_dict):
        self.data_dir = state_dict["data_dir"]
        self.size = state_dict["size"]
        self.segment = state_dict["segment"]
        self.deterministic = state_dict["deterministic_data"]
