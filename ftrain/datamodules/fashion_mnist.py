import torch  # noqa: D100
from pytorch_lightning import LightningDataModule
from pytorch_lightning.utilities.types import (
    EVAL_DATALOADERS,
    TRAIN_DATALOADERS,
)
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from torchvision.datasets import FashionMNIST

__all__ = ["FashionMnistDataModule"]


class FashionMnistDataModule(LightningDataModule):  # noqa: D101
    channels = 1
    H, W = 28, 28

    def __init__(  # noqa: D107
        self,  # noqa: ANN101
        data_dir: str,
        batch_size: int = 32,
        transformations: torch.nn.Module = None,
        num_workers: int = 4,
    ) -> None:
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.data_dir = data_dir

        if transformations is None:
            self.transforms = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.RandomAffine(
                        degrees=(-15, 15), translate=(0.01, 0.01)
                    ),
                ]
            )
        else:
            self.transforms = transformations

        self.save_hyperparameters()

    def prepare_data(self) -> None:  # noqa: ANN101, D102
        FashionMNIST(self.data_dir, train=True, download=True)
        FashionMNIST(self.data_dir, train=False, download=True)

    def setup(self, stage: str) -> None:  # noqa: ANN101, D102
        if stage == "fit":
            self.train_data = FashionMNIST(
                self.data_dir, train=True, transform=self.transforms
            )
            self.train_data, self.val_data = random_split(
                self.train_data, lengths=[0.8, 0.2]
            )

        if stage == "test":
            self.test_data = FashionMNIST(
                self.data_dir, train=False, transform=self.transforms[0]
            )

        if stage == "predict":
            self.predict_data = FashionMNIST(
                self.data_dir, train=False, transform=self.transforms[0]
            )

    def train_dataloader(self) -> TRAIN_DATALOADERS:  # noqa: ANN101, D102
        return DataLoader(
            self.train_data,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
        )

    def val_dataloader(self) -> EVAL_DATALOADERS:  # noqa: ANN101, D102
        return DataLoader(
            self.val_data,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )

    def test_dataloader(self) -> EVAL_DATALOADERS:  # noqa: ANN101, D102
        return DataLoader(
            self.test_data,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )

    def predict_dataloader(self) -> EVAL_DATALOADERS:  # noqa: ANN101, D102
        return DataLoader(
            self.predict_data,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )
