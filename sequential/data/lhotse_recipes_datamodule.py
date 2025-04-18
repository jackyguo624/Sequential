import pytorch_lightning as pl
from typing import Optional, Dict, Callable
from lhotse import CutSet
from lhotse.dataset.sampling.base import CutSampler
from lhotse.dataset.dataloading import make_worker_init_fn
from torch.utils.data import DataLoader, Dataset
from pathlib import Path
from jsonargparse import ArgumentParser, set_parsing_settings
from typing import Tuple, Union
import os

class LhotseRecipesDataModule(pl.LightningDataModule):
    def __init__(self,
                 train_dataset: Callable[[], Dataset],
                 train_sampler: Callable[[Union[CutSet, Tuple[CutSet]]], CutSampler],
                 val_dataset: Callable[[], Dataset] = None,
                 val_sampler: Callable[[Union[CutSet, Tuple[CutSet]]], CutSampler] = None,
                 test_dataset: Callable[[], Dataset] = None,
                 test_sampler: Callable[[Union[CutSet, Tuple[CutSet]]], CutSampler] = None,
                 train_manifest: str = None,
                 val_manifest: str = None,
                 test_manifest: str = None,
                 download: Callable = None,
                 prepare: Callable = None,
                 ):
        super().__init__()
        self.manifests = None
        self.download = download
        self.prepare = prepare
        self.train_manifest = train_manifest
        self.train_dataset = train_dataset
        self.train_sampler = train_sampler
        self.val_dataset = val_dataset
        self.val_sampler = val_sampler
        self.test_dataset = test_dataset
        self.test_sampler = test_sampler
        self.train_manifest = train_manifest
        self.val_manifest = val_manifest
        self.test_manifest = test_manifest

    def setup(self, stage: Optional[str] = None):
        self.manifests = self.prepare

    def train_dataloader(self):
        return self.get_dataloader(self.train_dataset, self.train_sampler, self.train_manifest)

    def get_worker_init_fn(self) -> Callable | None:
        num_node = 1
        if os.environ.get("NODE_RANK", None) is not None:
            rank = int(os.environ.get("NODE_RANK", None)) * num_node + \
                int(os.environ.get("LOCAL_RANK", None))
            world_size = int(os.environ.get("WORLD_SIZE", None))
            worker_init_fn = make_worker_init_fn(rank=rank, world_size=world_size)
            print(f'rank: {rank}, world: {world_size} ')
        else:
            worker_init_fn = None
        return worker_init_fn

    def get_dataloader(self, dataset: Callable[[], Dataset],
                       sampler: Callable[[Union[CutSet, Tuple[CutSet]]], CutSampler],
                       manifest_part: str = None):
        manifests = self.manifests
        if manifest_part:
            manifests = self.manifests[manifest_part]

        batch_sampler = sampler(CutSet.from_manifests(**manifests))
        worker_init_fn = self.get_worker_init_fn()
        print(batch_sampler)
        dataloader = DataLoader(dataset(),
                                batch_sampler=batch_sampler,
                                num_workers=10,
                                persistent_workers=True,
                                worker_init_fn=worker_init_fn,
                                collate_fn=lambda x: x)


        return dataloader
