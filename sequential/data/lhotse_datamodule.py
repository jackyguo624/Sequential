import pytorch_lightning as pl
from typing import Optional, Callable, Dict
from lhotse import CutSet
from lhotse import RecordingSet, SupervisionSet
from lhotse.dataset.sampling.base import CutSampler
from lhotse.dataset.dataloading import make_worker_init_fn
from torch.utils.data import DataLoader, Dataset
from pathlib import Path
from typing import Tuple, Union
import os
from lhotse.audio.backend import AudioBackend, set_current_audio_backend

import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class LhotseDataModule(pl.LightningDataModule):
    def __init__(self,
                 train_dataset: Callable[[], Dataset],
                 train_sampler: Callable[[Union[CutSet, Tuple[CutSet]]], CutSampler],
                 val_dataset: Callable[[], Dataset] = None,
                 val_sampler: Callable[[Union[CutSet, Tuple[CutSet]]], CutSampler] = None,
                 test_dataset: Callable[[], Dataset] = None,
                 test_sampler: Callable[[Union[CutSet, Tuple[CutSet]]], CutSampler] = None,
                 audio_backend: AudioBackend = None,
                 ):
        super().__init__()
        self.train_dataset = train_dataset
        self.train_sampler = train_sampler
        self.val_dataset = val_dataset
        self.val_sampler = val_sampler
        self.test_dataset = test_dataset
        self.test_sampler = test_sampler
        self.audio_backend = audio_backend
        if self.audio_backend:
            logging.info(f"Setting audio backend to {self.audio_backend}")
            set_current_audio_backend(self.audio_backend)

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
                       cuts: CutSet):

        batch_sampler = sampler(cuts)
        worker_init_fn = self.get_worker_init_fn()
        dataloader = DataLoader(dataset(),
                                batch_sampler=batch_sampler,
                                num_workers=10,
                                persistent_workers=True,
                                worker_init_fn=worker_init_fn,
                                collate_fn=lambda x: x)

        return dataloader


class LhotseRecipesDataModule(LhotseDataModule):
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
                 audio_backend: AudioBackend = None,
                 ):
        super().__init__(train_dataset, train_sampler, val_dataset, val_sampler, test_dataset, test_sampler, audio_backend)
        self.download = download
        self.prepare = prepare
        self.train_manifest = train_manifest
        self.val_manifest = val_manifest
        self.test_manifest = test_manifest


    def setup(self, stage: Optional[str] = None):
        self.manifests = self.prepare

    def train_dataloader(self):
        manifests = self.manifests
        if self.train_manifest:
            manifests = self.manifests[self.train_manifest]
        if 'cuts' in manifests:
            cuts = manifests['cuts']
        else:
            cuts = CutSet.from_manifests(recordings=manifests['recordings'], 
                                         supervisions=manifests['supervisions'])
        return self.get_dataloader(self.train_dataset, self.train_sampler, cuts)



class LhotseManifestDataModule(LhotseDataModule):
    def __init__(self,
                 train_manifest_dir: Path,
                 train_dataset: Callable[[], Dataset],
                 train_sampler: Callable[[Union[CutSet, Tuple[CutSet]]], CutSampler],
                 val_manifest_dir: Path = None,
                 val_dataset: Callable[[], Dataset] = None,
                 val_sampler: Callable[[Union[CutSet, Tuple[CutSet]]], CutSampler] = None,
                 test_manifest_dir: Path = None,
                 test_dataset: Callable[[], Dataset] = None,
                 test_sampler: Callable[[Union[CutSet, Tuple[CutSet]]], CutSampler] = None,
                 audio_backend: AudioBackend = None,
                 ):
        super().__init__(train_dataset, train_sampler, val_dataset, val_sampler, test_dataset, test_sampler, audio_backend)
        self.train_manifest_dir = train_manifest_dir
        self.val_manifest_dir = val_manifest_dir
        self.test_manifest_dir = test_manifest_dir
    
    def load_manifest(self, manifest_dir: Path) -> Dict:
        recording_set = RecordingSet.from_jsonl(manifest_dir / "recordings.jsonl")
        supervision_set = SupervisionSet.from_jsonl(manifest_dir / "supervisions.jsonl")
        cut_set = CutSet.from_manifests(recordings=recording_set, supervisions=supervision_set)
        return {"recordings": recording_set, "supervisions": supervision_set, "cuts": cut_set}
    
    def setup(self, stage: Optional[str] = None):
        if self.train_manifest_dir and stage == "fit":
            self.train_manifest = self.load_manifest(self.train_manifest_dir)
        if self.val_manifest_dir and stage == "fit":
            self.val_manifest = self.load_manifest(self.val_manifest_dir)
        if self.test_manifest_dir and stage == "test":
            self.test_manifest = self.load_manifest(self.test_manifest_dir)
    
    def train_dataloader(self):
        if self.train_manifest: 
            cuts = self.train_manifest['cuts']
        else:
            cuts = CutSet.from_manifests(recordings=self.train_manifest['recordings'], 
                                         supervisions=self.train_manifest['supervisions'])
        return self.get_dataloader(self.train_dataset, self.train_sampler, cuts)
    
