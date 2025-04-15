import pytorch_lightning as pl
from typing import Optional, Dict, Callable
from lhotse import CutSet
from lhotse.dataset.sampling.base import CutSampler
from torch.utils.data import DataLoader, Dataset
from pathlib import Path
from jsonargparse import ArgumentParser, set_parsing_settings
from typing import Tuple, Union

class LhotseRecipesDataModule(pl.LightningDataModule):
    def __init__(self,
                 recipe_config: Union[Path, Dict],
                 train_params: Dict):
        super().__init__()
        self.recipe_config = recipe_config
        self.train_params = train_params
        self.manifests = None


    def setup(self, stage: Optional[str] = None):
        parser = ArgumentParser()
        parser.add_argument('--download', type=Callable)
        parser.add_argument('--prepare', type=Callable)
        
        if isinstance(self.recipe_config, (Path, str)):
            cfg = parser.parse_path(self.recipe_config)
        else:  # Dict case
            cfg = parser.parse_object(self.recipe_config)
        
        init = parser.instantiate_classes(cfg)
        self.manifests = init.prepare

    def train_dataloader(self):
        return self.get_dataloader(self.train_params)

    def get_dataloader(self, params: Dict):
        parser = ArgumentParser()

        set_parsing_settings(parse_optionals_as_positionals=True)
        parser.add_argument('--manifest_part', type=str, default=None)
        parser.add_argument('--dataset', type=Callable[[], Dataset])
        parser.add_argument('--sampler', type=Callable[[Union[CutSet, Tuple[CutSet]]], CutSampler])
        cfg = parser.parse_object(params)
        
        part = cfg.manifest_part
        if part is None:
            manifests = self.manifests
        else:
            manifests = self.manifests[part]

        init = parser.instantiate_classes(cfg)
        cuts = CutSet.from_manifests(**manifests)

        sampler = init.sampler(cuts)
        dataset = init.dataset()
        dataloader = DataLoader(dataset, batch_sampler=sampler, num_workers=20, collate_fn=lambda x: x)

        return dataloader
