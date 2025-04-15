from jsonargparse import ArgumentParser
from typing import Callable, Any
from lhotse.cut import CutSet
from lhotse.dataset.sampling.base import CutSampler

parser = ArgumentParser()
parser.add_argument("--config", action="config")
parser.add_argument('--download', type=Callable)
parser.add_argument('--prepare', type=Callable)
parser.add_argument('--sampler', type=Callable[[CutSet], CutSampler])

cfg = parser.parse_args()
init = parser.instantiate_classes(cfg)
d_path = init.download
print(f'd_res: {d_path}')
p_res = init.prepare
print(f'p_res: {p_res}')
cuts = CutSet.from_manifests(**p_res['train-clean-5'])
sampler = init.sampler(cuts)
print(f'sampler: {sampler}')
