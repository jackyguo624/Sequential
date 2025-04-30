from lhotse.dataset.sampling import SimpleCutSampler
from typing import Optional
from lhotse import Seconds
from lhotse import CutSet

class SimpleSamplerWraper(SimpleCutSampler):
    def __init__(self, cuts: CutSet, max_duration: Seconds = None, max_cuts: Optional[int] = None, **kwargs):
        super().__init__(cuts, max_duration=max_duration, max_cuts=max_cuts, **kwargs)
        if max_duration is not None:
            self.batch_size = int(max_duration / 2.0)
        elif max_cuts is not None:
            self.batch_size = max_cuts
        else:
            self.batch_size = None
