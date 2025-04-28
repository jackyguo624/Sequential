# 实现一个audio backend 继承 lhotse.audio.AudioBackend
# 可以使用kaldiio.load_mat 读取 wav.ark 中保存的音频数据
# 使用kaldiio.save_mat 保存音频数据到 wav.ark 中

import kaldiio
import numpy as np
from typing import Tuple, Optional, Union
from lhotse.utils import Pathlike, Seconds
from lhotse.audio.backend import AudioBackend, FileObject
from scipy.signal import resample

class KaldiAudioBackend(AudioBackend):
    def __init__(self, *args, force_sampling_rate: Optional[int] = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.force_sampling_rate = force_sampling_rate

    def read_audio(
        self,
        path_or_fd: Union[Pathlike, FileObject],
        offset: Seconds = 0.0,
        duration: Optional[Seconds] = None,
        force_opus_sampling_rate: Optional[int] = None,
    ) -> Tuple[np.ndarray, int]:
        sr, audio = kaldiio.load_mat(path_or_fd)
        force_sampling_rate = self.force_sampling_rate or force_opus_sampling_rate
        if force_sampling_rate and sr != force_sampling_rate:
            # 计算新的样本数
            num_samples = int(len(audio) * force_sampling_rate / sr)
            audio = resample(audio, num_samples)
            sr = force_sampling_rate

        begin = int(offset * sr)
        end = begin + int(duration * sr) if duration is not None else None
        audio = audio[begin:end]
        return audio, sr

