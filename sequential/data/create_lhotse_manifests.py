import kaldiio
import json
import tqdm
import argparse
from typing import Dict, List, Tuple, Union, Optional
from pathlib import Path
from lhotse import (
    Recording, AudioSource, SupervisionSegment, RecordingSet, SupervisionSet
)
from multiprocessing import Pool
from functools import partial
import logging
from collections import defaultdict
from scipy.signal import resample
# 设置日志格式, 经典格式
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

'''
aisv1 format:
{
    "key": "xxx",
    "task": "ASR",
    "target": "text",
    "path": "kaldi.ark:offset",
}
'''

def fix_duplicate_ids(lines: List[str], verbose: bool=False) -> Dict[str, List[str]]:
    # First, find duplicate IDs
    id_counts = defaultdict(int)
    for i, line in tqdm.tqdm(enumerate(lines), desc="Fixing duplicate IDs", total=len(lines)):
        data = json.loads(line)
        id_counts[data['key']] += 1
        if id_counts[data['key']] > 1:
            data['key'] = f"{data['key']}-dedup{id_counts[data['key']] - 1}"
        lines[i] = json.dumps(data)

    logging.info(f"Fixed {sum([1 for count in id_counts.values() if count > 1])} duplicate IDs")

    if verbose:
        for key, count in id_counts.items():
            if count > 1:
                logging.info(f"Duplicate ID: {key}, count: {count}")

    return lines


def process_aisv1_jsonl(line: str, 
                        num_channels: int=1,
                        force_sampling_rate: Optional[int]=None) -> Tuple[Recording, SupervisionSegment]:

    data = json.loads(line)

    # 使用kaldiio读取mat文件, read wav.ark:offset to (sample_rate, narray1d)
    sampling_rate, samples = kaldiio.load_mat(data['path'])
    if force_sampling_rate and sampling_rate != force_sampling_rate:
        logging.info(f"Resampled {data['key']} from {sampling_rate} to {force_sampling_rate} to calucate num_samples")
        num_samples = int(len(samples) * force_sampling_rate / sampling_rate)
        samples = resample(samples, num_samples)
        sampling_rate = force_sampling_rate

    num_samples = samples.shape[0]
    channels = [i for i in range(num_channels)]
    duration = num_samples / num_channels / sampling_rate

    # 创建Recording
    recording = Recording(
        id=data['key'],
        sources=[
            AudioSource(type='file',
                        channels=channels,
                        source=data['path'])
        ],
        sampling_rate=sampling_rate,
        duration=duration,
        num_samples=num_samples
    )

    # 创建Supervision
    supervision = SupervisionSegment(
        id=data['key'],
        recording_id=data['key'],
        start=0.0,
        duration=duration,
        channel=channels,
        text=data['target'],
    )
    return recording, supervision

def convert_to_manifest(jsonl_path: Union[Path, str], 
                        output_dir: Union[Path, str], 
                        format: str, 
                        num_channels: int=1, 
                        num_workers: int=1, 
                        force_sampling_rate: Optional[int]=None):
    with open(jsonl_path, 'r') as f:
        lines = f.readlines()

    # fix duplicate ids
    lines = fix_duplicate_ids(lines)

    # Process the data
    recordings = []
    supervisions = []
    func_name = f"process_{format}_jsonl"
    func = globals()[func_name]

    func = partial(func, num_channels=num_channels, force_sampling_rate=force_sampling_rate)
    with Pool(num_workers) as p:
        for recording, supervision in tqdm.tqdm(p.imap(func, lines), total=len(lines)):
            recordings.append(recording)
            supervisions.append(supervision)

    # 转换为Lhotse对象
    recording_set = RecordingSet.from_recordings(recordings)
    supervision_set = SupervisionSet.from_segments(supervisions)

    # 保存结果
    logging.info(f"Saving recordings to {output_dir}/recordings.jsonl")
    recording_set.to_file(f"{output_dir}/recordings.jsonl")
    logging.info(f"Saving supervisions to {output_dir}/supervisions.jsonl")
    supervision_set.to_file(f"{output_dir}/supervisions.jsonl")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--jsonl_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--num-channels", type=int, default=1)
    parser.add_argument("--format", type=str, default="aisv1", choices=["aisv1"])
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--force-sampling-rate", type=int, default=None)
    args = parser.parse_args()
    convert_to_manifest(args.jsonl_path, args.output_dir, args.format, args.num_channels, args.num_workers, args.force_sampling_rate)
