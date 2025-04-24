import kaldiio
import json
import tqdm
import argparse
from lhotse import (
    Recording, AudioSource, SupervisionSegment, RecordingSet, SupervisionSet
)
from multiprocessing import Pool
from functools import partial
import logging
'''
aisv1 format:
{
    "key": "xxx",
    "task": "ASR",
    "target": "text",
    "path": "kaldi.ark:offset",
}
'''

def process_aisv1_jsonl(line, num_channels: int =1):
    data = json.loads(line)
    # 使用kaldiio读取mat文件, read wav.ark:offset to (sample_rate, narray1d)
    sampling_rate, samples = kaldiio.load_mat(data['path'])
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

def convert_aisv1_to_manifest(jsonl_path, output_dir, num_channels=1, num_workers=1):
    recordings = []
    supervisions = []

    with open(jsonl_path, 'r') as f:
        lines = f.readlines()
        func = partial(process_aisv1_jsonl, num_channels=num_channels)
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

    args = parser.parse_args()
    func = f"convert_{args.format}_to_manifest"
    globals()[func](
        jsonl_path=args.jsonl_path,
        output_dir=args.output_dir,
        num_channels=args.num_channels,
        num_workers=args.num_workers
    )