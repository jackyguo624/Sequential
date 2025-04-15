from lhotse import RecordingSet, SupervisionSet, CutSet
import json

def convert_jsonl_to_lhotse(jsonl_path, output_dir):
    recordings = []
    supervisions = []
    
    with open(jsonl_path) as f:
        for line in f:
            data = json.loads(line)
            
            # 创建Recording
            recordings.append({
                'type': 'file',
                'path': data['audio_path'],
                'duration': data['duration'],
                'sampling_rate': 16000  # 根据实际采样率修改
            })
            
            # 创建Supervision
            supervisions.append({
                'recording_id': data['audio_path'],  # 用路径作为唯一ID
                'start': 0.0,
                'duration': data['duration'],
                'text': data['text'],
                'language': data['language'],
                # 可添加其他自定义字段
                'custom': {k: v for k, v in data.items() if k not in ['audio_path', 'text', 'duration']}
            })
    
    # 转换为Lhotse对象
    recording_set = RecordingSet.from_jsonl_lazy(recordings)
    supervision_set = SupervisionSet.from_dicts(supervisions)
    
    # 创建CutSet
    cut_set = CutSet.from_manifests(
        recordings=recording_set,
        supervisions=supervision_set
    )
    
    # 保存结果
    recording_set.to_file(f"{output_dir}/recordings.jsonl")
    cut_set.to_file(f"{output_dir}/cuts.jsonl")

# 使用示例
convert_jsonl_to_lhotse(
    jsonl_path="/aistor/aispeech/hpc_stor01/group/asr/collection/zh+en+fangyan-2w-asr/train/multitask.jsonl",
    output_dir="./lhotse_data"
) 