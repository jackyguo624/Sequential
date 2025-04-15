from lhotse import CutSet

json_path = "/aistor/aispeech/hpc_stor01/group/asr/collection/zh+en+fangyan-2w-asr/train/multitask.jsonl"
cuts = CutSet.from_file(json_path)

print(cuts)



