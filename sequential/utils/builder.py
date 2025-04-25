from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

def sanity_check_kwargs(kwargs):
    if 'torch_dtype' in kwargs:
        if kwargs['torch_dtype'] is not None:
            # dymanic import torch_dtype, since kwargs not support torch_dtype in string format
            # e.g. 'torch.float16' -> torch.float16
            from importlib import importlib
            torch_dtype = importlib.import_module(kwargs['torch_dtype'])
            kwargs['torch_dtype'] = torch_dtype
            print(f"torch_dtype: {kwargs['torch_dtype']}")
    return kwargs

def auto_tokenizer_from_pretrained_wrapper(pretrained_model_name_or_path: str, *args, **kwargs):
    sanity_check_kwargs(kwargs)
    return AutoTokenizer.from_pretrained(pretrained_model_name_or_path, *args, **kwargs)

def auto_model_for_causal_lm_from_pretrained_wrapper(pretrained_model_name_or_path: str, *args, **kwargs):
    sanity_check_kwargs(kwargs)
    return AutoModelForCausalLM.from_pretrained(pretrained_model_name_or_path, *args, **kwargs)
