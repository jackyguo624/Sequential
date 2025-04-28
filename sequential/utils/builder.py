from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

def sanity_check_kwargs(kwargs):
    if 'torch_dtype' in kwargs:
        if kwargs['torch_dtype'] is not None:
            # Convert string representation to actual torch dtype
            if isinstance(kwargs['torch_dtype'], str):
                if kwargs['torch_dtype'].startswith('torch.'):
                    dtype_name = kwargs['torch_dtype'].split('.')[-1]
                    kwargs['torch_dtype'] = getattr(torch, dtype_name)
    return kwargs

def auto_tokenizer_from_pretrained_wrapper(pretrained_model_name_or_path: str, *args, **kwargs):
    sanity_check_kwargs(kwargs)
    return AutoTokenizer.from_pretrained(pretrained_model_name_or_path, *args, **kwargs)

def auto_model_for_causal_lm_from_pretrained_wrapper(pretrained_model_name_or_path: str, *args, **kwargs):
    sanity_check_kwargs(kwargs)
    return AutoModelForCausalLM.from_pretrained(pretrained_model_name_or_path, *args, **kwargs)
