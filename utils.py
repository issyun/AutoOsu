import torch
from pathlib import Path

def round_base(input, base, index=False):
    if isinstance(input, float) or isinstance(input, int):
        return round(input / base) if index else base * round(input / base)
    elif isinstance(input, torch.Tensor):
        return (input / base).round() if index else (input / base).round() * base
    else:
        raise NotImplementedError
    
def combination_to_index(combination, num_classes):
    index = 0
    for i, value in enumerate(combination):
        index += value * (num_classes ** i)
    return index

def index_to_combination(index, num_classes):
    combination = []
    for i in range(num_classes):
        value = (index // (num_classes ** i)) % num_classes
        combination.append(value)
    return tuple(combination)

def move_file(old_path:Path, new_path:Path, overwrite=True):
    if new_path.exists():
        if overwrite:
            new_path.unlink()
            old_path.rename(new_path)
        elif old_path.exists():
            old_path.unlink()
    elif old_path.exists():
        old_path.rename(new_path)

def find_file_by_stem(fn_list, stem, substring=False):
        for fn in fn_list:
            if substring and stem in fn.stem:
                return fn
            elif not substring and fn.stem == stem:
                return fn
        return -1