import os
import torch
from data.fewshot_datasets import *

ID_to_DIRNAME = {
    'oxford_flowers': 'oxford_flowers',
    'dtd': 'dtd',
    'oxford_pets': 'oxford_pets',
    'stanford_cars': 'stanford_cars',
    'ucf101': 'ucf101',
    'caltech101': 'caltech-101',
    'food101': 'food-101',
    'sun397': 'sun397',
    'fgvc_aircraft': 'fgvc_aircraft',
    'eurosat': 'eurosat',
}


def build_test_loader(set_id, transform, data_root, batch_size):
    if set_id in fewshot_datasets:
        testset = build_fewshot_dataset(
            set_id, os.path.join(data_root, ID_to_DIRNAME[set_id.lower()]), transform)
    else:
        raise NotImplementedError(f"unsupported dataset: {set_id}")
    val_loader = torch.utils.data.DataLoader(
        testset, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)
    return val_loader
