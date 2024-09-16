import torch
import os
import numpy as np


# Dataset class to load train and validation data
class Dataset(torch.utils.data.Dataset):

    def __init__(self, root, partition= "train-clean-100"): # Feel free to add more arguments

        self.input_dir       = f'{root}/inputs'
        self.output_dir = f'{root}/outputs'

        input_filenames          = sorted(os.listdir(self.input_dir))
        output_filenames    = sorted(os.listdir(self.output_dir))

        pass

    def __len__(self):
        return self.length

    def __getitem__(self, ind):
        pass

class TestDataset(torch.utils.data.Dataset):
    # A test dataset class similar to the previous class but you dont have outputs for this
    # Imp: Read the mfccs in sorted order, do NOT shuffle the data here or in your dataloader.
    def __init__(self, root, phonemes = PHONEMES, context=0, partition= "train-clean-100"): # Feel free to add more arguments
        self.input_dir       = f'{root}/inputs'
        self.output_dir = f'{root}/outputs'

        input_filenames          = sorted(os.listdir(self.input_dir))
        output_filenames    = sorted(os.listdir(self.output_dir))

        pass

    def __len__(self):
        return self.length

    def __getitem__(self, ind):
        pass
