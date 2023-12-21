from os.path import exists, join, basename
from os import makedirs, remove
from six.moves import urllib
import tarfile
from torchvision.transforms import Compose, ToTensor

from data.dataset import LoadTestDataFromFolder, LoadTrainingDataFromFolder,LoadEvalDataFromFolder

def transform():
    return Compose([
        ToTensor(),
    ])

def get_training_set(data_dir, nFrames, upscale_factor, data_augmentation, file_list, patch_size):
    return LoadTrainingDataFromFolder(data_dir, nFrames, upscale_factor, data_augmentation, file_list,patch_size,
                             transform=transform())


def get_eval_set(data_dir, nFrames, upscale_factor, file_list):
    return LoadEvalDataFromFolder(data_dir, nFrames, upscale_factor, file_list, transform=transform())


def get_real_test_set(data_dir, nFrames, upscale_factor, file_list):
    return LoadTestDataFromFolder(data_dir, nFrames, upscale_factor, file_list,transform=transform())

