import os
from glob import glob
import numpy as np
import buteo as beo
import torch
from torch.utils.data import DataLoader


def callback_preprocess(x, y):
    x_norm = np.empty_like(x, dtype=np.float32)
    np.divide(x, 10000.0, out=x_norm)

    y = y.astype(np.float32, copy=False)

    return x_norm, y

def callback_postprocess(x, y):
    x = beo.channel_last_to_first(x)
    y = beo.channel_last_to_first(y)

    return torch.from_numpy(x), torch.from_numpy(y)

def callback(x, y):
    x, y = callback_preprocess(x, y)
    x, y = callback_postprocess(x, y)

    return x, y


def load_data(*,
    x="s2",
    y="lc",
    with_augmentations=False,
    num_workers=0,
    batch_size=16,
    folder="./data",
):
    """
    Loads the data from the data folder.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    x_train = beo.MultiArray([np.load(f, mmap_mode="r") for f in sorted(glob(os.path.join(folder, f"*train_{x}.npy")))])
    y_train = beo.MultiArray([np.load(f, mmap_mode="r") for f in sorted(glob(os.path.join(folder, f"*train_label_{y}.npy")))])

    x_val = beo.MultiArray([np.load(f, mmap_mode="r") for f in sorted(glob(os.path.join(folder, f"*val_{x}.npy")))])
    y_val = beo.MultiArray([np.load(f, mmap_mode="r") for f in sorted(glob(os.path.join(folder, f"*val_label_{y}.npy")))])

    x_test = beo.MultiArray([np.load(f, mmap_mode="r") for f in sorted(glob(os.path.join(folder, f"*test_{x}.npy")))])
    y_test = beo.MultiArray([np.load(f, mmap_mode="r") for f in sorted(glob(os.path.join(folder, f"*test_label_{y}.npy")))])

    assert len(x_train) == len(y_train) and len(x_test) == len(y_test) and len(x_val) == len(y_val), "Lengths of x and y do not match."

    if with_augmentations:
        ds_train = beo.DatasetAugmentation(
            x_train, y_train,
            callback_pre_augmentation=callback_preprocess,
            callback_post_augmentation=callback_postprocess,
            augmentations=[
                beo.AugmentationRotationXY(p=0.2, inplace=True),
                beo.AugmentationMirrorXY(p=0.2, inplace=True),
                beo.AugmentationCutmix(p=0.2, inplace=True),
                beo.AugmentationChannelScale(p=0.2, inplace=True),
                # beo.AugmentationBlur(p=0.2, inplace=True),
                # beo.AugmentationSharpen(p=0.2, inplace=True),
                beo.AugmentationNoiseNormal(p=0.2, inplace=True),
                beo.AugmentationNoiseUniform(p=0.2, inplace=True),
            ],
        )
    else:
        ds_train = beo.Dataset(x_train, y_train, callback=callback)

    ds_test = beo.Dataset(x_test, y_test, callback=callback)
    ds_val = beo.Dataset(x_val, y_val, callback=callback)

    dl_train = DataLoader(ds_train, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=num_workers, drop_last=True, generator=torch.Generator(device=device))
    dl_test = DataLoader(ds_test, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=num_workers, drop_last=True, generator=torch.Generator(device=device))
    dl_val = DataLoader(ds_val, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=num_workers, drop_last=True, generator=torch.Generator(device=device))

    return dl_train, dl_test, dl_val


