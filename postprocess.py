import os
import buteo as beo
import numpy as np

FOLDER = "./predictions/"

path_hard = os.path.join(FOLDER, "pred_HARD_LABEL_LOSS_MIXER_01_53.tif")
path_soft = os.path.join(FOLDER, "pred_SOFT_LABEL_LOSS_MIXER_01_79.tif")

hard = beo.raster_to_array(path_hard)
soft = beo.raster_to_array(path_soft)

beo.array_to_raster(
    np.argmax(hard, axis=2, keepdims=True).astype(np.uint8),
    reference=path_hard,
    out_path=path_hard.replace(".tif", "_argmax.tif"),
)

beo.array_to_raster(
    np.argmax(soft, axis=2, keepdims=True).astype(np.uint8),
    reference=path_soft,
    out_path=path_soft.replace(".tif", "_argmax.tif"),
)
