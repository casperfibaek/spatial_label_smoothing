import os
import buteo as beo
import numpy as np

FOLDER = "./predictions/"

for path in [
    # os.path.join(FOLDER, "pred_HARD_LABEL_LOSS_MIXER_01_53.tif"),
    # os.path.join(FOLDER, "pred_SOFT_LABEL_LOSS_MIXER_01_79.tif"),
    os.path.join(FOLDER, "pred_SOFT_LABEL_LOSS_MIXER_02_38.tif"), 
]:
    arr = beo.raster_to_array(path, filled=True, fill_value=0.0, cast=np.float32)

    beo.array_to_raster(
        np.argmax(arr, axis=2, keepdims=True).astype(np.uint8),
        reference=path,
        out_path=path.replace(".tif", "_argmax.tif"),
    )
