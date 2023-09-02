import buteo as beo
import numpy as np
import torch

def predict_func(model, epoch, name, tile_size=64, n_offsets=3, batch_size=16, device=None):
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model.eval()
    model.to(device)

    img_path = "./data/naestved_s2.tif"
    img_arr = beo.raster_to_array(img_path, filled=True, fill_value=0, cast=np.float32) / 10000.0

    def predict(arr):
        swap = beo.channel_last_to_first(arr)
        as_torch = torch.from_numpy(swap).float()
        on_device = as_torch.to(device)
        predicted = model(on_device)
        on_cpu = predicted.cpu()
        as_numpy = on_cpu.detach().numpy()
        swap_back = beo.channel_first_to_last(as_numpy)

        return swap_back

    with torch.no_grad():
        predicted = beo.predict_array(
            img_arr,
            callback=predict,
            tile_size=tile_size,
            n_offsets=n_offsets,
            batch_size=batch_size,
        )
    beo.array_to_raster(
        predicted,
        reference=img_path,
        out_path=F"./predictions/pred_{name}_{epoch}.tif",
    )
