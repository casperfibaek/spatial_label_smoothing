import os
import torch
from tqdm import tqdm

from data_loaders import load_data
from utils import cosine_scheduler, convert_torch_to_float
from predict import predict_func
from model import MLPMixer
from soft_spatial_labels import SoftSpatialCrossEntropyLoss, OneHotEncoder2D
from torchmetrics.classification import MulticlassF1Score, MulticlassPrecision, MulticlassRecall, MulticlassJaccardIndex
from functools import partial


NAME = "SOFT_LABEL_LOSS_MIXER_02"
BATCH_SIZE = 16
NUM_EPOCHS = 100
WARMUP_EPOCHS = 10
MIN_EPOCHS = 25
PATIENCE = 10
LEARNING_RATE = 0.001
LEARNING_RATE_END = 0.00001
SAVE_BEST_MODEL = True
AUGMENTATIONS = True
CREATE_PREDICTIONS = True
USE_SOFT_LOSS = False

classes = [10, 30, 40, 50, 60, 80, 90]
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = MLPMixer(
    chw=(10, 64, 64),
    output_dim=len(classes),
    patch_size=4,
    dim=512,
    depth=5,
    channel_scale=2,
    drop_n=0.1,
    drop_p=0.1,
)

if USE_SOFT_LOSS:
    criterion = SoftSpatialCrossEntropyLoss(
        method="max",
        loss_method="focal_error_squared",
        classes=classes,
        kernel_radius=2.0,
        kernel_circular=True,
        kernel_sigma=2.0,
        device=device,
    )
else:
    class EntropyLossWrapper(torch.nn.Module):
        def __init__(self, criterion, classes, device):
            super().__init__()
            self.criterion = criterion
            self.onehot = OneHotEncoder2D(classes=classes, device=device)

        def forward(self, output, target):
            return self.criterion(output, self.onehot(target))
        
    criterion = EntropyLossWrapper(torch.nn.CrossEntropyLoss(), classes, device)

def metric_wrapper(output, target, metric_func, classes, device, raw=True):
    batch_size, channels, height, width = output.shape
    classes = torch.Tensor(classes).view(1, -1, 1, 1).to(device)

    target_hot = (target == classes).float().to(device)
    target_max = torch.argmax(target_hot, dim=1, keepdim=True).to(device)

    if raw:
        _output = output.permute(0, 2, 3, 1).reshape(-1, channels)
        _target = target_max.permute(0, 2, 3, 1).reshape(-1)
    else:
        output_max = torch.argmax(output, dim=1, keepdim=True).to(device)
        _output = output_max.permute(0, 2, 3, 1).reshape(-1)
        _target = target_max.permute(0, 2, 3, 1).reshape(-1)

    metric = metric_func(_output, _target)

    return metric

metric_jac = partial(metric_wrapper, metric_func=partial(MulticlassJaccardIndex(num_classes=len(classes), average="macro").to(device)), classes=classes, device=device)
metric_f1 = partial(metric_wrapper, metric_func=partial(MulticlassF1Score(num_classes=len(classes), average="macro").to(device)), classes=classes, device=device)
metric_precision = partial(metric_wrapper, metric_func=partial(MulticlassPrecision(num_classes=len(classes), average="macro").to(device)), classes=classes, device=device)
metric_recall = partial(metric_wrapper, metric_func=partial(MulticlassRecall(num_classes=len(classes), average="macro").to(device)), classes=classes, device=device)

_metrics = {
    "jac": metric_jac,
    "f1": metric_f1,
    "prec": metric_precision,
    "rec": metric_recall,
}

dl_train, dl_val, dl_test = load_data(with_augmentations=True, batch_size=BATCH_SIZE)

torch.set_default_device(device)

if torch.cuda.is_available():
    torch.cuda.empty_cache()
else:
    print("No CUDA device available.")

if WARMUP_EPOCHS > 0:
    print(f"Starting warmup for {WARMUP_EPOCHS} epochs...")
else:
    print("Starting training...")

model.to(device)

lr_schedule_values = cosine_scheduler(
    LEARNING_RATE, LEARNING_RATE_END, NUM_EPOCHS + WARMUP_EPOCHS, WARMUP_EPOCHS, LEARNING_RATE_END,
)

# Loss and optimizer
optimizer = torch.optim.AdamW(model.parameters(), eps=1e-7)

# Save the initial learning rate in optimizer's param_groups
for param_group in optimizer.param_groups:
    param_group['initial_lr'] = lr_schedule_values[0]

best_epoch = 0
best_loss = None
best_model_state = model.state_dict().copy()
epochs_no_improve = 0

# Training loop
for epoch in range(NUM_EPOCHS + WARMUP_EPOCHS):
    if epoch == WARMUP_EPOCHS and WARMUP_EPOCHS > 0:
        print("Finished warmup. Starting training...")

    model.train()

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr_schedule_values[epoch]

    # Initialize the running loss
    train_loss = 0.0
    train_metrics_values = { name : 0.0 for name in _metrics }

    # Initialize the progress bar for training
    epoch_current = epoch + 1 if epoch < WARMUP_EPOCHS else epoch + 1 - WARMUP_EPOCHS
    epoch_max = NUM_EPOCHS if epoch >= WARMUP_EPOCHS else WARMUP_EPOCHS

    train_pbar = tqdm(dl_train, total=len(dl_train), desc=f"Epoch {epoch_current}/{epoch_max}")

    for i, (images, labels) in enumerate(train_pbar):
        # Move inputs and targets to the device (GPU)
        images, labels = images.to(device), labels.to(device)

        # Zero the gradients
        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

        for metric_name in _metrics:
            metric = _metrics[metric_name]
            metric_value = metric(outputs, labels)
            train_metrics_values[metric_name] += metric_value

        train_pbar.set_postfix({
            "loss": f"{train_loss / (i + 1):.4f}",
            **{name: f"{value / (i + 1):.4f}" for name, value in train_metrics_values.items()}
        })

        # Validate at the end of each epoch
        # This is done in the same scope to keep tqdm happy.
        if i == len(dl_train) - 1:

            val_metrics_values = { name : 0.0 for name in _metrics }
            # Validate every epoch
            with torch.no_grad():
                model.eval()

                val_loss = 0
                for j, (images, labels) in enumerate(dl_val):
                    images = images.to(device)
                    labels = labels.to(device)

                    outputs = model(images)

                    loss = criterion(outputs, labels)
                    val_loss += loss.item()

                    for metric_name in _metrics:
                        metric = _metrics[metric_name]
                        metric_value = metric(outputs, labels)
                        val_metrics_values[metric_name] += metric_value

            # Append val_loss to the train_pbar
            loss_dict = {
                "loss": train_loss / (i + 1),
                **{name: value / (i + 1) for name, value in train_metrics_values.items()},
                "val_loss": val_loss / (j + 1),
                **{f"val_{name}": value / (j + 1) for name, value in val_metrics_values.items()},
            }
            loss_dict = { key: convert_torch_to_float(value) for key, value in loss_dict.items() }
            loss_dict_str = { key: f"{value:.4f}" for key, value in loss_dict.items() }

            train_pbar.set_postfix(loss_dict_str, refresh=True)

            if best_loss is None:
                best_epoch = epoch_current
                best_loss = val_loss
                best_model_state = model.state_dict().copy()

                if CREATE_PREDICTIONS and predict_func is not None and epoch >= WARMUP_EPOCHS:
                    predict_func(model, epoch_current, name=NAME)

            elif best_loss > val_loss:
                best_epoch = epoch_current
                best_loss = val_loss
                best_model_state = model.state_dict().copy()

                if CREATE_PREDICTIONS and predict_func is not None and epoch >= WARMUP_EPOCHS:
                    predict_func(model, epoch_current, name=NAME)

                epochs_no_improve = 0
            else:
                epochs_no_improve += 1

    # Early stopping
    if epochs_no_improve == PATIENCE and epoch >= WARMUP_EPOCHS and epoch_current >= MIN_EPOCHS + PATIENCE:
        print(f'Early stopping triggered after {epoch_current} epochs.')
        break

# Load the best weights
model.load_state_dict(best_model_state)

print("Finished Training. Best epoch: ", best_epoch)
print("")
print("Starting Testing... (Best val epoch).")
model.eval()

# Test the model
with torch.no_grad():
    test_loss = 0
    test_metrics_values = { name : 0.0 for name in _metrics }

    test_pbar = tqdm(dl_test, total=len(dl_test), desc=f"Testing.. Best epoch: {best_epoch}")
    for k, (images, labels) in enumerate(test_pbar):
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)

        loss = criterion(outputs, labels)
        test_loss += loss.item()

        for metric_name in _metrics:
            metric = _metrics[metric_name]
            metric_value = metric(outputs, labels)
            test_metrics_values[metric_name] += metric_value

        loss_dict = { key: convert_torch_to_float(value) for key, value in loss_dict.items() }
        loss_dict_str = { key: f"{value:.4f}" for key, value in loss_dict.items() }

        test_pbar.set_postfix({
            "test_loss": f"{test_loss / (k + 1):.4f}",
            **{name: f"{value / (k + 1):.4f}" for name, value in test_metrics_values.items()}
        })

        train_pbar.set_postfix(loss_dict_str, refresh=True)

    print(f"Test Loss: {test_loss / (k + 1):.4f}")

# Save the model
if SAVE_BEST_MODEL:
    torch.save(best_model_state, os.path.join("./models", f"{NAME}.pt"))

# Test Loss: 0.0412
# criterion = SoftSpatialCrossEntropyLoss(
#     method="max",
#     classes=classes,
#     kernel_radius=2.0,
#     kernel_circular=True,
#     kernel_sigma=2.0,
#     device=device,
# )
# val_loss=0.0456, val_jac=0.5896, val_f1=0.6798, val_prec=0.7382, val_rec=0.6719]

# val_loss=1.2871, val_jac=0.5890, val_f1=0.6762, val_prec=0.7113, val_rec=0.6706]
# Test Loss: 1.2825