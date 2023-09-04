import os
import torch
from tqdm import tqdm

from data_loaders import load_data
from utils import cosine_scheduler, convert_torch_to_float, metric_wrapper
from predict import predict_func
from model import MLPMixer
from soft_spatial_labels import SoftSpatialSegmentationLoss, SoftSegmentationLoss
from torchmetrics.classification import MulticlassF1Score, MulticlassPrecision, MulticlassRecall, MulticlassJaccardIndex
from functools import partial


BATCH_SIZE = 16
NUM_EPOCHS = 50
WARMUP_EPOCHS = 10
MIN_EPOCHS = 25
PATIENCE = 10
LEARNING_RATE = 0.001
LEARNING_RATE_END = 0.00001
SAVE_BEST_MODEL = True
AUGMENTATIONS = True
CREATE_PREDICTION = True


def run_test(
    loss_method="cross_entropy",
    flip_protection="half",
    use_softloss=True,
    smoothing=0.1,
    kernel_radius=1.0,
    kernel_circular=True,
    kernel_sigma=2.0,
    scale_using_var=False,
    iteration=0,
):
    NAME = f"EXP2-{iteration}"
    NAME += f"_LOSS-{loss_method}"
    NAME += f"_SOFT-{use_softloss}"
    NAME += f"_SMOOTH-{smoothing}" if not use_softloss else ""
    NAME += f"_VAR-{scale_using_var}" if use_softloss else ""

    print(f"Running test {NAME}...")
    print("")

    if os.path.exists(f"./logs/{NAME}.csv"):
        print(f"Test {NAME} already run. Skipping...")
        return

    log_file = open(f"./logs/{NAME}.csv", "w")

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

    if use_softloss:
        criterion = SoftSpatialSegmentationLoss(
            method=flip_protection,
            loss_method=loss_method,
            classes=classes,
            scale_using_var=scale_using_var,
            kernel_radius=kernel_radius,
            kernel_circular=kernel_circular,
            kernel_sigma=kernel_sigma,
            device=device,
        )
    else:
        criterion = SoftSegmentationLoss(
            smoothing=smoothing,
            loss_method=loss_method,
            classes=classes,
            device=device,
        )

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
            loss = criterion(outputs, labels, images)
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

                        loss = criterion(outputs, labels, images)
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

                elif best_loss > val_loss:
                    best_epoch = epoch_current
                    best_loss = val_loss
                    best_model_state = model.state_dict().copy()

                    epochs_no_improve = 0
                else:
                    epochs_no_improve += 1

                if epoch == 0:
                    log_file.write(",".join(loss_dict.keys()) + "\n")

                log_file.write(",".join([str(round(value, 6)) for value in loss_dict.values()]) + "\n")

        # Early stopping
        if epochs_no_improve >= PATIENCE and epoch >= WARMUP_EPOCHS and epoch_current >= MIN_EPOCHS + PATIENCE:
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

            loss = criterion(outputs, labels, images)
            test_loss += loss.item()

            for metric_name in _metrics:
                metric = _metrics[metric_name]
                metric_value = metric(outputs, labels)
                test_metrics_values[metric_name] += metric_value

            loss_dict = {
                "test_loss": test_loss / (k + 1),
                **{f"test_{name}": value / (k + 1) for name, value in test_metrics_values.items()},
            }

            loss_dict = { key: convert_torch_to_float(value) for key, value in loss_dict.items() }
            loss_dict_str = { key: f"{value:.4f}" for key, value in loss_dict.items() }

            test_pbar.set_postfix(loss_dict_str)

        print(f"Test Loss: {test_loss / (k + 1):.4f}")
        log_file.write(",".join([str(round(value, 6)) for value in loss_dict.values()]) + "\n")
        log_file.close()

    # Save the model
    if SAVE_BEST_MODEL:
        torch.save(best_model_state, os.path.join("./models", f"{NAME}.pt"))
        print(f"Saved best model to ./models/{NAME}.pt")

    if CREATE_PREDICTION and predict_func is not None:
        predict_func(model, best_epoch, name=NAME)


if __name__ == "__main__":
    model_run = 0
    for loss_method in ["cross_entropy", "logcosh_dice", "kl_divergence", "nll"]:
        for scale_using_var in [True, False]:
            for iteration in [0, 1, 2]:
                run_test(
                    loss_method=loss_method,
                    use_softloss=True,
                    iteration=iteration,
                    scale_using_var=scale_using_var,
                )
                model_run += 1
                print(model_run)

    for loss_method in ["cross_entropy", "logcosh_dice", "kl_divergence", "nll"]:
        for smoothing in [0.0, 0.1]:
            for iteration in [0, 1, 2]:
                run_test(
                    loss_method=loss_method,
                    use_softloss=False,
                    smoothing=smoothing,
                    iteration=iteration,
                )
                model_run += 1
                print(model_run)