from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm
import argparse
import torch
import os

from soft_spatial_labels import SoftSpatialCrossEntropyLoss, OneHotEncoder2D
from utils import cosine_scheduler, convert_torch_to_float
from torchmetrics import Accuracy, Precision, Recall
from data_loaders import load_data
from predict import predict_func
from model import MLPMixer


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train Model')
    # Model's arguments
    parser.add_argument('--augmentation',           type=bool,  default=False)
    parser.add_argument('--batch_size',             type=int,   default=16)
    parser.add_argument('--num_epochs',             type=int,   default=100)
    parser.add_argument('--warmup_epochs',          type=int,   default=10)
    parser.add_argument('--min_epochs',             type=int,   default=25)
    parser.add_argument('--patience',               type=int,   default=10)
    parser.add_argument('--learning_rate',          type=float, default=0.0001)
    parser.add_argument('--learning_rate_end',      type=float, default=0.000001)
    parser.add_argument('--save_best_model',        type=bool,  default=True)
    parser.add_argument('--create_predictions',     type=bool,  default=True)
    parser.add_argument('--classes',                type=list,  default=[10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 100])
    # Soft Spatial Labels arguments
    parser.add_argument('--loss_method',            type=str,   default='max')
    parser.add_argument('--loss_strenght',          type=float, default=1.01)
    parser.add_argument('--loss_kernel_radius',     type=float, default=2.0)
    parser.add_argument('--loss_kernel_circular',   type=bool,  default=True)
    parser.add_argument('--loss_kernel_sigma',      type=float, default=2.0)

    # Reading arguments
    args = parser.parse_args()
    AUGMENTATION            = bool(args.augmentation)
    BATCH_SIZE              = int(args.batch_size)
    NUM_EPOCHS              = int(args.num_epochs)
    WARMUP_EPOCHS           = int(args.warmup_epochs)
    MIN_EPOCHS              = int(args.min_epochs)
    PATIENCE                = int(args.patience)
    LEARNING_RATE           = float(args.learning_rate)
    LEARNING_RATE_END       = float(args.learning_rate_end)
    SAVE_BEST_MODEL         = bool(args.save_best_model)
    CREATE_PREDICTIONS      = bool(args.create_predictions)
    CLASSES                 = list(args.classes)

    LOSS_METHOD             = str(args.loss_method)
    LOSS_STRENGHT           = float(args.loss_strenght)
    LOSS_KERNEL_RADIUS      = float(args.loss_kernel_radius)
    LOSS_KERNEL_CIRCULAR    = bool(args.loss_kernel_circular)
    LOSS_KERNEL_SIGMA       = float(args.loss_kernel_sigma)

    NAME = f"aug{AUGMENTATION}-bs{BATCH_SIZE}-e{NUM_EPOCHS}-we{WARMUP_EPOCHS}-me{MIN_EPOCHS}-p{PATIENCE}-lr{LEARNING_RATE}-lre{LEARNING_RATE_END}-lm{LOSS_METHOD}-ls{LOSS_STRENGHT}-lkr{LOSS_KERNEL_RADIUS}-lkc{LOSS_KERNEL_CIRCULAR}-lks{LOSS_KERNEL_SIGMA}"

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    '''
        @Casper -- add quick description of the model
    '''
    model = MLPMixer(
        chw=(10, 64, 64),
        output_dim=11,
        patch_size=4,
        dim=512,
        depth=5,
        channel_scale=2,
        drop_n=0.0,
        drop_p=0.1,
    )

    metric_accuracy  = Accuracy(task="multiclass", num_classes=11); metric_accuracy.__name__ = "accuracy"
    metric_precision = Precision(task="multiclass", num_classes=11, average="macro"); metric_precision.__name__ = "precision"
    metric_recall    = Recall(task="multiclass", num_classes=11, average="macro"); metric_recall.__name__ = "recall"
    metrics = [
    #    metric_accuracy,
    #    metric_precision,
    #    metric_recall,
    ]

    criterion = SoftSpatialCrossEntropyLoss(
        method=LOSS_METHOD,
        classes=CLASSES,
        strength=LOSS_STRENGHT,
        kernel_radius=LOSS_KERNEL_RADIUS,
        kernel_circular=LOSS_KERNEL_CIRCULAR,
        kernel_sigma=LOSS_KERNEL_SIGMA,
        device=device,
    )
    encoder = torch.nn.Identity()

    # If we use the normal cross entropy loss, we need to use the OneHotEncoder2D
    # criterion = torch.nn.CrossEntropyLoss()
    # encoder = OneHotEncoder2D(classes, device=device)

    dl_train, dl_val, dl_test = load_data(with_augmentations=AUGMENTATION, batch_size=BATCH_SIZE)

    torch.set_default_device(device)

    if torch.cuda.is_available(): torch.cuda.empty_cache()
    else: print("No CUDA device available.")

    if WARMUP_EPOCHS > 0: print(f"Starting warmup for {WARMUP_EPOCHS} epochs...")
    else: print("Starting training...")

    
    model.to(device)

    lr_schedule_values = cosine_scheduler(
        LEARNING_RATE, LEARNING_RATE_END, NUM_EPOCHS + WARMUP_EPOCHS, WARMUP_EPOCHS, LEARNING_RATE_END,
    )

    # Loss and optimizer
    optimizer = torch.optim.AdamW(model.parameters(), eps=1e-7)
    scaler = GradScaler()

    # Save the initial learning rate in optimizer's param_groups
    for param_group in optimizer.param_groups:
        param_group['initial_lr'] = lr_schedule_values[0]

    best_epoch        = 0
    best_loss         = None
    best_model_state  = model.state_dict().copy()
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
        train_metrics_values = { metric.__name__: 0.0 for metric in metrics }

        # Initialize the progress bar for training
        epoch_current = epoch + 1 if epoch < WARMUP_EPOCHS else epoch + 1 - WARMUP_EPOCHS
        epoch_max = NUM_EPOCHS if epoch >= WARMUP_EPOCHS else WARMUP_EPOCHS

        train_pbar = tqdm(dl_train, total=len(dl_train), desc=f"Epoch {epoch_current}/{epoch_max}")

        for i, (images, labels) in enumerate(train_pbar):
            # Move inputs and targets to the device (GPU)
            images, labels = images.to(device), labels.to(device)
            labels = encoder(labels)

            # Zero the gradients
            optimizer.zero_grad()

            # Cast to bfloat16
            with autocast(dtype=torch.float16):
                outputs = model(images)
                loss = criterion(outputs, labels)

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

            train_loss += loss.item()

            for metric in metrics:
                train_metrics_values[metric.__name__] += metric(outputs, labels)

            train_pbar.set_postfix({
                "loss": f"{train_loss / (i + 1):.4f}",
                **{name: f"{value / (i + 1):.4f}" for name, value in train_metrics_values.items()}
            })

            # Validate at the end of each epoch
            # This is done in the same scope to keep tqdm happy.
            if i == len(dl_train) - 1:

                val_metrics_values = { metric.__name__: 0.0 for metric in metrics }
                # Validate every epoch
                with torch.no_grad():
                    model.eval()

                    val_loss = 0
                    for j, (images, labels) in enumerate(dl_val):
                        images = images.to(device)
                        labels = labels.to(device)
                        labels = encoder(labels)

                        outputs = model(images)

                        loss = criterion(outputs, labels)
                        val_loss += loss.item()

                        for metric in metrics:
                            val_metrics_values[metric.__name__] += metric(outputs, labels)

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
        for k, (images, labels) in enumerate(dl_test):
            images = images.to(device)
            labels = labels.to(device)
            labels = encoder(labels)

            outputs = model(images)

            loss = criterion(outputs, labels)
            test_loss += loss.item()

        print(f"Test Accuracy: {test_loss / (k + 1):.4f}")

    # Save the model
    if SAVE_BEST_MODEL:
        torch.save(best_model_state, os.path.join("./models", f"{NAME}.pt"))
