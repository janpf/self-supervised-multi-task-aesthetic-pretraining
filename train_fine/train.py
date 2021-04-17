import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from torch import cuda, optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

sys.path[0] = "/workspace"
from train_fine.dataset import AVA
from train_fine.NIMA import NIMA, earth_movers_distance

parser = argparse.ArgumentParser()

# training parameters
parser.add_argument("--conv_lr", type=float, default=0.0001)
parser.add_argument("--dense_lr", type=float, default=0.001)
parser.add_argument("--lr_decay_rate", type=float, default=0.95)
parser.add_argument("--percentage_of_dataset", type=int, default=None)


parser.add_argument("--load_path", type=str, default=None)

# misc
parser.add_argument("--log_dir", type=str, default="/scratch/train_logs/IA2NIMA/AVA")
parser.add_argument("--ckpt_path", type=str, default="/scratch/ckpts/IA2NIMA/AVA")

config = parser.parse_args()

logging.basicConfig(format="%(asctime)s %(levelname)-8s %(message)s", level=logging.INFO, datefmt="%Y-%m-%d %H:%M:%S")
logging.getLogger("PIL.PngImagePlugin").setLevel(logging.INFO)

if config.percentage_of_dataset is not None:
    config.log_dir = config.log_dir + str(config.percentage_of_dataset) + "/"
    config.ckpt_path = config.ckpt_path + str(config.percentage_of_dataset) + "/"
else:
    config.log_dir = config.log_dir + "/"
    config.ckpt_path = config.ckpt_path + "/"

settings: List[str] = []

if config.load_path is not None:
    if "scores-None" in config.load_path:
        settings.append(str(None))
    elif "scores-one" in config.load_path:
        settings.append("one")
    elif "scores-three" in config.load_path:
        settings.append("three")

    if "change_regress" in config.load_path:
        settings.append("change_regress")

    if "change_class" in config.load_path:
        settings.append("change_class")

    if "m-0.4" in config.load_path:
        settings.append("m-0.4")
    elif "m-0.6" in config.load_path:
        settings.append("m-0.6")

else:
    settings.append("imagenet")


logging.info(f"saving to {config.ckpt_path}")

Path(config.log_dir).mkdir(parents=True, exist_ok=True)
writer = SummaryWriter(log_dir=config.log_dir)

device = torch.device("cuda" if cuda.is_available() else "cpu")

logging.info("loading model")
nima = NIMA(config.load_path).to(device)

# fmt:off
optimizer = optim.Adam(
[
    {"params": nima.features.parameters(), "lr": 0, "initial_lr":0},
    {"params": nima.classifier.parameters(), "lr": 0.001, "initial_lr":0.001}
],
lr=0,
weight_decay=0.00004)
# fmt:on
lr_scheduler: optim.lr_scheduler.ReduceLROnPlateau = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, factor=0.5, patience=10
)

# if settings[0] is not None:
#    config.dense_lr = 0.001
#    config.conv_lr = 0.0001

scaler = cuda.amp.GradScaler()

epoch = 0
g_step = 0

conv_locked = True
new_lr_init = True

# loading checkpoints, ... or not
warm_epoch = 0
logging.info("checking for checkpoints")
if Path(config.ckpt_path).exists():
    logging.info("loading checkpoints")
    if not (Path(config.ckpt_path) / "epoch-0.pth").exists():
        logging.info("none found")
    else:
        for warm_epoch in range(0, 80):
            p = Path(config.ckpt_path) / f"epoch-{warm_epoch}.pth"
            if not (Path(config.ckpt_path) / f"epoch-{warm_epoch+1}.pth").exists():
                break
        state = torch.load(str(p))

        nima.load_state_dict(state["model_state"])
        optimizer.load_state_dict(state["optimizer_state"])
        lr_scheduler.load_state_dict(state["scheduler_state"])
        scaler.load_state_dict(state["scaler_state"])

        epoch = state["epoch"] + 1
        g_step = state["g_step"]

        conv_locked = state["conv_locked"]
        new_lr_init = state["new_lr_init"]

        logging.info(f"Successfully loaded model {p}")

logging.info("setting learnrates")


# counting parameters
param_num = 0
for param in nima.parameters():
    param_num += int(np.prod(param.shape))
logging.info(f"trainable params: {(param_num / 1e6):.2f} million")

logging.info("creating datasets")
train_loader = DataLoader(
    AVA(mode="train", percentage_of_dataset=config.percentage_of_dataset),
    batch_size=450,
    shuffle=True,
    drop_last=True,
    num_workers=50,
    pin_memory=True,
)
val_loader = DataLoader(AVA(mode="val"), batch_size=450, shuffle=False, drop_last=True, num_workers=50, pin_memory=True)
logging.info("datasets created")

logging.info("start training")
for epoch in range(epoch, 80):
    if conv_locked:
        for param in nima.features.parameters():
            param.requires_grad = False
    else:
        for param in nima.parameters():
            param.requires_grad = True

    if not conv_locked and new_lr_init:
        optimizer = optim.Adam(
            [
                {"params": nima.features.parameters(), "lr": config.conv_lr, "initial_lr": config.conv_lr},
                {"params": nima.classifier.parameters(), "lr": config.dense_lr, "initial_lr": config.dense_lr},
            ],
            lr=0,
            weight_decay=0.00004,
        )

        lr_scheduler: optim.lr_scheduler.ReduceLROnPlateau = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, factor=0.5, patience=5
        )
        new_lr_init = False

    for i, data in enumerate(train_loader):
        logging.info(f"batch loaded: step {i}")

        optimizer.zero_grad()
        # forward pass + loss calculation
        with cuda.amp.autocast():
            loss = earth_movers_distance(data["y_true"].to(device), nima(data["img"].to(device)))

        logging.info(f"Epoch: {epoch} | Step: {i}/{len(train_loader)}")

        # optimizing
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        writer.add_scalar("loss/train", loss.item(), g_step)
        writer.add_scalar("progress/epoch", epoch, g_step)
        writer.add_scalar("progress/step", i, g_step)
        writer.add_scalar("hparams/conv_lr", optimizer.param_groups[0]["lr"], g_step)
        writer.add_scalar("hparams/dense_lr", optimizer.param_groups[1]["lr"], g_step)

        g_step += 1
        logging.info("waiting for new batch")

    logging.info("validating")
    val_loss = []
    for i, data in enumerate(val_loader):
        with cuda.amp.autocast():
            with torch.no_grad():
                loss = earth_movers_distance(data["y_true"].to(device), nima(data["img"].to(device)))
        val_loss.append(loss)

    val_loss = sum(val_loss) / len(val_loss)
    writer.add_scalar("loss/val", val_loss.item(), g_step)

    # learning rate decay:
    lr_scheduler.step(val_loss.item())

    if (
        conv_locked and optimizer.param_groups[1]["lr"] != optimizer.param_groups[1]["initial_lr"]
    ):  #  learn rate scheduler would have stepped, so activate everything
        logging.info("unlocking cnn")
        conv_locked = False
        new_lr_init = True

logging.info("saving!")
Path(config.ckpt_path).mkdir(parents=True, exist_ok=True)
state = {
    "model_state": nima.state_dict(),
    "optimizer_state": optimizer.state_dict(),
    "scheduler_state": lr_scheduler.state_dict(),
    "scaler_state": scaler.state_dict(),
    "epoch": epoch,
    "g_step": g_step,
    "conv_locked": conv_locked,
    "new_lr_init": new_lr_init,
}
torch.save(state, str(Path(config.ckpt_path) / f"model.pth"))

logging.info("Training complete!")
writer.close()
