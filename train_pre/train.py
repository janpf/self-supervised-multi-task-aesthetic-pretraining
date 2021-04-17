import argparse
import logging
import sys
import math
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from torch import cuda, optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

sys.path[0] = "/workspace"
from train_pre.dataset import SSPexelsSmall as SSPexels
from train_pre.IA import IA
from train_pre.utils import mapping

parser = argparse.ArgumentParser()

# training parameters
parser.add_argument("--lr", type=float, default=0.0001)
parser.add_argument("--margin", type=float, default=0.2)
parser.add_argument("--lr_decay_rate", type=float, default=0.9)
parser.add_argument("--train_batch_size", type=int, default=5)
parser.add_argument("--num_workers", type=int, default=40)

parser.add_argument("--scores", type=str, default=None)  # "one", "three", None; None is a valid input
parser.add_argument("--change_regress", action="store_true")
parser.add_argument("--change_class", action="store_true")

# misc
parser.add_argument("--log_dir", type=str, default="/scratch/train_logs/IA/pexels/")
parser.add_argument("--ckpt_path", type=str, default="/scratch/ckpts/IA/pexels/")

config = parser.parse_args()

settings = [f"scores-{config.scores}"]

if config.change_regress:
    settings.append("change_regress")

if config.change_class:
    settings.append("change_class")

if not math.isclose(config.margin, 0.2):
    settings.append(f"m-{config.margin}")

for s in settings:
    config.log_dir = str(Path(config.log_dir) / s)
    config.ckpt_path = str(Path(config.ckpt_path) / s)

margin = dict()
margin["styles"] = config.margin
margin["technical"] = config.margin
margin["composition"] = config.margin

logging.basicConfig(format="%(asctime)s %(levelname)-8s %(message)s", level=logging.INFO, datefmt="%Y-%m-%d %H:%M:%S")
# logging.getLogger("PIL.PngImagePlugin").setLevel(logging.INFO)

Path(config.log_dir).mkdir(parents=True, exist_ok=True)
writer = SummaryWriter(log_dir=config.log_dir)

device = torch.device("cuda" if cuda.is_available() else "cpu")

logging.info("loading model")
ia = IA(
    scores=config.scores,
    change_regress=config.change_regress,
    change_class=config.change_class,
    mapping=mapping,
    margin=margin,
    pretrained=True,
).to(device)

# loading checkpoints, ... or not
warm_epoch = 0
logging.info("checking for checkpoints")
if Path(config.ckpt_path).exists():
    logging.info("loading checkpoints")
    if not (Path(config.ckpt_path) / "epoch-1.pth").exists():
        logging.info("none found")
    else:
        for warm_epoch in range(1, 100):
            p = Path(config.ckpt_path) / f"epoch-{warm_epoch}.pth"
            if not (Path(config.ckpt_path) / f"epoch-{warm_epoch+1}.pth").exists():
                break
        ia.load_state_dict(torch.load(str(p)))
        logging.info(f"Successfully loaded model {p}")

logging.info("setting learnrates")

# fmt:off
optimizer = optim.RMSprop(
    [
        {"params": ia.parameters(), "lr": config.lr}],
        momentum=0.9,
        weight_decay=0.00004,
)
# fmt:on
lr_scheduler = optim.lr_scheduler.MultiplicativeLR(optimizer, lambda epoch: config.lr_decay_rate ** (epoch + 1))
scaler = cuda.amp.GradScaler()

# counting parameters
param_num = 0
for param in ia.parameters():
    param_num += int(np.prod(param.shape))
logging.info(f"trainable params: {(param_num / 1e6):.2f} million")

logging.info("creating datasets")
SSPexels_train = SSPexels(file_list_path="/workspace/dataset_processing/train_set.txt", mapping=mapping)
Pexels_train_loader = DataLoader(
    SSPexels_train, batch_size=config.train_batch_size, shuffle=True, drop_last=True, num_workers=config.num_workers
)
logging.info("datasets created")

if warm_epoch > 0:
    g_step = warm_epoch * len(Pexels_train_loader)
    for _ in range(warm_epoch):
        lr_scheduler.step()
else:
    g_step = 0


logging.info("start training")
for epoch in range(warm_epoch + 1, 50):
    for i, data in enumerate(Pexels_train_loader):

        if g_step <= 200:
            for param in ia.features.parameters():
                param.requires_grad = False
        elif g_step <= 300:
            for param in ia.parameters():
                param.requires_grad = True

        logging.info(f"batch loaded: step {i}")

        optimizer.zero_grad()
        # forward pass + loss calculation
        with cuda.amp.autocast():
            losses = ia.calc_loss(data, T)

        for k, loss in losses.items():
            writer.add_scalar(f"loss/train/balanced/{k}", loss.item(), g_step)
        loss = sum([v for _, v in losses.items()])
        writer.add_scalar(f"loss/train/balanced/overall", loss.item(), g_step)

        logging.info(f"Epoch: {epoch} | Step: {i}/{len(Pexels_train_loader)} | Training loss: {loss.item()}")

        # optimizing
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        writer.add_scalar("progress/epoch", epoch, g_step)
        writer.add_scalar("progress/step", i, g_step)
        writer.add_scalar("hparams/lr", optimizer.param_groups[0]["lr"], g_step)
        writer.add_scalar("hparams/margin/styles", float(margin["styles"]), g_step)
        writer.add_scalar("hparams/margin/technical", float(margin["technical"]), g_step)
        writer.add_scalar("hparams/margin/composition", float(margin["composition"]), g_step)

        g_step += 1
        logging.info("waiting for new batch")

    # learning rate decay:
    lr_scheduler.step()

    logging.info("saving!")
    Path(config.ckpt_path).mkdir(parents=True, exist_ok=True)
    torch.save(ia.state_dict(), str(Path(config.ckpt_path) / f"epoch-{epoch}.pth"))

logging.info("Training complete!")
writer.close()
