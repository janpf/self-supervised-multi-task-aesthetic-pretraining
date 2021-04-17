import argparse
import logging
import sys
from pathlib import Path

import pandas as pd
import torch
from torch import cuda

sys.path[0] = "/workspace"
from train_fine.dataset import AVA
from train_fine.NIMA import NIMA

logging.basicConfig(format="%(asctime)s %(levelname)-8s %(message)s", level=logging.INFO, datefmt="%Y-%m-%d %H:%M:%S")
# logging.getLogger("PIL.PngImagePlugin").setLevel(logging.INFO)

parser = argparse.ArgumentParser()

parser.add_argument("--model-path", type=str, default=None)
parser.add_argument("--last-frozen", action="store_true")
parser.add_argument("--first-unfrozen", action="store_true")

config = parser.parse_args()

if config.last_frozen and config.first_unfrozen:
    exit("what")

if config.last_frozen:
    for p in sorted(Path(config.model_path).glob("*.pth"), key=lambda x: int(x.stem.split("epoch-")[1])):
        if not torch.load(p)["conv_locked"] and torch.load(p)["new_lr_init"]:
            config.model_path = p
            break

if config.first_unfrozen:
    for p in sorted(Path(config.model_path).glob("*.pth"), key=lambda x: int(x.stem.split("epoch-")[1])):
        if not torch.load(p)["conv_locked"] and not torch.load(p)["new_lr_init"]:
            config.model_path = p
            break

config.model_path = str(config.model_path)

logging.info(f"loading {config.model_path}")

out_file = "/workspace/analysis/not_uploaded/IA2NIMA/AVA/" + config.model_path.replace("/", ".") + ".txt"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

logging.info("loading model")

nima = NIMA().to(device)
nima.load_state_dict(torch.load(config.model_path)["model_state"])
nima.eval()

logging.info("creating datasets")
# datasets
AVA = AVA(mode="test", horizontal_flip=False)
AVA_test = torch.utils.data.DataLoader(AVA, batch_size=450, drop_last=False, num_workers=50, pin_memory=True)
logging.info("datasets created")

logging.info("testing")

output = []

for i, data in enumerate(AVA_test):
    logging.info(f"{i}/{len(AVA_test)}")

    with cuda.amp.autocast():
        with torch.no_grad():
            out = nima(data["img"].to(device))

    for p, s in zip(data["path"], out.tolist()):
        out_dict = dict()
        out_dict["img"] = p
        out_dict["scores"] = s
        output.append(out_dict)

logging.info(f"writing {out_file}")
pd.DataFrame(output).to_csv(out_file, index=False)
