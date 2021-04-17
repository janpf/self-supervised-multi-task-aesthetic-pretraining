import argparse
import logging
import sys
from pathlib import Path

import pandas as pd
import torch
from torch import cuda

sys.path[0] = "/workspace"
from train_pre.dataset import SSPexelsSmall as SSPexels
from train_fine.NIMA import NIMA
from train_pre.utils import mapping

logging.basicConfig(format="%(asctime)s %(levelname)-8s %(message)s", level=logging.INFO, datefmt="%Y-%m-%d %H:%M:%S")
# logging.getLogger("PIL.PngImagePlugin").setLevel(logging.INFO)

parser = argparse.ArgumentParser()

parser.add_argument("--model-path", type=str, default=None)
config = parser.parse_args()


test_file = "/workspace/dataset_processing/test_set.txt"
out_file = "/workspace/analysis/not_uploaded/IA2NIMA/" + config.model_path.replace("/", ".") + ".txt"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

logging.info("loading model")

nima = NIMA().to(device)
nima.load_state_dict(torch.load(config.model_path)["model_state"])
nima.eval()

logging.info("creating datasets")
# datasets
SSPexels_test = SSPexels(file_list_path=test_file, mapping=mapping, return_file_name=True)
Pexels_test = torch.utils.data.DataLoader(SSPexels_test, batch_size=5, drop_last=False, num_workers=40)
logging.info("datasets created")

logging.info("testing")

output = []

for i, data in enumerate(Pexels_test):
    logging.info(f"{i}/{len(Pexels_test)}")

    for key in data.keys():
        if key == "file_name":
            continue

        img = data[key].to(device)
        with cuda.amp.autocast():
            with torch.no_grad():
                out = nima(img)

        for p, s in zip(data["file_name"], out.tolist()):
            if key == "original":
                key = "original;0"
            if key == "rotate_original":
                key = "rotate;0"
            if key == "crop_original":
                key = "crop;0"

            out_dict = dict()
            out_dict["img"] = p
            out_dict["distortion"] = key.split(";")[0]
            out_dict["level"] = key.split(";")[1]
            out_dict["scores"] = s
            output.append(out_dict)

logging.info(f"writing {out_file}")
pd.DataFrame(output).to_csv(out_file, index=False)
