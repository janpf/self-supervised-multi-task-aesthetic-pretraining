# https://github.com/kentsyx/Neural-IMage-Assessment/blob/master/model/model.py

import torch
import torch.nn as nn
import torchvision

from train_pre.IA import IA
from train_pre.utils import mapping


class NIMA(nn.Module):
    """Neural IMage Assessment model by Google"""

    def __init__(self, load_path: str = None):
        super(NIMA, self).__init__()
        self.load_path = load_path
        self.feature_count = 1280

        if load_path is not None:
            score = None
            if "scores-one" in load_path:
                score = "one"
            elif "scores-three" in load_path:
                score = "three"

            ia = IA(score, "change_regress" in load_path, "change_class" in load_path, mapping, None)
            ia.load_state_dict(torch.load(str(load_path)))
            self.features = ia.features
        else:
            self.features = IA(None, False, False, mapping, None, pretrained=True).features
        # fmt: off
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.75),
            nn.Linear(in_features=self.feature_count, out_features=10),
            nn.Softmax())
        # fmt: on

    def forward(self, x):
        out = self.features(x)
        out = nn.functional.adaptive_avg_pool2d(out, 1).reshape(out.shape[0], -1)
        out = self.classifier(out)
        return out


def earth_movers_distance(y: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
    cdf_y = torch.cumsum(y, dim=1)
    cdf_pred = torch.cumsum(y_pred, dim=1)
    emd_loss = torch.sqrt(torch.mean(torch.square(cdf_pred - cdf_y)))
    return emd_loss.mean()
