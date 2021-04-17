# https://github.com/kentsyx/Neural-IMage-Assessment/blob/master/model/model.py

import torch
import torch.nn as nn
import torchvision
from train_pre.IA import CheckpointModule


class NIMA(nn.Module):
    """Neural IMage Assessment model by Google"""

    def __init__(self, load_path: str = None):
        super(NIMA, self).__init__()
        self.load_path = load_path
        self.feature_count = 1280

        if "SimClR" in load_path:
            from train_pre.IA import IA
            from relatedWorks.SimCLR.SimCLR.simclr import SimCLR

            encoder = IA(
                scores=False,
                change_regress=False,
                change_class=False,
                mapping=None,
                margin=0,
                pretrained=False,
            ).features
            n_features = 1280  # get dimensions of fc layer

            # initialize model
            model = SimCLR(encoder, 64, n_features)
            model.load_state_dict(torch.load(load_path, map_location="cpu"))
            self.features = model.encoder

        elif "rotnet" in load_path:
            from relatedWorks.RotNet.FeatureLearningRotNet.architectures.MobileNet import MobileNet

            model = MobileNet({"num_classes": 4, "pretrained": False})
            model.load_state_dict(torch.load(str(load_path))["network"])
            self.features = nn.Sequential(*[m for m in model._feature_blocks[0:-1]])
            self.features = CheckpointModule(module=self.features, num_segments=len(self.features))

        self.classifier = nn.Sequential(
            nn.Dropout(p=0.75), nn.Linear(in_features=self.feature_count, out_features=10), nn.Softmax()
        )

        print(self.features)
        print(self.classifier)

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
