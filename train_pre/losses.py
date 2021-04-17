import logging
from typing import List

import torch
import torch.nn
from torch import cuda


class EfficientRankingLoss(torch.nn.Module):
    def __init__(self, margin: float):
        super(EfficientRankingLoss, self).__init__()
        self.mrloss = torch.nn.MarginRankingLoss(margin)
        self.one = torch.Tensor([1]).to(torch.device("cuda" if cuda.is_available() else "cpu"))

    def forward(self, original, x, polarity: str, score: str) -> torch.Tensor:
        loss: List[torch.Tensor] = []
        for idx1, change1 in enumerate(x.keys()):
            logging.debug(f"score\toriginal\t{change1}")
            loss.append(self.mrloss(original[score], x[change1][score], self.one))
            for idx2, change2 in enumerate(x.keys()):
                if idx1 >= idx2:
                    continue
                if polarity == "pos":
                    logging.debug(f"score\t{change1}\t{change2}")
                    loss.append(self.mrloss(x[change1][score], x[change2][score], self.one))
                elif polarity == "neg":
                    logging.debug(f"score\t{change2}\t{change1}")
                    loss.append(self.mrloss(x[change2][score], x[change1][score], self.one))
        return sum(loss)


def h(z: torch.Tensor, T: int = 50):
    """loss balancing function: https://arxiv.org/pdf/2002.04792.pdf"""
    return torch.exp(z / T)
