import pyuTensor as ut
import numpy as np
import torch

class ConvLayer:
    def __init__(self, weight, bias, padding="SAME", strides=None):
        self.weight = weight
        self.bias = bias
        self.padding = padding
        self.strides = strides or [1, 1, 1, 1]

    def __call__(self, x):
        return ut.conv2d_f(x, filter=self.weight, bias=self.bias, padding=self.padding, strides=self.strides)

class uSuperPoint:
    def __init__(self, state_dict_path: str):
        self._state_dict = {
            k: v.detach().cpu().numpy() for k, v in torch.load(state_dict_path).items()
        }
        for k, v in self._state_dict.items():
            if k.endswith(".weight"):
                self._state_dict[k] = v.transpose([0, 2, 3, 1])

    def __call__(self, img: np.ndarray):
        x = img
        for name in ["conv1a",
            "conv1b",
            "conv2a",
            "conv2b",
            "conv3a",
            "conv3b",
            "conv4a",
            "conv4b",]:
            w = self._state_dict[f"{name}.weight"]
            b = self._state_dict[f"{name}.bias"]
            x = ut.conv2d_f(x, filter=w, bias=b, padding="SAME")
        return x
