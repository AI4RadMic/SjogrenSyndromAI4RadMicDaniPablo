from src.model.feed_forward import FeedForwardNN
from src.model.convnet import ConvNet
from src.model.transfer_learning import TL_ResNet
import torch

def build_model(config):
    if config.type == "FeedForwardNN":
        model = FeedForwardNN(**config.model_config)
    elif config.type == "ConvNet":
        model = ConvNet(**config.model_config)
    elif config.type == "TL_ResNet":
        model = TL_ResNet(**config.model_config)

    return model