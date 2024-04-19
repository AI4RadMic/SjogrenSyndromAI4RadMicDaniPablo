from src.model.feed_forward import FeedForwardNN
from src.model.convnet import ConvNet
import torch

def build_model(config):
    if config.type == "FeedForwardNN":
        model = FeedForwardNN(**config.model_config)
    elif config.type == "ConvNet":
        model = ConvNet(**config.model_config)

    return model