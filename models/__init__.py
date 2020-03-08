import torch.nn as nn
from importlib import import_module


class Model(nn.Module):
    def __init__(self, args):
        super(Model, self).__init__()
        module = import_module('models.' + args.model.lower())
        self.model = module.make_model(args)

    def forward(self, frame0, frame1):
        return self.model(frame0, frame1)

    def load(self, state_dict):
        self.model.load_state_dict(state_dict)

    def get_state_dict(self):
        return self.model.state_dict()

    def get_kernel(self, frame0, frame1):
        return self.model.get_kernel(frame0, frame1)
