from torch.nn import Linear
import torch

class Classifier(torch.nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        torch.manual_seed(12345)
        self.lin = Linear(768, 2)

    def forward(self, x):
        x = self.lin(x)
        return x
