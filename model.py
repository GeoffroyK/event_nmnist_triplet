import torch 
import torch.nn as nn
import torch.nn.functional as F
from spikingjelly.clock_driven import neuron, surrogate

class SNNEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(SNNEncoder, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(input_dim, 32, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
            neuron.IFNode(surrogate_function=surrogate.ATan())
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
            neuron.IFNode(surrogate_function=surrogate.ATan())
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
            neuron.IFNode(surrogate_function=surrogate.ATan())
        )

        self.fc = nn.Linear(128 * 4 * 4, 128)
        self.sn_final = neuron.IFNode(surrogate_function=surrogate.ATan(), v_threshold=float("inf"), v_reset=0.)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        x = self.sn_final(x)
        x = self.sn_final.v.detach()
        return x
