import torch.nn as nn
def inti(m):
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu'); 
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)