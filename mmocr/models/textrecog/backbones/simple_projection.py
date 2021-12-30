import torch
import torch.nn as nn
import torch.nn.functional as F

from mmdet.models.builder import BACKBONES
@BACKBONES.register_module()
class SimpleProjection(nn.Module):
    def __init__(self, input_format = 'RGB', downsample = (32,32)):
        super(SimpleProjection, self).__init__()
        self.input_format = input_format
        kernel_size = downsample
        stride = downsample
        self.conv = nn.Conv2d(3, 512, kernel_size=kernel_size, stride=stride, padding=0, bias=False)

    def forward(self, x):
        '''
        Args: x (B, C=3, H, W)
        Return: f [(B, C = 512, H/32, W/32)]
        
        '''
        f = []
        if self.input_format == "BGR":
            # RGB->BGR, images are read in as RGB by default
            x = x[:, [2, 1, 0], :, :]
        x = self.conv(x)
        f.append(x)
        return f

if __name__ == '__main__':
    net = SimpleProjection(input_format = 'RGB', downsample = (32,32))
    input = torch.randn(4,3,256,256)
    output = net(input)[-1]
    print(output.size())
