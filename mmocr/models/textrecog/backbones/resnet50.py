import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision import models

from mmdet.models.builder import BACKBONES


@BACKBONES.register_module()
class PretrainedResNet50(nn.Module):
    def __init__(self, input_format = 'RGB', checkpoint = None):
        super(PretrainedResNet50, self).__init__()
        self.model = models.resnet50(pretrained=False)
        self.input_format = input_format
        if checkpoint is not None:
            self.model.load_state_dict(torch.load(checkpoint))
            print('loaded resnet50 checkpoint!')
        self.grid_encoder = nn.Sequential(
            nn.Conv2d(2048, 512, kernel_size=3, stride=1, padding=1, bias=False),
            # nn.MaxPool2d(kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
        )
        
    def forward(self, x):
        '''
        Args: x (B, C=3, H, W)
        Return: f [(B, C = 2048, H/32, W/32), (B, C = 512, H/32, W/32)]
        
        '''
        f = []
        if self.input_format == "BGR":
            # RGB->BGR, images are read in as RGB by default
            x = x[:, [2, 1, 0], :, :]
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)
        f.append(x)
        x = self.grid_encoder(x)
        f.append(x)
        return f

if __name__ == '__main__':
    net = PretrainedResNet50(input_format='RGB',checkpoint='expr_result/pretrained_resnet50_checkpoint/resnet50-0676ba61.pth')
    input = torch.randn(4,3,256,256)
    output = net(input)
    target = torch.randn(4, 512, 8, 8)
    loss = F.mse_loss(target, output[-1])
    
    loss.backward()
    