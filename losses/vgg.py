import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision


class VGG(nn.Module):
    def __init__(self):
        super(VGG, self).__init__()
        vgg16 = torchvision.models.vgg16(pretrained=True)
        self.vgg16_conv_4_3 = torch.nn.Sequential(*list(vgg16.children())[0][:22])
        for param in self.vgg16_conv_4_3.parameters():
            param.requires_grad = False

    def forward(self, output, gt):
        vgg_output = self.vgg16_conv_4_3(output)
        with torch.no_grad():
            vgg_gt = self.vgg16_conv_4_3(gt.detach())

        loss = F.mse_loss(vgg_output, vgg_gt)

        return loss
