import torch.nn as nn
import torch
import torch.nn.functional as F


class Illumination(nn.Module):
    """
    Module to perform weighted illumination using an input image and led weights
    no learnable parameters
    """

    def __init__(self, num_leds = 25):
        super().__init__()
        self.image_shape = [1, 28, 28]
        self.default_illumination = nn.Sequential(nn.Conv2d(num_leds, 1, 1, bias=False))
        self.num_leds = num_leds

    def forward(self, images, channel_weights):
        # images are of shape: B x C x (H x W)
        # led_weights are of shape: B x num_leds
        # channel weights are: B x C x 1
        batch_size = images.shape[0]
        if channel_weights is not None:
            led_weights = channel_weights.view(-1, self.num_leds, 1, 1)
            led_weights = torch.tanh(torch.square(led_weights))
            images_grouped = images.view(1, batch_size*self.num_leds, images.shape[2], images.shape[3])
            weights_grouped = led_weights.view(batch_size*1, self.num_leds, 1, 1)
            intermediate = F.conv2d(images_grouped, weights_grouped, groups=batch_size)
            output = intermediate.view(batch_size, 1, intermediate.shape[2], intermediate.shape[3])
            return output, led_weights.view(-1, 1, self.num_leds)
        else:
            led_weights = self.default_illumination[0].weight.view(1, 1, self.num_leds).expand(len(images), 1, self.num_leds)
            output = self.default_illumination(images)
            return output, led_weights


class FixedIllumination(nn.Module):
    """
    Module to perform weighted illumination using an input image and led weights
    no learnable parameters
    """

    def __init__(self, num_leds = 25):
        super().__init__()
        self.image_shape = [1, 28, 28]
        self.default_illumination = nn.Sequential(nn.Conv2d(num_leds, 1, 1, bias=False))
        self.ill_1 = nn.Sequential(nn.Conv2d(num_leds, 1, 1, bias=False))
        self.ill_2 = nn.Sequential(nn.Conv2d(num_leds, 1, 1, bias=False))
        self.default_illumination_modules = [nn.Sequential(nn.Conv2d(num_leds, 1, 1, bias=False)) for _ in range(3)]
        self.num_leds = num_leds

    def forward(self, images, iter):
        # images are of shape: B x C x (H x W)
        # led_weights are of shape: B x num_leds
        # channel weights are: B x C x 1
        if iter == 0:
            led_weights = self.default_illumination[0].weight.view(1, 1, self.num_leds).expand(len(images), 1, self.num_leds)
            output = self.default_illumination(images)
        elif iter == 1:
            led_weights = self.ill_1[0].weight.view(1, 1, self.num_leds).expand(len(images), 1,
                                                                                               self.num_leds)
            output = self.ill_1(images)
        else:
            led_weights = self.ill_2[0].weight.view(1, 1, self.num_leds).expand(len(images), 1,
                                                                                               self.num_leds)
            output = self.ill_2(images)
        return output, led_weights
