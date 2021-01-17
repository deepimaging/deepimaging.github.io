import torch.nn as nn
import torch

from modules.image_processing import ImageEncoder
from modules.physical_layers import Illumination, FixedIllumination
from modules.transformers import TransformerEmbed
from modules.decoders import ClassificationDecoder, PhysicalDecoder, DecisionDecoder


class TransformedIlluminator(nn.Module):

    def __init__(self, d_model=64, num_leds=25, image_size=28, num_heads=2, num_classes=None, std=0.17):
        super().__init__()
        self.device = 'cpu'
        self.num_leds = num_leds
        self.counter = 0
        self.illuminator = Illumination(num_leds=num_leds)
        #self.illuminator = FixedIllumination(num_leds=num_leds)
        # d_model += num_leds % 2
        self.image_encoder = ImageEncoder(image_size=image_size, mlp_size=d_model, channels_in=1)
        # d_model += num_leds
        self.transformer = TransformerEmbed(num_heads=num_heads, d_model=d_model, num_layers=4,
                                            feedforward_dim=d_model, reduction='mean')
        self.decision_decoder = DecisionDecoder(d_model=d_model)
        self.physical_decoder = PhysicalDecoder(d_model=d_model, num_leds=num_leds, std=std)
        self.classification_decoder = ClassificationDecoder(d_model=d_model, num_classes=num_classes)

    def forward(self, x, phi, zs=None):
        # x is a single image, phi is an illumination pattern, zs is the previous encodings
        image, phi = self.illuminator(x, phi)
        # image, phi = self.illuminator(x, self.counter)
        enc = self.image_encoder(image)
        z = enc
        output_embedding = self.transformer(z, zs)
        # check decision
        decision = self.decision_decoder(output_embedding)
        classification = self.classification_decoder(output_embedding)
        new_phi = self.physical_decoder(output_embedding)
        self.counter += 1
        self.counter = self.counter % 3
        return decision, classification, new_phi, z, image

    def cuda(self, **kwargs):
        self.device = 'cuda'
        self.illuminator.cuda()
        self.transformer.cuda()
        self.decision_decoder.cuda()
        self.physical_decoder.cuda()
        self.classification_decoder.cuda()
        super(TransformedIlluminator, self).cuda()

    def get_parameters(self):
        params = []
        # params += [self.first_illumination]
        params += list(self.illuminator.parameters())
        params += list(self.transformer.parameters())
        params += list(self.decision_decoder.parameters())
        params += list(self.physical_decoder.parameters())
        params += list(self.classification_decoder.parameters())
        return params


class RecurrentIlluminator(nn.Module):

    def __init__(self, d_model=64, num_leds=25, image_size=28, num_heads=2, num_classes=None, std=0.17):
        super().__init__()
        self.device = 'cpu'
        self.num_leds = num_leds
        self.illuminator = Illumination(num_leds=num_leds)
        self.image_encoder = ImageEncoder(image_size=image_size, mlp_size=d_model, channels_in=1)
        # d_model += num_leds % 2
        # d_model += num_leds
        self.rnn = nn.RNNCell(d_model, d_model)
        self.decision_decoder = DecisionDecoder(d_model=d_model)
        self.physical_decoder = PhysicalDecoder(d_model=d_model, num_leds=num_leds, std=std)
        self.classification_decoder = ClassificationDecoder(d_model=d_model, num_classes=num_classes)

    def forward(self, x, phi, h=None):
        # x is a single image, phi is an illumination pattern, zs is the previous encodings
        image, phi = self.illuminator(x, phi)
        enc = self.image_encoder(image)
        # z = torch.cat([enc, phi], dim=-1)
        z = enc
        new_hidden_state = self.rnn(z.squeeze(), h)
        decision = self.decision_decoder(new_hidden_state)
        # check decision
        classification = self.classification_decoder(new_hidden_state)
        new_phi, log_pi = self.physical_decoder(new_hidden_state)
        return decision, classification, new_phi, new_hidden_state, image

    def cuda(self, **kwargs):
        self.device = 'cuda'
        self.illuminator.cuda()
        self.rnn.cuda()
        self.decision_decoder.cuda()
        self.physical_decoder.cuda()
        self.classification_decoder.cuda()
        super(RecurrentIlluminator, self).cuda()
