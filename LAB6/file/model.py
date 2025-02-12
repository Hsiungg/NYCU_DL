from diffusers import UNet2DModel
import torch
import torchvision
from torch import nn
#ref https://github.com/huggingface/diffusion-models-class/blob/main/unit2/02_class_conditioned_diffusion_model_example.ipynb

class ClassConditionalDDPM(nn.Module):
    def __init__(self, num_classes=24, class_emb_size=4):
        super().__init__()
        
        self.class_emb = nn.Embedding(num_classes, class_emb_size)
        self.model = UNet2DModel(
        sample_size = 64,          
        in_channels = 3 + class_emb_size * num_classes, # Additional input channels for class cond.
        out_channels = 3,           # the number of output channels
        layers_per_block=2,       # how many ResNet layers to use per UNet block
        block_out_channels=(128, 256, 512, 1024), 
        down_block_types = ("DownBlock2D", "AttnDownBlock2D", "AttnDownBlock2D", "AttnDownBlock2D"),
        up_block_types= ("AttnUpBlock2D", "AttnUpBlock2D", "AttnUpBlock2D", "UpBlock2D"),
        )
    def forward(self, x, t, class_labels):
        bs, c, w, h = x.shape
        class_cond = self.class_emb(class_labels)
        class_cond = class_cond.view(bs, -1, 1, 1).expand(bs, -1, w, h)
        input = torch.cat((x, class_cond), 1)
        
        return self.model(input, t).sample
        
        