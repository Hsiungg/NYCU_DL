import torch 
import torch.nn as nn
import yaml
import os
import math
import numpy as np
import random
import torch.nn.functional as F
from .VQGAN import VQGAN
from .Transformer import BidirectionalTransformer



#TODO2 step1: design the MaskGIT model
class MaskGit(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.vqgan = self.load_vqgan(configs['VQ_Configs'])
    
        self.num_image_tokens = configs['num_image_tokens']
        self.mask_token_id = configs['num_codebook_vectors']
        self.choice_temperature = configs['choice_temperature']
        self.gamma = self.gamma_func(configs['gamma_type'])
        self.transformer = BidirectionalTransformer(configs['Transformer_param'])

    def load_transformer_checkpoint(self, load_ckpt_path):
        checkpoint = torch.load(load_ckpt_path)
        state_dict = {k: v for k, v in checkpoint.items() if k not in ['epoch', 'model_state_dict', 'optimizer_state_dict', 'lr']}

        if 'transformer' in state_dict:
            self.transformer.load_state_dict(state_dict['transformer'])
    
        print("Checkpoint loaded successfully.")

    @staticmethod
    def load_vqgan(configs):
        cfg = yaml.safe_load(open(configs['VQ_config_path'], 'r'))
        model = VQGAN(cfg['model_param'])
        model.load_state_dict(torch.load(configs['VQ_CKPT_path']), strict=True ) 
        model = model.eval()
        return model
    
##TODO2 step1-1: input x fed to vqgan encoder to get the latent and zq
    @torch.no_grad()
    def encode_to_z(self, x):
        z_q, z_idx, loss = self.vqgan.encode(x)
        z_idx = z_idx.view(z_q.shape[0], -1)

        return z_q, z_idx
    
##TODO2 step1-2:    
    def gamma_func(self, mode="cosine"):
        """Generates a mask rate by scheduling mask functions R.

        Given a ratio in [0, 1), we generate a masking ratio from (0, 1]. 
        During training, the input ratio is uniformly sampled; 
        during inference, the input ratio is based on the step number divided by the total iteration number: t/T.
        Based on experiements, we find that masking more in training helps.
        
        ratio:   The uniformly sampled ratio [0, 1) as input.
        Returns: The mask rate (float).

        """
        assert mode in ["linear", "cosine", "square"]

        if mode == "linear":
            return lambda ratio: 1 - ratio
        elif mode == "cosine":
            return lambda ratio: np.cos(ratio * np.pi / 2)
        elif mode == "square":
            return lambda ratio: 1 - ratio ** 2
        else:
            raise NotImplementedError

##TODO2 step1-3:            
    def forward(self, x):
        _, z_indices =  self.encode_to_z(x)#ground truth
        #z_indices in shape (B, 256)
        ratio = random.uniform(0.2, 0.6)
        mask = torch.zeros_like(z_indices, dtype=torch.bool).to(z_indices.device)
        for i in range(z_indices.shape[0]):
            mask[i] = torch.rand(z_indices.shape[1]) < ratio
            
        masked_z_indices = z_indices.masked_fill(mask, self.mask_token_id)
        
        logits = self.transformer(masked_z_indices)
        return logits, z_indices
        
##TODO3 step1-1: define one iteration decoding   
    @torch.no_grad()
    def inpainting(self, z_indices, step_ratio, mask, mask_num):
        #mask = True if it need mask
        masked_indices = z_indices.masked_fill(mask, self.mask_token_id)
        
        logits = self.transformer(masked_indices)
        #Apply softmax to convert logits into a probability distribution across the last dimension.
        pred_prob = F.softmax(logits, dim= -1)
        #shape(B, 256, 1024)
        
        #FIND MAX probability for each token value
        z_indices_predict_prob, z_indices_predict = torch.max(pred_prob, dim =-1)
        #(B, 256)
        z_indices_predict = torch.where(~mask, z_indices, z_indices_predict)
        
        #predicted probabilities add temperature annealing gumbel noise as confidence
        g = torch.distributions.Gumbel(0, 1).sample(z_indices_predict_prob.shape).to(mask.device)  # gumbel noise
        temperature = self.choice_temperature * (1 - step_ratio)
        confidence = z_indices_predict_prob + temperature * g
        
        #hint: If mask is False, the probability should be set to infinity, so that the tokens are not affected by the transformer's prediction
        #sort the confidence for the rank 
        #define how much the iteration remain predicted tokens by mask scheduling
        #At the end of the decoding process, add back the original token values that were not masked to the predicted tokens
        mask_ratio = math.floor(self.gamma(step_ratio) * mask_num)
        confidence[~mask] = torch.inf
        _, idx_mask = confidence.topk(mask_ratio, dim=-1, largest=False)
        #mask the smallest k confident to mask back
        mask_bc= torch.zeros(z_indices.shape, dtype=torch.bool, device= mask.device)
        mask_bc = mask_bc.scatter_(dim= 1, index= idx_mask, value= True)
        return z_indices_predict, mask_bc
    
__MODEL_TYPE__ = {
    "MaskGit": MaskGit
}



        
