import torch.nn as nn
import torch
import math

#TODO1
#ref https://medium.com/@aisagescribe/building-a-multi-head-attention-with-pytorch-from-scratch-a-simple-yet-detailed-explanation-e000e4b84c0a
class MultiHeadAttention(nn.Module):
    def __init__(self, dim=768, num_heads=16, attn_drop=0.1):
        super(MultiHeadAttention, self).__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.attn_drop = attn_drop
        self.dim_per_head = self.dim // self.num_heads
        self.dropout = nn.Dropout(p=attn_drop)

        self.Q = nn.Linear(self.dim, self.dim, bias=False)
        self.K = nn.Linear(self.dim, self.dim, bias=False)
        self.V = nn.Linear(self.dim, self.dim, bias=False)
        self.O = nn.Linear(self.dim, self.dim, bias=False)

    def forward(self, x):
        ''' Hint: input x tensor shape is (batch_size, num_image_tokens, dim), 
            because the bidirectional transformer first will embed each token to dim dimension, 
            and then pass to n_layers of encoders consist of Multi-Head Attention and MLP. 
            # of head set 16
            Total d_k , d_v set to 768
            d_k , d_v for one head will be 768//16.
        ''' 
        query, key, value = self.Q(x), self.K(x), self.V(x)
        query = query.view(x.shape[0], x.shape[1], self.num_heads, self.dim_per_head).transpose(1, 2)
        key = key.view(x.shape[0], x.shape[1], self.num_heads, self.dim_per_head).transpose(1, 2)
        value = value.view(x.shape[0], x.shape[1], self.num_heads, self.dim_per_head).transpose(1, 2)
        #reshape q,k,v into shape (B, heads_num, token_num, dim_per_head)
        attention_weight = query @ (key.transpose(-1, -2))
        #do matrix multiply to q @ k on last two dim
        #attension in shape (B, head_num, token_num, token_num)
        attention_weight = attention_weight / (self.dim_per_head ** 0.5)
        #scale attention_weight to normorlize dot product
        attention_weight = torch.softmax(attention_weight, dim=-1).to(value.dtype)
        attention_weight = self.dropout(attention_weight)
        out = attention_weight @ value
        #out in shape (B, head_num, token_num, dim_per_head)
        out = out.transpose(1, 2).contiguous().view(x.shape[0], x.shape[1], self.dim)
        #out in shape (B, token_num, dim)
        out = self.O(out)
        #out = torch.nn.functional.dropout(out, p = self.attn_drop)
        return out

        
class MLP(nn.Sequential):
    def __init__(self, dim=768, hidden_dim=3072, drop_rate=0.1):
        super(MLP, self).__init__(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(drop_rate)
        )
        
    def forward(self, input):
        return super().forward(input)
    
    
class TokenPredictor(nn.Sequential):
    def __init__(self, dim=768):
        super(TokenPredictor, self).__init__(
            nn.Linear(in_features=dim, out_features=dim),
            nn.GELU(),
            nn.LayerNorm(dim, eps=1e-12)
        )
        
    def forward(self, input):
        return super().forward(input)
    
    
class Encoder(nn.Module):
    def __init__(self, dim=768, hidden_dim=1536):
        super(Encoder, self).__init__()
        self.Attention = MultiHeadAttention(dim)
        self.LayerNorm1 = nn.LayerNorm(dim, eps=1e-12)
        self.LayerNorm2 = nn.LayerNorm(dim, eps=1e-12)
        self.MLP = MLP(dim, hidden_dim)
        self.dropout = nn.Dropout(p=0.1)

    def forward(self, x):
        attn = self.Attention(x)
        attn = self.dropout(attn)
        
        x = x + attn
        x = self.LayerNorm1(x)
        
        mlp = self.MLP(x)
        x = x + mlp
        return self.LayerNorm2(x)
    