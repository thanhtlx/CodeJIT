import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Linear, ReLU


class Clasifier(nn.Module):
    def __init__(self, encoder, dropout):
        super(Clasifier, self).__init__()
        self.encoder = encoder
        self.dropout = dropout
        self.relu = ReLU(inplace=True)
        self.dan = Linear(768, 768)
        self.out = Linear(768, 2)

    def forward(self, inputs_ids, attn_masks):
        outputs = self.encoder(input_ids=inputs_ids, attention_mask=attn_masks)[0]
        # 24*256*768
        # =>  24*768
        # mean 
        # DAN
        # classifier
        outputs = outputs[:, 0, :]
        outputs = F.dropout(outputs, p=self.dropout)
        out = torch.mean(outputs, dim=0)
        out = self.dan(out)
        out = self.dan(out)
        out = self.relu(out)
        out = self.out(out)
        return out

