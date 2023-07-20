import torch
import torch.nn as nn
import torch
from torch.autograd import Variable
import copy
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss, MSELoss, BCELoss


class RobertaClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(self, config):
        super().__init__()
        self.manual_dense = nn.Linear(config.feature_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.out_proj_new = nn.Linear(config.hidden_size + config.hidden_size, 1)

    def forward(self, features, manual_features=None, **kwargs):
        x = features#[:, 0, :]  # take <s> token (equiv. to [CLS])  [bs,hidden_size]
        y = manual_features.float()  # [bs, feature_size]
        y = self.manual_dense(y)
        y = torch.tanh(y)

        x = torch.cat((x, y), dim=-1)
        x = self.dropout(x)
        x = self.out_proj_new(x)
        return x


class Model(nn.Module):
    def __init__(self, config, tokenizer, args):
        super(Model, self).__init__()
        self.config = config
        self.tokenizer = tokenizer
        self.classifier = RobertaClassificationHead(config)
        self.args = args

    def forward(self, embedd, manual_features=None,
                labels=None, output_attentions=None):
        # outputs = self.model.wv.get_sentence_vector(inputs_ids)
        
        last_layer_attn_weights = None

        logits = self.classifier(embedd, manual_features)

        prob = torch.sigmoid(logits)
        if labels is not None:
            loss_fct = BCELoss()
            loss = loss_fct(prob, labels.unsqueeze(1).float())
            return loss, prob, last_layer_attn_weights
        else:
            return prob

