# Original: https://github.com/pytorch/examples/blob/490243127c02a5ea3348fa4981ecd7e9bcf6144c/snli/model.py

import torch
import torch.nn as nn


class Bottle(nn.Module):

    def forward(self, input):
        if len(input.size()) <= 2:
            return super(Bottle, self).forward(input)
        size = input.size()[:2]
        out = super(Bottle, self).forward(input.view(size[0]*size[1], -1))
        return out.view(size[0], size[1], -1)


class Linear(Bottle, nn.Linear):
    pass


class Encoder(nn.Module):

    def __init__(self, config):
        super(Encoder, self).__init__()
        self.config = config
        input_size = config.d_proj if config.projection else config.d_embed
        dropout = 0 if config.n_layers == 1 else config.dp_ratio
        self.rnn = nn.LSTM(input_size=input_size, hidden_size=config.d_hidden,
                        num_layers=config.n_layers, dropout=dropout,
                        bidirectional=config.birnn)

    def forward(self, inputs):
        batch_size = inputs.size()[1]
        state_shape = self.config.n_cells, batch_size, self.config.d_hidden
        h0 = c0 =  inputs.new_zeros(state_shape)
        outputs, (ht, ct) = self.rnn(inputs, (h0, c0))
        return ht[-1] if not self.config.birnn else ht[-2:].transpose(0, 1).contiguous().view(batch_size, -1)


class SNLIClassifier(nn.Module):

    def __init__(self, config):
        super(SNLIClassifier, self).__init__()
        self.config = config
        self.embed = nn.Embedding(config.n_embed, config.d_embed)
        self.projection = Linear(config.d_embed, config.d_proj)
        self.encoder = Encoder(config)
        self.dropout = nn.Dropout(p=config.dp_ratio)
        self.relu = nn.ReLU()
        seq_in_size = 2*config.d_hidden
        if self.config.birnn:
            seq_in_size *= 2
        lin_config = [seq_in_size]*2
        self.out = nn.Sequential(
            Linear(*lin_config),
            self.relu,
            self.dropout,
            Linear(*lin_config),
            self.relu,
            self.dropout,
            Linear(*lin_config),
            self.relu,
            self.dropout,
            Linear(seq_in_size, config.d_out))

    # EDIT(momohatt): batch -> premise, hypothesis
    def forward(self, premise, hypothesis):
        prem_embed = self.embed(premise)
        hypo_embed = self.embed(hypothesis)
        if self.config.fix_emb:
            prem_embed = prem_embed.detach()
            hypo_embed = hypo_embed.detach()
        if self.config.projection:
            prem_embed = self.relu(self.projection(prem_embed))
            hypo_embed = self.relu(self.projection(hypo_embed))
        premise = self.encoder(prem_embed)
        hypothesis = self.encoder(hypo_embed)
        scores = self.out(torch.cat([premise, hypothesis], 1))
        return scores


# Config and example input
# Taken from https://github.com/pytorch/pytorch/blob/master/test/jit/test_models.py

class Config:
    n_embed = 100
    d_embed = 100
    d_proj = 300
    dp_ratio = 0.0  # For deterministic testing TODO: change by fixing seed in checkTrace?
    d_hidden = 30
    birnn = True
    d_out = 300
    fix_emb = True
    projection = True
    n_layers = 2
    n_cells = 4  # 2 * n_layers because birnn = True


def gen_SNLI_model():
    model = SNLIClassifier(Config)
    premise = torch.LongTensor(48, 64).random_(0, 100)
    hypothesis = torch.LongTensor(24, 64).random_(0, 100)
    return model, (premise, hypothesis)
