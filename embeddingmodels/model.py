import torch
from torch import nn
from transformers import RobertaModel
from std import std_multigate, std_twotow


class TextEmbeddingModel(nn.Module):

    def __init__(self, vocab_size, embed_dim, linear_trans=True, **kwargs):
        super(TextEmbeddingModel, self).__init__()
        self.kwargs = kwargs
        self.linear_trans = linear_trans

        self.embedding = nn.EmbeddingBag(vocab_size, embed_dim, sparse=True)
        self.roberta = RobertaModel.from_pretrained('roberta-base')
        if linear_trans:
            self.fc = nn.Linear(embed_dim, embed_dim)

        self.init_weights()

    def init_weights(self):
        initrange = self.kwargs.get('initrange', 0.5)
        self.embedding.weight.data.uniform_(-initrange, initrange)

        if self.linear_trans:
            self.fc.weight.data.uniform_(-initrange, initrange)
            self.fc.bias.data.zero_()

    def forward(self, text, offsets):
        embedded = self.embedding(text, offsets)
        return self.fc(embedded) if self.linear_trans else embedded


def TowerModel(**config) -> nn.Module:
    embedding_layer = nn.Linear(config['input_dim'], config['hidden_dim'])
    hidden_layer = nn.Linear(config['hidden_dim'], config['hidden_dim'])
    output_layer = nn.Linear(config['hidden_dim'], config['embedding_dim'])
    return nn.Sequential(*[embedding_layer,
                           *[nn.ReLU(), hidden_layer] * config['layers'],
                           nn.ReLU(), output_layer])


class TwoTowerModel(nn.Module):
    r"""
    Two tower model for offline ranking of nudges against users (both are reduced in complexity through embedding).

    """

    def __init__(self, **config):
        super(TwoTowerModel, self).__init__()
        for key in config:
            std_twotow[key] = config[key]
        config = std_twotow

        # parameters starting with u are for the user model, whilst those starting with n are for the nudge model
        self.u_hidden_dim = config.get('u_hidden_dim', config['hidden_dim'])
        self.u_layers = config.get('u_layers', config['layers'])

        self.user_tower = TowerModel(embedding_dim=config['embedding_dim'],
                                     hidden_layers=self.u_layers, hidden_dim=self.u_hidden_dim,
                                     input_dim=config['u_input_dim'])
        self.nudge_tower = TowerModel(embedding_dim=config['embedding_dim'],
                                      hidden_layers=config.get('n_layers', self.u_layers),
                                      hidden_dim=config.get('n_hidden_dim', self.u_hidden_dim),
                                      input_dim=config['n_input_dim'])

    def forward(self, **inputs):
        user_embedding = self.user_tower(inputs['user_input'])
        nudge_embedding = self.nudge_tower(inputs['nudge_input'])
        return torch.matmul(user_embedding, nudge_embedding.T)


def Expert(**config) -> nn.Module:
    embedding_layer = nn.Linear(config['input_dim'], config['exp_hidden_dim'])
    hidden_layer = nn.Linear(config['exp_hidden_dim'], config['exp_hidden_dim'])
    output_layer = nn.Linear(config['exp_hidden_dim'], 1)
    return nn.Sequential(*[embedding_layer, nn.Sigmoid(),
                           *[nn.ReLU(), hidden_layer] * config['exp_layers'],
                           output_layer])


def Gate(**config) -> nn.Module:
    return nn.Sequential(*[nn.Linear(config['input_dim'], config['exp_num']), nn.Softmax()])


class MultiGateMixExperts(nn.Module):

    def __init__(self, **config):
        super(MultiGateMixExperts, self).__init__()
        for key in config:
            std_multigate[key] = config[key]
        config = std_multigate

        self.input_dim = config['input_dim']

        self.gate1 = Gate(**config)
        self.gate2 = Gate(**config)

        self.experts = [Expert(**config) for _ in config['exp_num']]

        self.utility1 = TowerModel(**config)

    def forward(self, inp):
        # input should be concatenation of: 
        # user id, distance vector in embedding, category of user and items, user data and geodata
        f = torch.tensor([expert(inp) for expert in self.experts])
        g1, g2 = self.gate1(inp), self.gate2(inp)
        return torch.matmul(g1, f), torch.matmul(g2, f)


if __name__ == '__main__':
    TextEmbeddingModel(500, 70, initrange=0.5)
    model = TwoTowerModel()
    model.forward(user_input=torch.randn(300, 15), nudge_input=torch.randn(20, 10))


def std_cold_start(*in_):
    pass
