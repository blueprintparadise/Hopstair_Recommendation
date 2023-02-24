import torch
from torch import nn
from model import TowerModel
from std import *

'''
The data is stored as [number of users, user vector] 
'''


class ColdStartRank:
    """
    base class for cold start ranking. can be modified later (reason for class architecture).
    simple ranking is inverse distance

    __init__:
        :arg
        configuration modification, for args check :std:std_ranker

    __call__:
        :arg
        anchor: the point we want to compare a number of points with
        target: the points we want to compare them with

        :return
        indices of nearest neighbours; if return_dist as tuple with distances (in general weightings for likelihood)
    """

    def __init__(self, **config):
        for key in config:
            std_ranker[key] = config[key]
        self.config = std_ranker

    @staticmethod
    def weighting(dist: torch.tensor) -> torch.tensor:
        return 1/dist

    def __call__(self, anchor: torch.tensor, target: torch.tensor, k=10, random=2, **config) -> torch.tensor or tuple:
        dist = torch.norm(anchor.unsqueeze(1) - target.unsqueeze(0), p=2, dim=-1)
        knn = dist.topk(k, largest=False)

        out_ = tuple()
        if config.get('return_weight', True):
            out_ += tuple(self.weighting(knn[0].squeeze()))
        if config.get('return_dist', False):
            out_ += tuple(knn[0].squeeze())

        if config.get('return_val', False):
            idx = (torch.arange(1000).view(-1, 1, 1).repeat(1, k, 1), knn[1].unsqueeze(-1))
            idx = torch.concat(idx, dim=-1) #.view(-1, 2)
            out = target.unsqueeze(0).repeat(anchor.shape[0], 1, 1).gather(1, idx)
            return tuple(out) + out_ if out_ else out

        return tuple(knn[1].squeeze()) + out_ if out_ else knn[1].squeeze()


class ColdStartEmbed(nn.Module):
    """
    base class for cold start embedding. can be modified later (reason for class architecture).

   __init__:
       :arg
       embed_type: either 'user' or 'nudge'
       configuration modification, for args check :std:std_twotow

   __call__:
       :arg
       input: tensor with user or nudge information

       :return
       embedding of the input

   init_weights:
        :arg
        path to specific weight initialisation

        sets the weights of the model
   """

    def __init__(self, embed_type: str, **config):
        super(ColdStartEmbed, self).__init__()
        for key in config:
            std_twotow[key] = config[key]
        config = std_twotow

        self.embed_type = embed_type

        _input_dim = 'u_input_dim' if embed_type == 'user' else 'n_input_dim' if embed_type == 'nudge' else None
        self.model = TowerModel(**config, input_dim=config.get('input_dim', config[_input_dim]))

        #self.init_weights(**config)

    def init_weights(self, *args, **config) -> None:
        if args is not None:
            self.model.load_state_dict(torch.load(args[0]))
        else:
            self.model.load_state_dict(torch.load(config[f'{self.embed_type}_embed_dict']))

        self.model.eval()

    def forward(self, inp: torch.tensor) -> torch.tensor:
        return self.model(inp)


if __name__ == '__main__':
    users = torch.randn(1000, 15)
    nudges = torch.randn(100, 10)

    user_embed = ColdStartEmbed('user')
    nudge_embed = ColdStartEmbed('nudge')

    ranker = ColdStartRank()

    u_embed, n_embed = user_embed(users), nudge_embed(nudges)
    ranking, weighting = ranker(u_embed, n_embed, return_val=True)
