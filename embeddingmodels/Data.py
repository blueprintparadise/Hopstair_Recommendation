from os.path import exists
from time import time

import torch
from model import std_cold_start, TwoTowerModel
from torch.utils.data import DataLoader

from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator


class TextLoader(DataLoader):

    def __init__(self, data, tokenizer=get_tokenizer('basic_english'), batch_size=8, shuffle=False):
        # data is iterator which yields (nudge, label)
        self.data = data
        self.tokenizer = tokenizer

        self.vocab = build_vocab_from_iterator(self.yield_tokens(self.data), specials=["<unk>"])
        self.vocab.set_default_index(self.vocab["<unk>"])

        self.label_pipeline = lambda x: self.vocab(self.tokenizer(x))
        self.text_pipeline = lambda x: int(x) - 1

        super(TextLoader, self).__init__(self.data, batch_size=batch_size, shuffle=shuffle,
                                         collate_fn=self.collate_batch)

    def yield_tokens(self, data_iter):
        for _, text in data_iter:
            yield self.tokenizer(text)

    def collate_batch(self, batch):
        label_list, text_list, offsets = [], [], [0]
        for (_label, _text) in batch:
            label_list.append(self.label_pipeline(_label))
            processed_text = torch.tensor(self.text_pipeline(_text), dtype=torch.int64)
            text_list.append(processed_text)
            offsets.append(processed_text.size(0))
        label_list = torch.tensor(label_list, dtype=torch.int64)
        offsets = torch.tensor(offsets[:-1]).cumsum(dim=0)
        text_list = torch.cat(text_list)
        return label_list, text_list, offsets


'''
Models:
- RL suffers from scalability issues
- Two-tower approach scales well and is not data-intensive
- Rec Sim we simulate a recommender that tests against a simulated user (reinforcement learning)

https://miro.medium.com/max/1400/1*66WV7RCzS2dv4_C4-_gsRw.webp


Recommendation Primer:
- Content-user vs. collaborative matching 
  - I think that the best way is collaborative as it allows to learn the preferences from a larger data set 
    (more robust)


Things to consider:
- Cold start
  - Start based off of questionnaire and have predefined profiles
  - 
- Feedback loops to be avoided
- Online vs. Offline evaluation (crucially it comes down to whether we want the user to interact spontaneously or just 
  when prompted)
  - Not a problem for nudges since we have time to plan for them
    - Initially more scalable
      - Daily updates would be sufficient
      - we can continue on stale recommendation distributions while new ones are evaluated
    - maybe modify broadly offline and do some online subroutine
    - like reinforcement learning which we average to feedback in the overall classifier
    - Collaborative filtering is outperformed by Alibaba Swing
      - Avoids the problem of feedback loops as it conversely weights the similarity between two users
- How to handle large action space?
  - We can consider classifying all prompts in a small number of classes (unsupervised)
    - Begin with a simple labelled version
  - 
  
"Ranking can be framed as either a classification or learning to rank problem. As a
classification problem, we can score candidates based on probability of click or purchase.
Logistic regression with crossed features is simple to implement and a difficult baseline to
beat. Decision trees are also commonly used. As a learning to rank problem, commonly
used algorithms include LambdaMart, XGBoost, and LightGBM."

- We can consider performing a multitask learning routine (multiple different guesses in parallel) to estimate different
  outcomes from the two-tower model embeddings (we choose the nearest neighbours here and pass them on). We rank by 
  using a weighted average
  - For example: we may be interested in predicting the category and the post, the first task is simpler and will 
    reinforce our decision
  - Considerations:
    - The tasks need to be related or they will compete
    - Freeze weights of each task when there is no data for the type of interaction (for example on YouTube we may not 
      have a like or dislike)
- The best model for this is the Multi-gate Mixture of Experts


Profiles:
- Probability distribution over classes of profiles
  - Each profile represents a nudge class
    - These can be later learnt in an unsupervised way to extend the available classes and get more nuance
  - Random sampling over whole distribution or top choices

“Google found that taking an additional 500ms to generate search results reduced traffic by 20%. Amazon shared that 
100ms additional latency reduces profit by 1%”

'''


class PersonChar:
    """
    Class containing all characteristics of a person

    Args:
        id_: user id or features for new users which are assigned new id

    Optional Args:
        load=True: whether to instantly load the data to memory (in the case of a new user whether to save them)
        cold_start_fn: the function used for the cold start of a new user, otherwise resorts to standard model
        save=True: whether to store all the computations for later recall

    Methods:
        update: updates the character vector
        embedding: embeds the data to the space or calls it if precomputed
    """

    def __init__(self, id_, **infra):

        self.data = id_
        lod = infra.get('load', True)

        if isinstance(self.data, dict):
            # new user process from dict currently

            self.user_id = 0

            if lod:
                cold_start = infra.get('cold_start_fn', std_cold_start)
                self.type = cold_start(self.data)
                self.character_vector = torch.load(f'./pers_types/{self.type}.pt')

        elif isinstance(self.data, int):
            if not exists(f'./users/{self.user_id}.pt'):
                raise FileNotFoundError(f'{self.user_id} is not a valid user ID in users directory')

            self.user_id = self.data

            if lod:
                self.character_vector = torch.load(f'./users/{self.user_id}.pt')

        else:
            raise TypeError(f'Input has to be a character dictionary (dict) for new users or ID (int) for existing '
                            f'ones. Type found was: {self.data.__class__.__name__}')

        if infra.get('save', False):
            torch.save(self.character_vector, f'./users/{self.user_id}.pt')
            torch.save(self.character_vector, f'./user_embedding/{self.user_id}.pt')

    def update(self, new_vector):
        # overwrite the existing character vector with a specific vector

        self.character_vector = new_vector
        torch.save(self.character_vector, f'./users/{self.user_id}.pt')

    def embedding(self, **kwargs) -> torch.Tensor:
        # compute the new embedding if a model is passed or alternatively try to recall the embedding

        if 'model' in kwargs:
            input_ = torch.concat([self.user_id, self.character_vector])  # create feature vector
            return kwargs['model'](input_)
        else:
            try:
                return torch.load(f'./user_embeddings/{self.user_id}.pt')
            except FileNotFoundError:
                model = TwoTowerModel()     # use standard Model
                input_ = torch.concat([self.user_id, self.character_vector])  # create feature vector
                output_ = model(input_)
                torch.save(output_, f'./user_embeddings/{self.user_id}.pt')
                return output_
