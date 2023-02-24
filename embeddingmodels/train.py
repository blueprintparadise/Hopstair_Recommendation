import time
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sn
from Data import TextLoader
from model import *
from transformers import RobertaTokenizer


def train(epochs=1000, loss=nn.CrossEntropyLoss(), LR=0.001):

    optimizer = torch.optim.SGD(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.1)
    loader = TextLoader()

    for epoch in range(epochs):
        last_time = time.time()
        accuracy, total = 0, 0

        for train_batch, test_batch in loader:

            optimizer.zero_grad()

            guess = model(train_batch)

            L = loss(guess, test_batch)
            L.backward()
            optimizer.step()

            accuracy += (guess == test_batch).sum().item()
            total += len(guess)

        print(f'| epoch {epoch} | accuracy {accuracy/total*100:3d}% | elapsed time {time.time()-last_time:3d}s')
        scheduler.step()


if __name__ == '__main__':
    labels = []
    conf_mat = []
    df_cm = pd.DataFrame(conf_mat, index=[i for i in labels], columns=[i for i in labels])
    plt.figure(figsize=(10, 7))
    plt.title('Confusion matrix')
    sn.heatmap(df_cm, annot=True, fmt='g')
    plt.show()

    # TODO: Word removals
    # TODO: Ensemble, boosting bagging?
    # TODO:
    # Hyperparameters
    # Guarantee classes are balanced

    train_loader = TextLoader(data, tokenizer=RobertaTokenizer.from_pretrained('roberta-base'))
    model = TextClassificationModel

    train(epochs=1000, model=model, loader=train_loader)


