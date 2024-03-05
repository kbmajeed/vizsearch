import datetime
import logging

import torch
import torch.nn as nn
from torch.optim import SGD
from torch.utils.data import Dataset

from etl.data_prep import cat_dogs_dataset


class CNNEmb(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.act1 = nn.Tanh()
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(16, 8, kernel_size=3, padding=1)
        self.act2 = nn.Tanh()
        self.pool2 = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(8 * 12 * 12, 32)
        self.act3 = nn.Tanh()
        self.fc2 = nn.Linear(32, 2)

    def forward(self, x):
        out = self.pool1(self.act1(self.conv1(x)))
        out = self.pool2(self.act2(self.conv2(out)))
        out = out.view(-1, 8 * 12 * 12)
        out = self.act3(self.fc1(out))
        out = self.fc2(out)
        return out


def train(n_epochs, optimizer, model, loss_fn, train_loader):

    for epoch in range(1, n_epochs + 1):
        print(f'Epoch: {epoch}')
        loss_train = 0.0

        for ix, (imgs, labels) in enumerate(train_loader):
            outputs = model(imgs)
            loss = loss_fn(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_train += loss.item()

        print(f'{datetime.datetime.now()} | Epoch: {epoch}, Training loss: {loss_train/len(train_loader)}')
    pass


if __name__ == '__main__':

    dataset_loader = torch.utils.data.DataLoader(cat_dogs_dataset, batch_size=32, shuffle=True, num_workers=0)
    logging.info("cat_dogs ML dataloader created")

    model = CNNEmb()
    optimizer = SGD(model.parameters(), lr=1e-2)
    loss_fn = nn.CrossEntropyLoss()

    train(n_epochs=50,
          optimizer=optimizer,
          model=model,
          loss_fn=loss_fn,
          train_loader=dataset_loader)
