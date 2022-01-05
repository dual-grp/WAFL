import torch
import torch.nn as nn
import torch.nn.functional as F
from FLAlgorithms.users.userbase import User
import numpy as np

# Implementation for FedAvg clients
FULL_BATCH = False

class UserAVG(User):
    def __init__(self, device, numeric_id, train_data, test_data, model, batch_size, learning_rate, robust, gamma, local_epochs, K):
        super().__init__(device, numeric_id, train_data, test_data, model[0], batch_size, learning_rate, robust, gamma, local_epochs)

        self.loss = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate)
        print(f"User Avg with FULL_BATCH = {FULL_BATCH}")

    def train(self, epochs):
        # This is for FedAvg
        LOSS = 0
        self.model.train()
        if FULL_BATCH == True:
            for epoch in range(1, self.local_epochs + 1):
                for X,y in self.trainloaderfull:
                    X, y = X.to(self.device), y.long().to(self.device)
                    self.optimizer.zero_grad()
                    loss = self.loss(self.model(X), y)
                    loss.backward()
                    self.optimizer.step()
                    LOSS += loss
            return LOSS
        else: # Running with minibatch
            for epoch in range(1, self.local_epochs + 1):
                for X,y in self.trainloader:
                    X, y = X.to(self.device), y.long().to(self.device)
                    self.optimizer.zero_grad()
                    loss = self.loss(self.model(X), y)
                    loss.backward()
                    self.optimizer.step()
                    LOSS += loss
            return LOSS
