import torch
import torch.nn as nn
import torch.nn.functional as F
from FLAlgorithms.users.userbase import User
import numpy as np
# Implementation for FedAvg clients

class UserAFL(User):
    def __init__(self, device, numeric_id, train_data, test_data, model, batch_size, learning_rate, robust, gamma, local_epochs, K):
        super().__init__(device, numeric_id, train_data, test_data, model[0], batch_size, learning_rate, robust, gamma, local_epochs)

        self.loss = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate)
        # print("Created userAFL!")
    def train(self, epochs):
        # print("Training in userAFL!")
        LOSS = 0
        iter_num = 0
        self.model.train()
        for epoch in range(1, self.local_epochs + 1):
            for X,y in self.trainloaderfull:
                X, y = X.to(self.device), y.long().to(self.device)
                self.optimizer.zero_grad()
                loss = self.loss(self.model(X), y)
                loss.backward()
                self.optimizer.step()
                LOSS += loss
                iter_num += 1
        # return LOSS, loss # return last loss value
        return LOSS, LOSS * 1.0 / iter_num
