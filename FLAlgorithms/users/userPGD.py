import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import json
from torch.utils.data import DataLoader
from FLAlgorithms.users.userbase import User
import numpy as np
# Implementation for FedAvg clients

class UserPGD(User):
    def __init__(self, device, numeric_id, train_data, test_data, model, batch_size, learning_rate, robust, gamma, local_epochs):
        super().__init__(device, numeric_id, train_data, test_data, model[0], batch_size, learning_rate, robust, gamma, local_epochs)

        self.loss = nn.CrossEntropyLoss() 
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate)

    def train(self, epochs, adv_option):
        LOSS = 0
        self.model.train()
        for epoch in range(1, self.local_epochs + 1):
            for X,y in self.trainloader:
                X, y = X.to(self.device), y.long().to(self.device)
                # Adversarial training
                X = self.pgd_linf(X = X, y = y, epsilon = adv_option[0], alpha =adv_option[1], num_iter = 10)
                self.optimizer.zero_grad()
                loss = self.loss(self.model(X), y)
                loss.backward()
                self.optimizer.step()
                LOSS += loss
        return LOSS
