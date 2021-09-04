import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import json
from torch.utils.data import DataLoader
from FLAlgorithms.users.userbase import User
import numpy as np
import copy
# Implementation for FedAvg clients

class UserRobF(User):
    def __init__(self, device, numeric_id, train_data, test_data, model, batch_size, learning_rate, robust, gamma, local_epochs, K):
        super().__init__(device, numeric_id, train_data, test_data, model[0], batch_size, learning_rate, robust, gamma, local_epochs)

        self.loss = nn.CrossEntropyLoss()
        self.gamma = gamma
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate)
        self.avd_epoch = K
        #self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
    
    def train(self, alpha):
        self.model.train()
        for _ in range(1, self.local_epochs + 1):
            for X ,y in self.trainloader:
                X, y = X.to(self.device), y.long().to(self.device)
                X_adv =  self.wasssertein(X, y, alpha = alpha, num_iter = self.avd_epoch)
                self.optimizer.zero_grad()
                output = self.model(X_adv)
                loss = self.loss(output, y)
                loss.backward(retain_graph=True)
                self.optimizer.step()
        return