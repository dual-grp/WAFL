import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import json
from torch.utils.data import DataLoader
from FLAlgorithms.users.userbase import User
import numpy as np
# Implementation for FedAvg clients

class UserAVG(User):
    def __init__(self, device, numeric_id, train_data, test_data, model, batch_size, learning_rate, beta, L_k, local_epochs):
        super().__init__(device, numeric_id, train_data, test_data, model[0], batch_size, learning_rate, beta, L_k, local_epochs)

        if(model[1] == "Mclr_CrossEntropy"):
            self.loss = nn.CrossEntropyLoss()
        else:
            self.loss = nn.NLLLoss()

        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate)

    def train(self, epochs):
        LOSS = 0
        self.model.train()
        for epoch in range(1, self.local_epochs + 1):
            for X,y in self.trainloader:
                X, y = X.to(self.device), y.long().to(self.device)#self.get_next_train_batch()
                #X  = self.pgd_linf(X = X, y = y)
                self.optimizer.zero_grad()
                #output = self.model(X)
                loss = self.loss(self.model(X), y)
                loss.backward()
                self.optimizer.step()
                LOSS += loss
        return LOSS

    # def train(self, epochs):
    #     LOSS = 0
    #     self.model.train()
    #     for epoch in range(1, self.local_epochs + 1):
    #         self.model.train()
    #         X, y = self.get_next_train_batch()
    #         self.optimizer.zero_grad()
    #         output = self.model(X)
    #         loss = self.loss(output, y)
    #         loss.backward()
    #         self.optimizer.step()
    #     return LOSS