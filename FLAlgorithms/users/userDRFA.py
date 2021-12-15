import torch
import torch.nn as nn
import torch.nn.functional as F
from FLAlgorithms.users.userbase import User
import numpy as np
import copy
# Implementation for FedDRFA clients

class UserDRFA(User):
    def __init__(self, device, numeric_id, train_data, test_data, model, batch_size, learning_rate, robust, gamma, local_epochs, K):
        super().__init__(device, numeric_id, train_data, test_data, model[0], batch_size, learning_rate, robust, gamma, local_epochs)

        self.loss = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate)
        self.sample_number = 1
    
    def get_sample_parameters(self):
        for param in self.user_sample_model.parameters():
            param.detach()
        return self.user_sample_model.parameters()
        
    def train(self, epochs):
        # This is for FedDRFA
        LOSS = 0
        self.model.train()
        for epoch in range(1, self.local_epochs + 1):
            for X,y in self.trainloaderfull:
                X, y = X.to(self.device), y.long().to(self.device)
                self.optimizer.zero_grad()
                loss = self.loss(self.model(X), y)
                loss.backward()
                self.optimizer.step()
                LOSS += loss
            if self.sample_number == epoch:
                self.user_sample_model = copy.deepcopy(self.model) # return W{i}_{t'}
        return LOSS

