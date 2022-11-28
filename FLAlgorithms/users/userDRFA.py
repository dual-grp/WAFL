import torch
import torch.nn as nn
import torch.nn.functional as F
from FLAlgorithms.users.userbase import User
import numpy as np
import copy
# Implementation for FedDRFA clients
FULL_BATCH = False # Option to select running with full batch or minibatch

class UserDRFA(User):
    def __init__(self, device, numeric_id, train_data, test_data, model, batch_size, learning_rate, robust, gamma, local_epochs, K):
        super().__init__(device, numeric_id, train_data, test_data, model[0], batch_size, learning_rate, robust, gamma, local_epochs)
        step_size = 60
        gamma_decay = 1 # No learning rate decay, choose gamma in [0;1) for learning rate decay
        self.loss = nn.CrossEntropyLoss()
        # self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate)
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate)
        self.schedule_optimizer = torch.optim.lr_scheduler.StepLR(optimizer=self.optimizer, step_size=step_size, gamma=gamma_decay)
        self.sample_number = 1 # Initialize the sample number, this number will be randomly chosed each iteration
        # print(f"step_size: {step_size}, gamma: {gamma}, learning_rate: {self.learning_rate}")
        self.model = self.model.to(self.device)

    def get_sample_parameters(self):
        for param in self.user_sample_model.parameters():
            param.detach()
        return self.user_sample_model.parameters()       

    def train(self, epochs):
        # This is for FedDRFA
        LOSS = 0
        self.model = self.model.to(self.device)
        self.model.train()
        count = 0
        if FULL_BATCH == True:
            for epoch in range(1, self.local_epochs + 1):
                for X,y in self.trainloaderfull:
                    X, y = X.to(self.device), y.long().to(self.device)
                    self.optimizer.zero_grad()
                    loss = self.loss(self.model(X), y)
                    loss.backward()
                    self.optimizer.step()
                    self.schedule_optimizer.step()
                    LOSS += loss
                if self.sample_number == epoch:
                    self.user_sample_model = copy.deepcopy(self.model) # return W_{i}^{t'}
            return LOSS
        else:
            for epoch in range(1, self.local_epochs + 1):
                for X,y in self.trainloader:
                    X, y = X.to(self.device), y.long().to(self.device)
                    self.optimizer.zero_grad()
                    loss = self.loss(self.model(X), y)
                    loss.backward()
                    self.optimizer.step()
                    self.schedule_optimizer.step()
                    LOSS += loss
                    count += 1
                if self.sample_number == epoch:
                    self.user_sample_model = copy.deepcopy(self.model) # return W_{i}^{t'}
            return LOSS, loss
    

