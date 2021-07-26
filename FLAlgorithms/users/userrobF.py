import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import json
from torch.utils.data import DataLoader
from FLAlgorithms.users.userbase import User
import numpy as np 
# Implementation for FedAvg clients

class UserRobF(User):
    def __init__(self, device, numeric_id, train_data, test_data, model, batch_size, learning_rate, beta, mu, local_epochs):
        super().__init__(device, numeric_id, train_data, test_data, model[0], batch_size, learning_rate, beta, mu, local_epochs)

        if(model[1] == "Mclr_CrossEntropy"):
            self.loss = nn.CrossEntropyLoss()
        else:
            self.loss = nn.NLLLoss()

        self.mu = mu
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate)

    def train(self, epochs):
        self.model.train()
        self.local_epochs_adversal = 20
        self.local_iteration = 1

        for i in range(1, self.local_iteration + 1):
            X, y = self.get_next_train_batch()
            X.requires_grad_(True)
            output = self.model(X)
            loss = self.loss(output, y)
            if X.grad is not None:
                X.grad.data.zero_()
            gradX = torch.autograd.grad(self.mu*loss, X,retain_graph=True)
            #X.requires_grad_(False)
            X_adv = X + gradX[0] 
            X_adv.requires_grad_(True)

            # Step1 : Finding adversarial sample
            for epoch in range(1, self.local_epochs_adversal + 1):
                output = self.model(X_adv)
                loss = self.loss(output, y)
                if X_adv.grad is not None:
                    X_adv.grad.data.zero_()
                gradX_avd1 = torch.autograd.grad(loss, X_adv,retain_graph=True)
                gradX_avd2 = torch.autograd.grad(0.5*self.mu*torch.norm(X_adv-X)**2,X_adv,retain_graph=True)
                gradX_avd = gradX_avd2[0] - gradX_avd1[0]
                X_adv = X_adv - 1./np.sqrt(epoch+2) * gradX_avd
                #X_adv = X_adv - self.learning_rate * gradX_avd
                #print("grad different", torch.norm(- 1./np.sqrt(epoch+2) * gradX_avd))
                #print("---different----",torch.norm(X_adv-X))
            #X_adv.requires_grad_(False)

            # Step2 : Update local model using adversarial sample
            for epoch in range(1, self.local_epochs + 1):
                self.optimizer.zero_grad()
                output = self.model(X_adv)
                loss = self.loss(output, y)
                loss.backward(retain_graph=True)
                self.optimizer.step()
                #self.clone_model_paramenter(self.model.parameters(), self.local_model)
        return 
