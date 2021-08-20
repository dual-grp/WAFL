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
    def __init__(self, device, numeric_id, train_data, test_data, model, batch_size, learning_rate, beta, mu, local_epochs):
        super().__init__(device, numeric_id, train_data, test_data, model[0], batch_size, learning_rate, beta, mu, local_epochs)

        if(model[1] == "Mclr_CrossEntropy"):
            self.loss = nn.CrossEntropyLoss()
        else:
            self.loss = nn.NLLLoss()
        self.epsilon = 0.3
        self.mu = mu
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate)
    '''
    def train(self, epochs):
        self.model.train()
        self.local_epochs_adversal = 10
        #self.local_iteration = 1
        for local_epoch in range(1, self.local_epochs + 1):
            for X ,y in self.trainloader:
                #X, y = self.get_next_train_batch()
                X, y = X.to(self.device), y.long().to(self.device)
                
                X.requires_grad_(True)
                output = self.model(X)
                loss = self.loss(output, y)
                if X.grad is not None:
                    X.grad.data.zero_()
                gradX = torch.autograd.grad(self.mu*loss, X, retain_graph=True)

                # Step1 : Finding adversarial sample
                X.requires_grad_(False)
                X_adv = X - gradX[0]
                for epoch in range(1, self.local_epochs_adversal + 1):
                    X_adv.requires_grad_(True)
                    output = self.model(X_adv)
                    loss = self.loss(output, y)
                    if X_adv.grad is not None:
                        X_adv.grad.data.zero_()
                    gradX_avd1 = torch.autograd.grad(loss, X_adv,retain_graph=True)
                    # using norm L1 or L2
                    #gradX_avd2 = torch.autograd.grad(0.5*self.mu*torch.norm(X_adv-X)**2,X_adv,retain_graph=True)
                    gradX_avd2 = torch.autograd.grad(self.mu*torch.norm(X_adv-X),X_adv,retain_graph=True)
                    gradX_avd = gradX_avd2[0] - gradX_avd1[0]
                    X_adv = X_adv - 1./np.sqrt(epoch+2) * gradX_avd
                    norm_grad = torch.norm(gradX_avd)
                    norm_sample = torch.norm(X_adv-X)
                    #X_adv = X_adv - self.learning_rate * gradX_avd
                    #print("grad different", torch.norm(gradX_avd))
                    #print("---different----",torch.norm(X_adv-X))
                #X_adv.requires_grad_(False)

                # Step2 : Update local model using adversarial sample
                self.optimizer.zero_grad()
                output = self.model(X_adv)
                loss = self.loss(output, y)
                loss.backward(retain_graph=True)
                self.optimizer.step()
        return 
    '''
    
    def train(self, epochs):
        #import torch, gc
        #gc.collect()
        #torch.cuda.empty_cache()
        self.model.train()
        self.local_epochs_adversal = 20
        #self.local_iteration = 1
        for _ in range(1, self.local_epochs + 1):
            for X ,y in self.trainloader:
                #X, y = self.get_next_train_batch()
                X, y = X.to(self.device), y.long().to(self.device)
                #X , y = self.perturb(X_org, y)
                # Step1 : Finding adversarial sample
                #X.requires_grad_(True)
                #loss = self.loss(self.model(X), y)
                #gradX = torch.autograd.grad(loss, X, retain_graph=True)
                
                #X_adv = X.clone() + gradX[0] 
                #X_adv = X.clone() + np.random.uniform(-self.epsilon, self.epsilon, X.shape)
                #X.requires_grad_(False)
                #X_adv.requires_grad_(True)
                #for epoch in range(1, self.local_epochs_adversal + 1):
                #    output = self.model(X_adv)
                #    loss = self.loss(output, y)
                #    adv_loss = -loss + self.mu*torch.norm(X_adv-X)#**2
                #    if(adv_loss < 0):
                #        break
                #    gradX_avd = torch.autograd.grad(adv_loss, X_adv, retain_graph=True)[0]#/len(X)
                #    X_adv = X_adv - 0.01 * gradX_avd
                    #gradX_avd_norm = torch.norm(gradX_avd)
                    #sample_nome = torch.norm(X_adv - X)
                    #X_adv = X_adv - self.learning_rate * gradX_avd
                    #print("-------------------------")
                    #print("Adver loss",adv_loss )
                    #print("grad different", gradX_avd_norm)
                    #rint("---sample avd different----",sample_nome)
                #X_adv.requires_grad_(False)
                # Step2 : Update local model using adversarial sample
                X_adv =  self.wasssertein_linf(X, y)
                self.optimizer.zero_grad()
                output = self.model(X_adv)
                loss = self.loss(output, y)
                loss.backward(retain_graph=True)
                self.optimizer.step()
        return