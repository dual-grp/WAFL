import torch
import torch.nn as nn
from FLAlgorithms.users.userbase import User

class UserPGD(User):
    def __init__(self, device, numeric_id, train_data, test_data, model, batch_size, learning_rate, robust, gamma, local_epochs, K):
        super().__init__(device, numeric_id, train_data, test_data, model[0], batch_size, learning_rate, robust, gamma, local_epochs)

        self.loss = nn.CrossEntropyLoss() 
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate)
        self.avd_epoch = K
        #self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)

    def train(self, epochs, adv_option):
        LOSS = 0
        self.model.train()
        for epoch in range(1, self.local_epochs + 1):
            for X,y in self.trainloader:
                X, y = X.to(self.device), y.long().to(self.device)
                # Adversarial training
                X = self.pgd_linf(X = X, y = y, epsilon = adv_option[0], alpha =adv_option[1], num_iter = self.avd_epoch)
                self.optimizer.zero_grad()
                loss = self.loss(self.model(X), y)
                loss.backward()
                self.optimizer.step()
                LOSS += loss
        return LOSS
