import torch
import torch.nn as nn
import torch.nn.functional as F
from FLAlgorithms.users.userbase import User

class UserWAFL(User):
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
                X_adv =  self.wasssertein(X, y)
                self.optimizer.zero_grad()
                output = self.model(X_adv)
                loss = self.loss(output, y)
                loss.backward(retain_graph=True)
                self.optimizer.step()
        return