import torch
import os
import torch.multiprocessing as mp

from utils.get_femnist_data import *
from FLAlgorithms.users.userWAFL import UserWAFL
from FLAlgorithms.servers.serverbase import Server
from utils.model_utils import read_user_data, read_domain_data
import numpy as np
import copy
# Implementation for FedAvg Server

class WAFL(Server):
    def __init__(self, experiment, device, dataset, algorithm, model, batch_size, learning_rate, robust, gamma, num_glob_iters, local_epochs, sub_users, num_users, K, alpha, times, epsilon):
        super().__init__(experiment, device, dataset, algorithm, model[0], batch_size, learning_rate, robust, gamma, num_glob_iters, local_epochs, sub_users, num_users, times)
        
        self.epsilon = epsilon
    
        # Initialize adver options
        if(dataset[0] == "Cifar10"):
            self.adv_option = [8/255,2/255,10]
        elif(dataset[0] == "Mnist"):
            self.adv_option = [0.3,0.01,40]
        elif(dataset[0] == "Emnist"):
            self.adv_option = [0.3,0.01,40]
        elif(dataset[0] == "FeMnist"):
            self.adv_option = [epsilon, 0.01,10]
        else:
            self.adv_option = [0,0,0]

        self.target_domain = None
        self.alpha = alpha
        if dataset[0] == "FeMnist":
            self.num_users = 35
        else: 
            self.num_users = num_users

        for i in range(num_users):
            if dataset[0] == "FeMnist":
                train, test = get_user_dataset(i)
            else:
                train , test = dataset[2][i]
            user = UserWAFL(device, i, train, test, model, batch_size, learning_rate, robust, gamma, local_epochs, K)
            if(self.robust < 0): # no robust, domain option
                if(i == dataset[1] or (i == num_users-1 and dataset[1] < 0)):
                    self.target_domain = user
                    user.set_target()
                    continue
            self.users.append(user)
            self.total_train_samples += user.train_samples
        print(f"Level of attack: epsilon = {self.epsilon}") 
        print("Number of users / total users:", int(sub_users * num_users), " / " ,num_users)
        print("Finished creating WAFL server.")

    def train(self):
        for glob_iter in range(self.num_glob_iters):
            if(self.experiment):
                self.experiment.set_epoch( glob_iter + 1)
            print("-------------Round number WAFL: ",glob_iter, " -------------")
            #loss_ = 0
            self.send_parameters()

            # Evaluate model each interation
            self.evaluate()

            if(self.robust < 0):
                self.evaluate_on_target()

            if(self.robust > 0):
                self.adv_users = self.select_users(glob_iter, self.robust)
                self.evaluate_robust('pgd', self.adv_option)
                #self.evaluate_robust('fgsm', self.adv_option)

            # Select subset of user for training
            self.selected_users = self.select_users(glob_iter, self.sub_users)
            for user in self.selected_users:
                user.train(self.alpha)

            self.aggregate_parameters(self.selected_users)
            
        self.save_results()
        self.save_model()