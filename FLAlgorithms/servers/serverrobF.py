import torch
import os
import torch.multiprocessing as mp

from FLAlgorithms.users.userrobF import UserRobF
from FLAlgorithms.servers.serverbase import Server
from utils.model_utils import read_user_data, read_domain_data
import numpy as np
import copy
# Implementation for FedAvg Server

class FedRob(Server):
    def __init__(self, experiment, device, dataset, algorithm, model, batch_size, learning_rate, robust, gamma, num_glob_iters, local_epochs, sub_users, num_users, times):
        super().__init__(experiment, device, dataset, algorithm, model[0], batch_size, learning_rate, robust, gamma, num_glob_iters, local_epochs, sub_users, num_users, times)

        # Initialize data for all  users
        if(dataset[0] == "Cifar10"):
            self.adv_option = [8/255,2/255]
        elif(dataset[0] == "Mnist" or dataset[0] == "EMNIST"):
            self.adv_option = [0.3,0.01]
        else:
            self.adv_option = [0,0]

        #if(num_users == 1):
        #    i = 0
        #    train , test = dataset[2][i]
        #    user = UserRobF(device, i, train, test, model, batch_size, learning_rate, robust, gamma, local_epochs)
        #    self.target_domain = user#copy.deepcopy(user)
        #    user.set_target()
        #    self.users.append(user)
        #    self.total_train_samples += user.train_samples
        #    return
        self.target_domain = None
        
        for i in range(num_users):
            train , test = dataset[2][i]
            user = UserRobF(device, i, train, test, model, batch_size, learning_rate, robust, gamma, local_epochs)
            if(self.robust <= 0): # no robust, domain option
                if(i == dataset[1] or (i == num_users-1 and dataset[1] < 0)):
                    self.target_domain = user
                    user.set_target()
                    continue
            self.users.append(user)
            self.total_train_samples += user.train_samples
            
        print("Number of users / total users:", int(sub_users * num_users), " / " ,num_users)
        print("Finished creating FedRob server.")

    def train(self):
        for glob_iter in range(self.num_glob_iters):
            if(self.experiment):
                self.experiment.set_epoch( glob_iter + 1)
            print("-------------Round number FedRob: ",glob_iter, " -------------")
            #loss_ = 0
            self.send_parameters()

            # Evaluate model each interation
            self.evaluate()

            if(self.robust <= 0):
                self.evaluate_on_target()

            if(self.robust > 0):
                self.evaluate_robust('pgd', self.robust, glob_iter,self.adv_option)
            #self.evaluate_robust('fgsm')

            # Select subset of user for training
            self.selected_users = self.select_users(glob_iter, self.sub_users)
            for user in self.selected_users:
                user.train(self.local_epochs)

            self.aggregate_parameters(self.selected_users)
            
        self.save_results()
        self.save_model()