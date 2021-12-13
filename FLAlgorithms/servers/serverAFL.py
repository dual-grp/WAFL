import torch
import os
import torch.multiprocessing as mp

from FLAlgorithms.users.userAFL import UserAFL
from FLAlgorithms.servers.serverbase import Server
from utils.model_utils import read_data, read_domain_data
import numpy as np
import copy
# Implementation for FedAvg Server

AFL = True 
AFL_GRAD = True
CHECK_AVG_PARAM = False

class FedAFL(Server):
    def __init__(self, experiment, device, dataset,algorithm, model, batch_size, learning_rate, robust, gamma, num_glob_iters, local_epochs, sub_users, num_users, K,  times):
        super().__init__(experiment, device, dataset,algorithm, model[0], batch_size, learning_rate, robust, gamma, num_glob_iters,local_epochs, sub_users, num_users, times)

        # Initialize data for all  users
        #if(num_users == 1):
        #    i = 0
        #    train , test = dataset[2][i]
        #    user = UserAVG(device, i, train, test, model, batch_size, learning_rate, robust, gamma, local_epochs)
        #    self.target_domain = user
        #    user.set_target()
        #    self.users.append(user)
        #    self.total_train_samples += user.train_samples
        #    return
        if(dataset[0] == "Cifar10"):
            self.adv_option = [8/255,2/255,10]
        elif(dataset[0] == "Mnist"):
            self.adv_option = [0.3,0.01,40]
        elif(dataset[0] == "Emnist"):
            self.adv_option = [0.3,0.01,40]
        else:
            self.adv_option = [0,0,0]
            
        self.target_domain = None
        # Averaged model
        # self.resulting_model = copy.deepcopy(model)
        # Grads
        self.grad_learning_rate = learning_rate
        # Initialize lambdas
        lamdas_length = int(num_users*sub_users)
        self.lambdas = np.ones(lamdas_length) * 1.0 /(lamdas_length)
        self.learning_rate_lambda = 0.001
        print(f"lambdas learning rate: {self.learning_rate_lambda}")
        for i in range(num_users):
            train , test = dataset[2][i]
            user = UserAFL(device, i, train, test, model, batch_size, learning_rate, robust, gamma, local_epochs, K)
            if(self.robust < 0): # no robust, domain option
                if(i == dataset[1] or (i == num_users-1 and dataset[1] < 0)):
                    self.target_domain = user
                    user.set_target()
                    continue
            self.users.append(user)
            self.total_train_samples += user.train_samples
            
        print("Number of users / total users:",int(sub_users * num_users), " / " ,num_users)
        print("Finished creating AFL server.")

    def send_grads(self):
        assert (self.users is not None and len(self.users) > 0)
        grads = []
        for param in self.model.parameters():
            if param.grad is None:
                grads.append(torch.zeros_like(param.data))
            else:
                grads.append(param.grad)
        for user in self.users:
            user.set_grads(grads)

    def project(self, y):
        ''' algorithm comes from:
        https://arxiv.org/pdf/1309.1541.pdf
        '''
        if np.sum(y) <=1 and np.alltrue(y >= 0):
            return y
        u = sorted(y, reverse=True)
        x = []
        rho = 0
        for i in range(len(y)):
            if (u[i] + (1.0/(i+1)) * (1-np.sum(np.asarray(u)[:i]))) > 0:
                rho = i + 1
        lambda_ = (1.0/rho) * (1-np.sum(np.asarray(u)[:rho]))
        for i in range(len(y)):
            x.append(max(y[i]+lambda_, 0))
        return x  

    def train(self):
        # Assign all values of resulting model = 0
        for resulting_model_param in self.resulting_model.parameters():
            resulting_model_param.data = torch.zeros_like(resulting_model_param.data)
        # Training for clients
        for glob_iter in range(self.num_glob_iters):
            losses = []
            if(self.experiment):
                self.experiment.set_epoch( glob_iter + 1)
            print("-------------Round number: AFL", glob_iter, " -------------")

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
                _, loss = user.train(self.local_epochs)
                if AFL==True:
                    losses.append(loss.data.item()) # Collect loss from users
                     # print(f"losses: {losses}")

            if AFL == True: # Select AFL algorithms
                # Aggregate training result from users
                if AFL_GRAD == True: # Aggregate based on gradients
                    print("AFL Grads!!!")
                    self.AFL_aggregate_grads(self.selected_users, self.lambdas)
                    # Projection weights
                    for server_param in self.model.parameters():
                        server_param.data -= server_param.grad * self.grad_learning_rate
                else: # Aggregate based on weights
                    print("AFL Weights!!!")
                    self.AFL_aggregate_parameters(self.selected_users, self.lambdas)

                # Update lambdas
                for idx in range(len(self.lambdas)):
                    self.lambdas[idx] += self.learning_rate_lambda * losses[idx]
                # Project lambdas
                # print(f"lambdas before projection: {self.lambdas}")
                print(f"Type of lamdas before project: {type(self.lambdas)} -- len: {len(self.lambdas)}")
                self.lambdas = self.project(self.lambdas)
                # print(f"lambdas after projection: {self.lambdas}")
                print(f"Type of lamdas after project: {type(self.lambdas)} -- len: {len(self.lambdas)}")
                # Avoid probability 0
                self.lambdas = np.asarray(self.lambdas)
                lambdas_zeros = self.lambdas <= 1e-3
                # print(lambdas_zeros.sum())
                if lambdas_zeros.sum() > 0:
                    self.lambdas[lambdas_zeros] = 1e-3
                    self.lambdas /= self.lambdas.sum()
            else:
                # FedAvg
                self.aggregate_parameters(self.selected_users)

            # Check averaged weights
            if CHECK_AVG_PARAM == True:
                for server_param in self.model.parameters():
                    print(server_param.data)

            # Averaging model
            for server_param, resulting_model_param in zip(self.model.parameters(), self.resulting_model.parameters()):
                resulting_model_param.data = (resulting_model_param.data*glob_iter + server_param.data) * 1.0 / (glob_iter + 1)

        # Distribute the final model to all users        
        print(f"-----Testing on final model-----")
        for server_param, resulting_model_param in zip(self.model.parameters(), self.resulting_model.parameters()):
            server_param.data = resulting_model_param.data
            if CHECK_AVG_PARAM == True:
                print(server_param.data)
        self.send_parameters()
        self.evaluate()
        
        self.save_results()
        self.save_model()