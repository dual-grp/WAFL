import torch
import os
import torch.multiprocessing as mp

from utils.get_femnist_data import *
from FLAlgorithms.users.userDRFA import UserDRFA
from FLAlgorithms.servers.serverbase import Server
from utils.model_utils import read_data, read_domain_data
import numpy as np
import copy

# Implementation for FedDRFA Server
CHECK_AVG_PARAM = False

class FedDRFA(Server):
    def __init__(self, experiment, device, dataset,algorithm, model, batch_size, learning_rate, robust, gamma, num_glob_iters, local_epochs, sub_users, num_users, K,  times):
        super().__init__(experiment, device, dataset,algorithm, model[0], batch_size, learning_rate, robust, gamma, num_glob_iters,local_epochs, sub_users, num_users, times)

        self.sampling_size = sub_users # Fraction to select sampling size * num_users --- m in range [0; num_users]
        self.sample_number = 1 # A sample number -- t' will be selected randomly in range: [1; local_epochs]

        # Grads
        self.grad_learning_rate = learning_rate

        # Initialize lambdas
        if(self.robust < 0): # running on domain adaptation. the last user will be the target distribution to test
            lamdas_length = int(num_users*sub_users) - 1
        else:
            lamdas_length = int(num_users*sub_users) # all clients will be involved in selection for training

        self.lambdas = np.ones(lamdas_length) * 1.0 /(lamdas_length)  # Initialize lambdas_0
        self.learning_rate_lambda = 0.001
        
        print(f"learning rate for lambdas: {self.learning_rate_lambda}")
        if(dataset[0] == "Cifar10"):
            self.adv_option = [8/255,2/255,10]
        elif(dataset[0] == "Mnist"):
            self.adv_option = [0.3,0.01,40]
            learning_rate *= 10
        elif(dataset[0] == "Emnist"):
            self.adv_option = [0.3,0.01,40]
        elif(dataset[0] == "FeMnist"):
            self.adv_option = [0.3,0.01,40]
        else:
            self.adv_option = [0,0,0]

        self.target_domain = None

        for i in range(num_users):
            if dataset[0] == "FeMnist":
                train, test = get_user_dataset(i)
            else:
                train , test = dataset[2][i]
            user = UserDRFA(device, i, train, test, model, batch_size, learning_rate, robust, gamma, local_epochs, K)
            if(self.robust < 0): # no robust, domain option
                if(i == dataset[1] or (i == num_users-1 and dataset[1] < 0)):
                    self.target_domain = user
                    user.set_target()
                    continue
            self.users.append(user)
            self.total_train_samples += user.train_samples
        print("Number of users / total users:",int(sub_users * num_users), " / " ,num_users)
        print("Finished creating FedDRFA server.")

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

    def send_data_selected_users(self, users):
        assert (users is not None and len(users) > 0)
        # Broadcast t'
        for user in users:
            user.sample_number = copy.deepcopy(self.sample_number)
        # Broadcast w
        for user in self.users:
            user.set_parameters(self.model)
        # if(self.target_domain):
        #     self.target_domain.set_parameters(self.model)

    def send_sample_parameters(self, users):
        assert (users is not None and len(users) > 0)
        for user in self.users:
            user.set_parameters(self.sample_model)
        # if(self.target_domain):
        #     self.target_domain.set_parameters(self.sample_model)

    def DRFA_aggregate_parameters(self, users):
        assert (users is not None and len(users) > 0)
        for param in self.model.parameters():
            param.data = torch.zeros_like(param.data)
        total_users = 0
        for user in users:
            total_users += 1
        print(total_users)
        for user in users:
            self.add_parameters(user, 1.0 / total_users)

    def DRFA_aggregate_parameters_with_lambdas(self, users, lambdas):
        assert (users is not None and len(users) > 0)
        for param in self.model.parameters():
            param.data = torch.zeros_like(param.data)
        total_train = 0

        # print(f"lambdas: {lambdas}")

        for user, lambda_ in zip(users, lambdas):
            self.add_parameters(user, lambda_)

    def DRFA_add_sample_parameters(self, user, ratio):
        for server_param, user_param in zip(self.sample_model.parameters(), user.get_sample_parameters()):
            server_param.data = server_param.data + user_param.data.clone() * ratio

    def DRFA_aggregate_sample_parameters(self, users):
        assert (users is not None and len(users) > 0)

        for param in self.sample_model.parameters():
            param.data = torch.zeros_like(param.data)

        total_users = 0
        for user in users:
            total_users += 1
        print(f"total user: {total_users}")
        for user in users:
            self.DRFA_add_sample_parameters(user, 1.0 / total_users)

    def project(self, y):
        # ''' algorithm comes from:
        # https://arxiv.org/pdf/1309.1541.pdf
        # '''
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
        # Assign all values of  models = 0
        for resulting_model_param in self.resulting_model.parameters():
            resulting_model_param.data = torch.zeros_like(resulting_model_param.data)
        
        for sample_model_param in self.sample_model.parameters():
            sample_model_param.data = torch.zeros_like(sample_model_param.data)

        # Training for clients
        for glob_iter in range(self.num_glob_iters):
            # self.learning_rate_lambda *= 0.9 # learning rate decay
            losses = []
            if(self.experiment):
                self.experiment.set_epoch( glob_iter + 1)
            print("-------------Round number: FedDRFA", glob_iter, " -------------")

            # Create a random sample number
            self.sample_number = torch.randint(low=0,high=(self.local_epochs),size=(1,)).item() # t'

            # Select subset of user for training
            self.selected_users = self.select_users(glob_iter, self.sub_users)

            # Broadcast w_s and t' to selected clients
            self.send_data_selected_users(self.selected_users)

            # Evaluate model each interation
            self.evaluate()

            if(self.robust < 0):
                self.evaluate_on_target()

            if(self.robust > 0):
                self.adv_users = self.select_users(glob_iter, self.robust)
                self.evaluate_robust('pgd', self.adv_option)
                # print("Evaluate on target!")
                #self.evaluate_robust('fgsm', self.adv_option)

            # Select subset of user for training
            for user in self.selected_users:
                _,loss = user.train(self.local_epochs)
                losses.append(loss.data.item())
            # Aggregate W^{s+1}
            # self.DRFA_aggregate_parameters(self.selected_users)
            self.DRFA_aggregate_parameters_with_lambdas(self.selected_users, self.lambdas) # aggregate with lambdas values

            # Aggregate W^{t'}
            self.DRFA_aggregate_sample_parameters(self.selected_users)

            # # Update lamdas
            # Select subset of users
            self.selected_users = self.select_users(glob_iter, self.sub_users)

            # Broadcast W^{t'}
            self.send_sample_parameters(self.selected_users)

            # Computing loss in selected users
            for user in self.selected_users:
                _,loss = user.train(self.local_epochs)
                losses.append(loss.data.item()) # gathering loss from users

            # Gradient ascent for finding lamdas
            for idx in range(len(self.lambdas)):
                self.lambdas[idx] += self.learning_rate_lambda * losses[idx] * 1.0 / self.sub_users * self.local_epochs

            # print(f"lambdas before projection: {self.lambdas}")
            self.lambdas = self.project(self.lambdas)
            # print(f"lambdas after projection: {self.lambdas}")
            # Avoid probability 0
            self.lambdas = np.asarray(self.lambdas)
            lambdas_zeros = self.lambdas <= 1e-3
            # print(lambdas_zeros.sum())
            if lambdas_zeros.sum() > 0:
                self.lambdas[lambdas_zeros] = 1e-3
                self.lambdas /= self.lambdas.sum()
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
        if(self.robust > 0):
            self.adv_users = self.select_users(glob_iter, self.robust)
            self.evaluate_robust('pgd', self.adv_option)
        if(self.robust < 0):
            self.evaluate_on_target()
        self.save_results()
        self.save_model()