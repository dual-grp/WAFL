import torch
import os
import torch.multiprocessing as mp

from FLAlgorithms.users.useravg import UserAVG
from FLAlgorithms.servers.serverbase import Server
from utils.model_utils import read_data, read_domain_data
import numpy as np
import copy
from otdd.pytorch.datasets import load_torchvision_data
from otdd.pytorch.distance import DatasetDistance
from scipy.optimize import linprog
# Implementation for DA Server

# AFL = False 
# AFL_GRAD = True

class DA(Server):
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
        self.dataset = dataset[0]
        print(f"Experiment is running on latest: {self.dataset}")
        if(dataset[0] == "Cifar10"):
            self.adv_option = [8/255,2/255,10]
        elif(dataset[0] == "Mnist"):
            self.adv_option = [0.3,0.01,40]
        elif(dataset[0] == "Emnist"):
            self.adv_option = [0.3,0.01,40]
        else:
            self.adv_option = [0,0,0]
        
        print(f"dataset using in the training: {dataset[0]}")
        self.target_domain = None

        for i in range(num_users):
            train , test = dataset[2][i]
            user = UserAVG(device, i, train, test, model, batch_size, learning_rate, robust, gamma, local_epochs, K)
            if(self.robust < 0): # no robust, domain option
                if(i == dataset[1] or (i == num_users-1 and dataset[1] < 0)):
                    self.target_domain = user
                    user.set_target()
                    continue
            self.users.append(user)
            self.total_train_samples += user.train_samples
            
        print("Number of users / total users:",int(sub_users * num_users), " / " ,num_users)
        print("Finished creating FedAvg server.")

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

    def DA_aggregate_parameters_with_lambdas(self, users, lambdas):
        assert (users is not None and len(users) > 0)
        for param in self.model.parameters():
            param.data = torch.zeros_like(param.data)
        total_train = 0
        for user, lambda_ in zip(users, lambdas):
            self.add_parameters(user, lambda_)

    def DA_find_lambdas(self):
        three_datasets = ['MNIST', 'SVHN','USPS']
        if self.dataset == 'msda1':
            source = [three_datasets[0], three_datasets[1]]
            target = [three_datasets[2]]
        elif self.dataset == 'msda2':
            source = [three_datasets[0], three_datasets[2]]
            target = [three_datasets[1]]
        else:
            source = [three_datasets[1], three_datasets[2]]
            target = [three_datasets[0]]
        print(f"source: {source}, target: {target[0]}")
        # Load datasets
        loaders_tgt = load_torchvision_data(target[0], valid_size=0, to3channels=True,resize = 28, maxsize=2000)[0]
        loader_source = []
        # Estimate Optimal Transport Dataset Distance between Datasets
        for i in range(len(source)):
            loader_source.append(load_torchvision_data(source[i],  valid_size=0, to3channels=True,resize = 28, maxsize=2000)[0])
            dist = []
            distance = []
            # Instantiate distance
            for i in range(len(loader_source)):
                dist.append(DatasetDistance(loader_source[i]['train'], loaders_tgt['train'],
                                        inner_ot_method = 'exact',
                                        debiased_loss = True,
                                        p = 2, entreg = 1e-1,
                                        device='cpu'))
                distance.append(dist[i].distance(maxsamples = 1000))
            print(f"distance from {source[i]} to {target[0]}: {distance[i]}")

        # Solve linear programming problem for lambdas
        # Define the constraint sumation of lambdas = 1
        A = [[1,1], [-1,-1]]
        b = [1, -1]
        # Define the constrant each lambda >= 0
        bounds = []
        for i in range(len(distance)):
            bounds.append((0, None))
        # Solve optimization problem
        res = linprog(distance, A_ub=A, b_ub=b,bounds=bounds)

        if(res.success):
            lambdas = copy.deepcopy(res.x)
        else:
            lambdas = [0] * len(distance)
        zeros_index = lambdas < 1e-6
        # Remove lambdas with small effect on objective function
        lambdas[zeros_index] = 0
        return lambdas

    def train(self):
        # Assign all values of resulting model = 0
        for resulting_model_param in self.resulting_model.parameters():
            resulting_model_param.data = torch.zeros_like(resulting_model_param.data)

        # Finding lambdas for domain adaptation
        lambdas_star = self.DA_find_lambdas()

        # Training for clients
        for glob_iter in range(self.num_glob_iters):
            losses = []
            if(self.experiment):
                self.experiment.set_epoch( glob_iter + 1)
            print("-------------Round number: DA with lambdas", glob_iter, " -------------")

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
                loss = user.train(self.local_epochs)

            self.DA_aggregate_parameters_with_lambdas(self.selected_users, lambdas_star)
        
        self.save_results()
        self.save_model()