import torch
import os
import numpy as np
import h5py
import copy

class Server:
    def __init__(self, experiment, device, dataset, algorithm, model, batch_size, learning_rate ,robust , L_k,
                 num_glob_iters, local_epochs, sub_users, num_users, times):

        # Set up the main attributes
        self.device = device
        self.dataset = dataset
        self.num_glob_iters = num_glob_iters
        self.local_epochs = local_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.total_train_samples = 0
        self.model = copy.deepcopy(model)
        self.resulting_model = copy.deepcopy(model) # Add final model
        self.sample_model = copy.deepcopy(model) # Add sample model for DRFA - W_t'
        self.users = []
        self.selected_users = []
        self.num_users = num_users
        self.sub_users = sub_users
        self.robust = robust
        self.L_k = L_k
        self.algorithm = algorithm
        self.rs_train_acc, self.rs_train_loss, self.rs_glob_acc, self.rs_target_acc, self.rs_test_loss , self.robust_glob_acc_pgd, self.robust_glob_acc_fgsm, self.robust_glob_loss_pgd, self.robust_glob_loss_fgsm = [], [], [], [], [] , [] , [], [], []
        self.times = times
        self.experiment = experiment
        self.sub_data = 0
        # Initialize the server's grads to zeros
        #for param in self.model.parameters():
        #    param.data = torch.zeros_like(param.data)
        #    param.grad = torch.zeros_like(param.data)
        #self.send_parameters()

    def get_data(self,train,test):
        if(self.sub_data == 1):
            train = train[int(0.95*len(train)):]
            test = test[int(0.8*len(test)):]
        else:
            train = train[int(0.8*len(train)):]
            test = test[int(0.6*len(test)):]
        return train, test

    def get_partion(self, total_users):
        if(self.sub_data):
            if(self.sub_data == 1):
                partion = int(0.9 * total_users)
            else:
                partion = int(0.5 * total_users)
        randomList = np.random.choice(range(0, total_users), partion, replace =False)
        return randomList

    def aggregate_grads(self):
        assert (self.users is not None and len(self.users) > 0)
        for param in self.model.parameters():
            param.grad = torch.zeros_like(param.data)
        for user in self.users:
            self.add_grad(user, user.train_samples / self.total_train_samples)

    def add_grad(self, user, ratio):
        user_grad = user.get_grads()
        for idx, param in enumerate(self.model.parameters()):
            param.grad = param.grad + user_grad[idx].clone() * ratio

    def send_parameters(self):
        assert (self.users is not None and len(self.users) > 0)
        for user in self.users:
            user.set_parameters(self.model)
        if(self.target_domain):
            self.target_domain.set_parameters(self.model)
    
    def add_parameters(self, user, ratio):
        model = self.model.parameters()
        for server_param, user_param in zip(self.model.parameters(), user.get_parameters()):
            server_param.data = server_param.data + user_param.data.clone() * ratio

    def aggregate_parameters(self, users):
        assert (users is not None and len(users) > 0)
        for param in self.model.parameters():
            param.data = torch.zeros_like(param.data)
        total_train = 0
        #if(self.num_users = self.to)
        for user in users:
            total_train += user.train_samples
        for user in users:
            self.add_parameters(user, user.train_samples / total_train)

    def AFL_aggregate_parameters(self, users, lambdas):
        # print("AFL aggregate parameters from Tung Anh!!!")
        assert (users is not None and len(users) > 0)
        for param in self.model.parameters():
            param.data = torch.zeros_like(param.data)
        total_train = 0

        for user in users:
            total_train += user.train_samples
        #if(self.num_users = self.to)

        total_weight = 0 
        for lambda_ in lambdas:
            total_weight += lambda_ # Sumation over all lambda values
        # print(f"total_weight: {total_weight}")
        for user, lambda_ in zip(users, lambdas):
            # print(f"lambdas_ratio: {round(lambda_ * 1.0 / total_weight, 3)} | data ratio: {round(user.train_samples / total_train, 3)}")
            self.add_parameters(user, lambda_ * 1.0 / total_weight)

    def AFL_aggregate_grads(self, users, lambdas):
        assert (self.users is not None and len(self.users) > 0)
        for param in self.model.parameters():
            param.grad = torch.zeros_like(param.data)

        total_weight = 0 
        for lambda_ in lambdas:
            total_weight += lambda_ # Sumation over all lambda values
        for user, lambda_ in zip(users, lambdas):
            self.add_grad(user, lambda_ * 1.0 / total_weight) # Add grads based on lambda values

    def save_model(self):
        model_path = os.path.join("models", self.dataset[0])
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        torch.save(self.model, os.path.join(model_path, "server" + ".pt"))

    def load_model(self):
        model_path = os.path.join("models", self.dataset, "server" + ".pt")
        assert (os.path.exists(model_path))
        self.model = torch.load(model_path)

    def model_exists(self):
        return os.path.exists(os.path.join("models", self.dataset, "server" + ".pt"))
    
    def select_users(self, round, fac_users):
        '''selects num_clients clients weighted by number of samples from possible_clients
        Args:
            num_clients: number of clients to select; default 20
                note that within function, num_clients is set to
                min(num_clients, len(possible_clients))
        
        Return:
            list of selected clients objects
        '''
        if(fac_users == 1):
            print("All users are selected")
            return self.users
        num_users = int(fac_users * len(self.users))
        num_users = min(num_users, len(self.users))
        np.random.seed(round * (self.times + 2))
        return np.random.choice(self.users, num_users, replace=False) #, p=pk)

    # Save loss, accurancy to h5 fiel
    def save_results(self):
        dir_path = "./results"
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
        #alg = self.dataset[1] + "_" + self.algorithm
        alg = str(self.dataset[0]) + "_" + self.algorithm
        alg = alg + "_" + str(self.learning_rate) + "_" + str(self.robust) + "_" + str(self.L_k) + "_" + str(self.num_users) + "u" + "_" + str(self.batch_size) + "b" + "_" + str(self.local_epochs) 
        #if(self.sub_data):
        #    alg = alg + "_" + "subdata"
        alg = alg + "_" + str(self.times)
        if (len(self.rs_glob_acc) != 0 &  len(self.rs_train_acc) & len(self.rs_train_loss)) :
            with h5py.File("./results/"+'{}.h5'.format(alg, self.local_epochs), 'w') as hf:
                hf.create_dataset('rs_glob_acc', data=self.rs_glob_acc)
                hf.create_dataset('rs_train_acc', data=self.rs_train_acc)
                hf.create_dataset('rs_train_loss', data=self.rs_train_loss)
                hf.create_dataset('rs_test_loss', data=self.rs_train_loss)

                hf.create_dataset('rs_target_acc', data=self.rs_target_acc)

                hf.create_dataset('rs_robust_pgd_acc', data=self.robust_glob_acc_pgd)
                hf.create_dataset('rs_robust_pgd_loss', data=self.robust_glob_loss_pgd)
                hf.create_dataset('rs_robust_fgsm_acc', data=self.robust_glob_acc_fgsm)
                hf.create_dataset('rs_robust_fgsm_loss', data=self.robust_glob_loss_fgsm)
                hf.close()

    def test(self):
        '''tests self.latest_model on given clients
        '''
        num_samples = []
        tot_correct = []
        losses = []
        for c in self.users:
            ct,cl, ns = c.test()
            tot_correct.append(ct*1.0)
            num_samples.append(ns)
            losses.append(cl*1.0)
        ids = [c.id for c in self.users]
        return ids, num_samples, tot_correct, losses

    def test_robust(self, attack_mode = 'pgd', adv_option = [0.3, 0.01, 40]):
        'can choose a fraction of user which is attattack: let say just choose 30-> 50% clients are attracked'
        robust_correct = []
        num_samples = []
        # choose a subset number of clients be attack-clients
        list_attack = []
        losses = []
        for c in self.adv_users:
            list_attack.append(c.id)
            ct,cl, ns = c.test_robust(attack_mode, adv_option)
            robust_correct.append(ct*1.0)
            num_samples.append(ns)
            losses.append(cl*1.0)

        for c in self.users:
            if c.id not in list_attack:
                ct, cl, ns = c.test()
                robust_correct.append(ct*1.0)
                num_samples.append(ns)
                losses.append(cl*1.0)

        ids = [c.id for c in self.users]
        return ids, num_samples, robust_correct, losses

    def train_error_and_loss(self):
        num_samples = []
        tot_correct = []
        losses = []
        for c in self.users:
            ct, cl, ns = c.train_error_and_loss() 
            tot_correct.append(ct*1.0)
            num_samples.append(ns)
            losses.append(cl*1.0)
        
        ids = [c.id for c in self.users]
        #groups = [c.group for c in self.clients]

        return ids, num_samples, tot_correct, losses

    def evaluate(self):
        stats = self.test()  
        stats_train = self.train_error_and_loss()
        glob_acc = np.sum(stats[2])*1.0/np.sum(stats[1])
        train_acc = np.sum(stats_train[2])*1.0/np.sum(stats_train[1])
        # train_loss = np.dot(stats_train[3], stats_train[1])*1.0/np.sum(stats_train[1])
        train_loss = sum([x * y for (x, y) in zip(stats_train[1], stats_train[3])]).item() / np.sum(stats_train[1])
        test_loss = sum([x * y for (x, y) in zip(stats[1], stats[3])]).item() / np.sum(stats[1])
        self.rs_glob_acc.append(glob_acc)
        self.rs_train_acc.append(train_acc)
        self.rs_train_loss.append(train_loss)
        self.rs_test_loss.append(test_loss)
        if(self.experiment):
            self.experiment.log_metric("glob_acc_source",glob_acc)
            self.experiment.log_metric("train_acc_source",train_acc)
            self.experiment.log_metric("train_loss_source",train_loss)
            self.experiment.log_metric("test_loss_source",test_loss)
        #print("stats_train[1]",stats_train[3][0])
        print("Average Global Accurancy on all Source Domain : ", glob_acc)
        print("Average Global Trainning Accurancy on all Source Domain: ", train_acc)
        print("Average Global Trainning Loss on all Source Domain: ",train_loss)
        print("Average Global Test Loss on all Source Domain: ",test_loss)
    
    def evaluate_robust(self, attack_mode = 'pgd', adv_option = [0.3, 0.01]):
        stats = self.test_robust(attack_mode, adv_option)

        if(attack_mode == 'pgd'):
            robust_glob_acc_pgd = np.sum(stats[2])*1.0/np.sum(stats[1])
            robust_glob_loss_pgd = sum([x * y for (x, y) in zip(stats[1], stats[3])]).item() / np.sum(stats[1])
            self.robust_glob_acc_pgd.append(robust_glob_acc_pgd)
            self.robust_glob_loss_pgd.append(robust_glob_loss_pgd)
            if(self.experiment):
                self.experiment.log_metric("robust_acc_pgd",robust_glob_acc_pgd)
                self.experiment.log_metric("robust_loss_pgd",robust_glob_loss_pgd)
            #print("stats_train[1]",stats_train[3][0])
            print("Average robust pgd accurancy on all: ", robust_glob_acc_pgd)
            print("Average robust pgd loss on all: ", robust_glob_loss_pgd)

        if(attack_mode == 'fgsm'):
            robust_glob_acc_fgsm = np.sum(stats[2])*1.0/np.sum(stats[1])
            self.robust_glob_acc_fgsm.append(robust_glob_acc_fgsm)
            robust_glob_loss_fgsm = sum([x * y for (x, y) in zip(stats[1], stats[3])]).item() / np.sum(stats[1])
            self.robust_glob_loss_fgsm.append(robust_glob_loss_pgd)
            if(self.experiment):
                self.experiment.log_metric("robust_acc_fgsm",robust_glob_acc_fgsm)
                self.experiment.log_metric("robust_loss_fgsm",robust_glob_loss_fgsm)
            #print("stats_train[1]",stats_train[3][0])
            print("Average robust fgsm accurancy on all: ", robust_glob_acc_fgsm)
            print("Average robust fgsm loss on all: ", robust_glob_loss_fgsm)


    def evaluate_on_target(self):
        # evaluate 
        glob_acc_target = self.target_domain.test_domain()
        self.rs_target_acc.append(glob_acc_target)
        if(self.experiment):
            self.experiment.log_metric("glob_acc_target",glob_acc_target)
        #print("stats_train[1]",stats_train[3][0])
        print("Average Global Accurancy on Target Domain: ", glob_acc_target)


