#!/usr/bin/env python
from comet_ml import Experiment
from FLAlgorithms.servers.serveravg import FedAvg
from FLAlgorithms.servers.serveravgAT import FedAvgAT
from FLAlgorithms.servers.serverrobF import FedRob
from utils.model_utils import read_domain_data
from FLAlgorithms.trainmodel.models import *
from utils.plot_utils import *
from utils.train_utils import get_model
import torch
torch.manual_seed(0)
from utils.options import args_parser

# import comet_ml at the top of your file

# Create an experiment with your api key:
def main(experiment, dataset, algorithm, batch_size, learning_rate, robust, gamma, num_glob_iters,
         local_epochs, sub_user, numusers, K, times, commet, gpu):
    
    # Get device status: Check GPU or CPU
    device = torch.device("cuda:{}".format(gpu) if torch.cuda.is_available() and gpu != -1 else "cpu")
    args.device = torch.device("cuda:{}".format(gpu) if torch.cuda.is_available() and gpu != -1 else "cpu")
    domain_data = dataset[0], dataset[1], read_domain_data(dataset[0])
    
    for i in range(times):
        print("---------------Running time:------------",i)
        # Generate model
        model =  get_model(args)

        # select algorithm
        if(algorithm == "FedAvg"):
            if(commet):
                experiment.set_name(dataset[0] + "_" + algorithm + "_" + model[1] + "_" + str(batch_size) + "_" + str(learning_rate) + "_" + str(num_glob_iters) + "_"+ str(local_epochs) + "_"+ str(numusers))
            server = FedAvg(experiment, device, domain_data, algorithm, model, batch_size, learning_rate, robust, gamma, num_glob_iters, local_epochs, sub_user, numusers, i)
        
        elif(algorithm == "FedAvgAT"):
            if(commet):
                experiment.set_name(dataset[0] + "_" + algorithm + "_" + model[1] + "_" + str(batch_size) + "_" + str(learning_rate) + "_" + str(num_glob_iters) + "_"+ str(local_epochs) + "_"+ str(numusers))
            server = FedAvgAT(experiment, device, domain_data, algorithm, model, batch_size, learning_rate, robust, gamma, num_glob_iters, local_epochs, sub_user, numusers, i)

        elif(algorithm == "FedRob"):
            if(commet):
                experiment.set_name(dataset[0] + "_" + algorithm + "_" + model[1] + "_" + str(batch_size) + "_" + str(learning_rate) + "_" + str(num_glob_iters) + "_"+ str(local_epochs) + "_"+ str(numusers))
            server = FedRob(experiment, device, domain_data, algorithm, model, batch_size, learning_rate, robust, gamma, num_glob_iters, local_epochs, sub_user, numusers, i)

        else:
            print("Algorithm is invalid")
            return

        server.train()
        server.test()

    average_data(num_users=numusers, loc_ep1=local_epochs, Numb_Glob_Iters=num_glob_iters, lamb=gamma,learning_rate=learning_rate, robust = robust, algorithms=algorithm, batch_size=batch_size, dataset=dataset[0], k = K,times = times)

if __name__ == "__main__":
    args = args_parser()
    print("=" * 80)
    print("Summary of training process:")
    print("Algorithm: {}".format(args.algorithm))
    print("Batch size: {}".format(args.batch_size))
    print("Learning rate: {}".format(args.learning_rate))
    print("Robust parameter {}".format(args.gamma))
    print("Robust Option, fraction of attack clients: {}".format(args.robust))
    print("Subset of users      : {}".format(args.subusers))
    print("Number of global rounds       : {}".format(args.num_global_iters))
    print("Number of local rounds       : {}".format(args.local_epochs))
    print("Dataset       : {}".format(args.dataset))
    print("Local Model       : {}".format(args.model))
    print("=" * 80)

    if(args.commet):
        # Create an experiment with your api key:
        experiment = Experiment(
            api_key="VtHmmkcG2ngy1isOwjkm5sHhP",
            project_name="domain-adaptation",
            workspace="federated-learning-exp",
        )

        hyper_params = {
            "dataset":args.dataset,
            "algorithm" : args.algorithm,
            "model":args.model,
            "batch_size":args.batch_size,
            "learning_rate":args.learning_rate,
            "target" : args.target, 
            "robust" : args.robust,
            "num_glob_iters":args.num_global_iters,
            "local_epochs":args.local_epochs,
            "numusers": args.subusers,
            "times" : args.times,
            "gpu": args.gpu,
        }
        
        experiment.log_parameters(hyper_params)
    else:
        experiment = 0
    main(
        experiment= experiment,
        dataset= [args.dataset, args.target],
        algorithm = args.algorithm,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        robust = args.robust, 
        gamma = args.gamma,
        num_glob_iters=args.num_global_iters,
        local_epochs=args.local_epochs,
        sub_user = args.subusers,
        numusers = args.numusers,
        K=args.K,
        times = args.times,
        commet = args.commet,
        gpu=args.gpu
        )


