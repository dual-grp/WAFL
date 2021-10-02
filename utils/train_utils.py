from torchvision import datasets, transforms
from FLAlgorithms.trainmodel.models import *

def get_model(args):
    
    if(args.model == "mclr"):
        if(args.dataset == "human_activity"):
            model = Mclr_Logistic(561,6).to(args.device)
        elif(args.dataset == "gleam"):
            model = Mclr_Logistic(561,6).to(args.device)
        elif(args.dataset == "vehicle_sensor"):
            model = Mclr_Logistic(100,2).to(args.device)
        elif(args.dataset == "Synthetic"):
            model = Mclr_Logistic(60,10).to(args.device)
        elif(args.dataset == "fiveDigit"):
            model = LogisticModel().to(args.device)
        elif(args.dataset == "Mnist"):
            model = Mclr_Logistic().to(args.device)
        elif(args.dataset == "Mnist"):   
            model = Mclr_Logistic(784,62).to(args.device)
        else:
            # Domain Adaptation Convex Model (e.g., mm->mt)
            model = LogisticModel().to(args.device)
        
    elif(args.model == "dnn"):
        if(args.dataset == "human_activity"):
            model = DNN(561,100,12).to(args.device)
        elif(args.dataset == "gleam"):
            model = DNN(561,20,6).to(args.device)
        elif(args.dataset == "vehicle_sensor"):
            model = DNN(100,20,2).to(args.device)
        elif(args.dataset == "Synthetic"):
            model = DNN(60,20,10).to(args.device)
        elif(args.dataset == "fiveDigit"):
            model = FeedFwdModel(2352,10).to(args.device)
            #model = DANCNNModel().to(args.device)
        else:#(dataset == "Mnist"):
            model = DNN2().to(args.device)
        
    elif(args.model == "cnn"):
        if(args.dataset == "Cifar10"):
            #model = CNNCifar().to(args.device)
            model = LeNet().to(args.device)
        if(args.dataset == "Emnist"):
            model = CNNEmnist(numlabels = 62).to(args.device)
        if(args.dataset == "Mnist"):
            model = CNNEmnist().to(args.device)
            #model = torch.hub.load('pytorch/vision:v0.10.0', 'alexnet', pretrained=True).to(args.device)
            #model.classifier[4] = torch.nn.Linear(4096,1024)
            #model.classifier[6] = torch.nn.Linear(1024,10)
        else:
            # Domain Adaptation Non-convex Model (e.g., mm->mt)
            model = CNNDA().to(args.device)
    else:
        exit('Error: unrecognized model')
    return model,args.model