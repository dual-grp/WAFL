import numpy as np
import os
import torch
import torch.nn as nn

import torchvision.transforms as transforms
from torch.utils.data.dataset import Dataset

import random

import scipy.io as sio
#from scipy.io import loadmat
import sys
import pickle

import argparse
import matplotlib.pyplot as plt
from matplotlib.ticker import StrMethodFormatter
plt.rcParams.update({'font.size': 14})

## Ignore Warnings
import warnings
warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning) 

### Base Folder
DATA_FOLDER = './original_data'

### Data Statistics
def print_image_data_stats(data_train, labels_train, data_test, labels_test):
    print("\nData: ")
    print(" -- Train Set: ({},{}), Range: [{:.3f}, {:.3f}], Labels: {},..,{}".format(
        data_train.shape, labels_train.shape, np.min(data_train), np.max(data_train),
        np.min(labels_train), np.max(labels_train)))
    print(" -- Test Set: ({},{}), Range: [{:.3f}, {:.3f}], Labels: {},..,{}".format(
        data_test.shape, labels_test.shape, np.min(data_train), np.max(data_train),
        np.min(labels_test), np.max(labels_test)))
  
### Load Source and Target Data Functions
def get_source_mnist():
    '''Return MNIST train/test data and labels as numpy arrays'''
    print("-- loading full mnist data")
    mnist_data = sio.loadmat(DATA_FOLDER + '/mnist_data.mat')

    mnist_train = np.reshape(mnist_data['train_28'], (55000, 28, 28, 1))
    mnist_test = np.reshape(mnist_data['test_28'], (10000, 28, 28, 1))

    # Turn to the 3 channel image with C*H*W
    mnist_data_train = np.concatenate([mnist_train, mnist_train, mnist_train], 3)
    mnist_data_test = np.concatenate([mnist_test, mnist_test, mnist_test], 3)

    mnist_labels_train = mnist_data['label_train']
    mnist_labels_test = mnist_data['label_test']

    train_label = np.argmax(mnist_labels_train, axis=1)
    test_label = np.argmax(mnist_labels_test, axis=1) 

    x_train, y_train = mnist_data_train.transpose(0, 3, 1, 2).astype(np.float32)/255.0, train_label
    x_test, y_test = mnist_data_test.transpose(0, 3, 1, 2).astype(np.float32)/255.0, test_label
    
    return x_train, y_train, x_test, y_test

def get_source_mnist_pkl():

    with open(DATA_FOLDER + "/mnist.pkl", "br") as fh:
        data = pickle.load(fh)

    train_imgs = data[0]
    test_imgs = data[1]
    train_labels = data[2]
    test_labels = data[3]

    mnist_train = [x.reshape(28,28,1).astype(np.float32) for x in train_imgs]
    mnist_data_train = np.concatenate([mnist_train, mnist_train, mnist_train], 3)
    mnist_data_train = mnist_data_train.transpose(0, 3, 1, 2)
    mnist_train_labels = train_labels.T[0].astype(int)
    #mnistm_train = [(x, y) for x, y in zip(train_data_2, train_labels.T[0])]

    mnist_test = [x.reshape(28,28,1).astype(np.float32) for x in test_imgs]
    mnist_data_test = np.concatenate([mnist_test, mnist_test, mnist_test], 3)
    mnist_data_test = mnist_data_test.transpose(0, 3, 1, 2)
    mnist_test_labels = test_labels.T[0].astype(int)

    #mnistm_test = [(x, y) for x, y in zip(test_data_2, test_labels.T[0])]

    return mnist_data_train, mnist_train_labels, mnist_data_test, mnist_test_labels

def get_source_mnistm():
    '''Return MNIST train/test data and labels as numpy arrays'''
    print("-- loading mnist-m data")
    mnistm_train = sio.loadmat(DATA_FOLDER + '/mnistm_train.mat')
    mnistm_test = sio.loadmat(DATA_FOLDER + '/mnistm_test.mat')

    mnistm_train_data = mnistm_train['train_data']
    mnistm_test_data = mnistm_test['test_data']

    # Get labels
    mnistm_labels_train = mnistm_train['train_label']
    mnistm_labels_test = mnistm_test['test_label']

    train_label = mnistm_labels_train.reshape(1,-1)[0]
    test_label = mnistm_labels_test.reshape(1,-1)[0]  
    
    x_train, y_train = mnistm_train_data.transpose(0, 3, 1, 2).astype(np.float32), train_label
    x_test, y_test = mnistm_test_data.transpose(0, 3, 1, 2).astype(np.float32), test_label
    
    return x_train, y_train, x_test, y_test

def get_source_usps():
    '''Return MNIST train/test data and labels as numpy arrays'''
    print("-- loading usps data")
    usps_dataset = sio.loadmat(DATA_FOLDER + '/usps_28x28.mat')
    usps_dataset = usps_dataset["dataset"]

    usps_train = usps_dataset[0][0]


    train_label = usps_dataset[0][1]
    train_label = train_label.reshape(-1)
    train_label[train_label == 10] = 0


    frac = 0.99/255
    usps_train *= 255
    usps_train = usps_train*frac + 0.01


    usps_train = np.concatenate([usps_train, usps_train, usps_train], 1)
    # usps_train = np.tile(usps_train, (4, 1, 1, 1))

    usps_test = usps_dataset[1][0]
    usps_test *= 255
    usps_test = usps_test*frac + 0.01

    test_label = usps_dataset[1][1]
    test_label = test_label.reshape(-1)
    test_label[test_label == 10] = 0

    usps_test = np.concatenate([usps_test, usps_test, usps_test], 1)
    # usps_test = np.tile(usps_test, (4, 1, 1, 1))  
    
    x_train, y_train = usps_train, train_label
    x_test, y_test = usps_test, test_label
    
    return x_train, y_train, x_test, y_test

def get_target_mnist_pkl(num_train, num_test):

    print("-- loading full mnist data")

    with open(DATA_FOLDER + "/mnist.pkl", "br") as fh:
      data = pickle.load(fh)

    train_imgs = data[0]
    test_imgs = data[1]
    train_labels = data[2]
    test_labels = data[3]

    mnist_train = [x.reshape(28,28,1).astype(np.float32) for x in train_imgs]
    mnist_data_train = np.concatenate([mnist_train, mnist_train, mnist_train], 3)
    mnist_data_train = mnist_data_train.transpose(0, 3, 1, 2)
    mnist_train_labels = train_labels.T[0].astype(int)
    #mnistm_train = [(x, y) for x, y in zip(train_data_2, train_labels.T[0])]

    mnist_test = [x.reshape(28,28,1).astype(np.float32) for x in test_imgs]
    mnist_data_test = np.concatenate([mnist_test, mnist_test, mnist_test], 3)
    mnist_data_test = mnist_data_test.transpose(0, 3, 1, 2)
    mnist_test_labels = test_labels.T[0].astype(int)


    if (num_train != 0 and num_test != 0):
      mnist_data_train = mnist_data_train[:num_train]
      mnist_train_labels = mnist_train_labels[:num_train]

      mnist_data_test = mnist_data_test[:num_test]
      mnist_test_labels = mnist_test_labels[:num_test]

    #mnistm_test = [(x, y) for x, y in zip(test_data_2, test_labels.T[0])]
    print_image_data_stats(mnist_data_train, mnist_train_labels, mnist_data_test, mnist_test_labels)


    train_data = [(x, y) for x, y in zip(mnist_data_train, mnist_train_labels)]
    test_data = [(x, y) for x, y in zip(mnist_data_test, mnist_test_labels)] 

    return train_data, test_data

def get_target_mnistm_full(num_train, num_test):
    print("-- loading mnist-m data")
    
    mnistm_train = sio.loadmat(DATA_FOLDER + '/mnistm_train.mat')
    mnistm_test = sio.loadmat(DATA_FOLDER + '/mnistm_test.mat')

    mnistm_train_data = mnistm_train['train_data']
    mnistm_test_data = mnistm_test['test_data']

    mnistm_train_data = mnistm_train_data.transpose(0, 3, 1, 2).astype(np.float32)
    mnistm_test_data = mnistm_test_data.transpose(0, 3, 1, 2).astype(np.float32)

    # get labels
    mnistm_labels_train = mnistm_train['train_label']
    mnistm_labels_test = mnistm_test['test_label']

    # random sample 25000 from train dataset and random sample 9000 from test dataset
    inds = np.random.permutation(mnistm_train_data.shape[0])

    train_label = mnistm_labels_train.reshape(1,-1)[0]

    mnistm_train_data = mnistm_train_data[inds]
    train_label = train_label[inds]

    test_label = mnistm_labels_test.reshape(1,-1)[0]
    '''
    print('mnist_m train X shape->',  mnistm_train_data.shape)
    print('mnist_m train y shape->',  train_label.shape)
    print('mnist_m test X shape->',  mnistm_test_data.shape)
    print('mnist_m test y shape->', test_label.shape)
    '''

    if (num_train != 0 and num_test != 0):
      mnistm_train_data = mnistm_train_data[:num_train]
      train_label = train_label[:num_train]

      mnistm_test_data = mnistm_test_data[:num_test]
      test_label = test_label[:num_test]

    print_image_data_stats(mnistm_train_data, train_label, mnistm_test_data, test_label)

    train_data = [(x, y) for x, y in zip(mnistm_train_data, train_label)]
    test_data = [(x, y) for x, y in zip(mnistm_test_data, test_label)]

    return train_data, test_data

def get_target_usps():
    print("-- loading usps data")
    
    dataset  = sio.loadmat(DATA_FOLDER + '/usps_28x28.mat')
    data_set = dataset['dataset']
    img_train = data_set[0][0]
    label_train = data_set[0][1]
    img_test = data_set[1][0]
    label_test = data_set[1][1]
    inds = np.random.permutation(img_train.shape[0])
    img_train = img_train[inds]
    label_train = label_train[inds]

    frac = 0.99/255
    img_train *= 255
    img_train = img_train * frac + 0.01

    img_test *= 255
    img_test = img_test * frac + 0.01

    label_train = label_train.reshape(-1)
    label_test = label_test.reshape(-1)

    img_train = np.concatenate([ img_train, img_train, img_train], 1)
    img_test = np.concatenate([img_test, img_test,img_test],1)
    '''
    print('usps train X shape->',  img_train.shape)
    print('usps train y shape->',  label_train.shape)
    print('usps test X shape->',  img_test.shape)
    print('usps test y shape->', label_test.shape)
    '''

    print_image_data_stats(img_train, label_train, img_test, label_test)

    train_data = [(x, y) for x, y in zip(img_train, label_train)]
    test_data = [(x, y) for x, y in zip(img_test, label_test)]
    return train_data, test_data

### Generate Clients' Shards
def clients_rand(train_len, nclients):
  '''
  train_len: size of the train data
  nclients: number of clients
  
  Returns: to_ret
  
  This function creates a random distribution 
  for the clients, i.e. number of images each client 
  possess.
  '''
  client_tmp=[]
  sum_=0
  #### creating random values for each client ####
  for i in range(nclients-1):
    tmp=random.randint(10,100)
    sum_+=tmp
    client_tmp.append(tmp)

  client_tmp= np.array(client_tmp)
  #### using those random values as weights ####
  clients_dist= ((client_tmp/sum_)*train_len).astype(int)
  num  = train_len - clients_dist.sum()
  to_ret = list(clients_dist)
  to_ret.append(num)
  return to_ret
  
### Non-IID Data Split
def split_image_data_complete(data_train, data_test, train_labels, test_labels, 
                                n_clients=100, classes_per_client=10, shuffle=True, verbose=True):
    '''
    Splits (data, labels) among 'n_clients s.t. every client can holds 'classes_per_client' number of classes
    Input:
        data : [n_data x shape]
        labels : [n_data (x 1)] from 0 to n_labels
        n_clients : number of clients
        classes_per_client : number of classes per client
        shuffle : True/False => True for shuffling the dataset, False otherwise
        verbose : True/False => True for printing some info, False otherwise
    Output:
        clients_split : client data into desired format
    '''
    #### constants #### 
    n_data_train = data_train.shape[0]
    n_data_test = data_test.shape[0]

    n_labels_train = np.max(train_labels) + 1
    n_labels_test = np.max(test_labels) + 1


    ### client distribution ####
    train_data_per_client = clients_rand(len(data_train), n_clients)
    train_data_per_client_per_class = [np.maximum(1,nd // classes_per_client) for nd in train_data_per_client]

    test_data_per_client = clients_rand(len(data_test), n_clients)
    test_data_per_client_per_class = [np.maximum(1,nd // classes_per_client) for nd in test_data_per_client]

    # sort for train labels
    train_data_idcs = [[] for i in range(n_labels_train)]
    for j, label in enumerate(train_labels):
        train_data_idcs[label] += [j]
    if shuffle:
        for idcs in train_data_idcs:
            np.random.shuffle(idcs)

    # sort for test labels
    test_data_idcs = [[] for i in range(n_labels_test)]
    for j, label in enumerate(test_labels):
        test_data_idcs[label] += [j]
    if shuffle:
        for idcs in test_data_idcs:
            np.random.shuffle(idcs)

    # split train data among clients
    train_clients_split = []
    test_clients_split = []
    c = 0
    for i in range(n_clients):
        train_client_idcs = []
        test_client_idcs = []
            
        train_budget = train_data_per_client[i]
        test_budget = test_data_per_client[i]
        c_train = np.random.randint(n_labels_train)
        c_test = c_train

        while train_budget > 0:
            take_train = min(train_data_per_client_per_class[i], len(train_data_idcs[c_train]), train_budget)
            
            train_client_idcs += train_data_idcs[c_train][:take_train]
            train_data_idcs[c_train] = train_data_idcs[c_train][take_train:]
            
            train_budget -= take_train
            c_train = (c_train + 1) % n_labels_train

        while test_budget > 0:
            take_test = min(test_data_per_client_per_class[i], len(test_data_idcs[c_test]), test_budget)
            
            test_client_idcs += test_data_idcs[c_test][:take_test]
            test_data_idcs[c_test] = test_data_idcs[c_test][take_test:]
            
            test_budget -= take_test
            c_test = (c_test + 1) % n_labels_test

        train_clients_split += [(data_train[train_client_idcs], train_labels[train_client_idcs])]
        test_clients_split += [(data_test[test_client_idcs], test_labels[test_client_idcs])]

    def print_split(clients_split, n_labels): 
        print("Data split:")
        for i, client in enumerate(clients_split):
            split = np.sum(client[1].reshape(1,-1)==np.arange(n_labels).reshape(-1,1), axis=1)
            print(" --- Client {}: {}".format(i,split))
        print()
        
    if verbose:
        print("Train:")
        print_split(train_clients_split, n_labels_train)
        print("Test:")
        print_split(test_clients_split, n_labels_test)
    
    train_clients_split = np.array(train_clients_split)
    test_clients_split = np.array(test_clients_split)
    
    return train_clients_split, test_clients_split

### Shuffle Data
def shuffle_list(data):
    '''
    This function returns the shuffled data
    '''
    for i in range(len(data)):
        tmp_len= len(data[i][0])
        index = [i for i in range(tmp_len)]
        random.shuffle(index)
        data[i][0],data[i][1] = shuffle_list_data(data[i][0],data[i][1])
    return data

def shuffle_list_data(x, y):
    '''
    This function is a helper function, shuffles an
    array while maintaining the mapping between x and y
    '''
    inds = list(range(len(x)))
    random.shuffle(inds)
    return x[inds],y[inds]

### A Custom Dataset Class for Images
class CustomImageDataset(Dataset):
    '''
    A custom Dataset class for images
    inputs : numpy array [n_data x shape]
    labels : numpy array [n_data (x 1)]
    '''
    def __init__(self, inputs, labels, transforms=None):
        assert inputs.shape[0] == labels.shape[0]
        self.inputs = torch.Tensor(inputs)
        self.labels = torch.Tensor(labels).long()
        self.transforms = transforms 

    def __getitem__(self, index):
        img, label = self.inputs[index], self.labels[index]

        if self.transforms is not None:
            img = self.transforms(img)

        return (img, label)

    def __len__(self):
        return self.inputs.shape[0]

### Load, Split and Shuffle Data for Clients in Non-IID Settings
def get_data_loaders(dataset, nclients, batch_size, classes_pc = 2, verbose=True):
    '''
    This function is a DataLoader function, Load, Split 
    and Shuffle Data for Clients in a Non-IID Setting
    '''  
    if dataset == "mnist":
        x_train, y_train, x_test, y_test = get_source_mnist_pkl()
    elif dataset == "mnistm":
        x_train, y_train, x_test, y_test = get_source_mnistm()
    elif dataset == "usps":
        x_train, y_train, x_test, y_test = get_source_usps()
    elif dataset == "usps_mnist":
        x_mnist_train, y_mnist_train, x_mnist_test, y_mnist_test = get_source_mnist_pkl()
        x_usps_train, y_usps_train, x_usps_test, y_usps_test = get_source_usps()
        x_train = np.concatenate([x_mnist_train, x_usps_train], 0)
        y_train = np.concatenate([y_mnist_train, y_usps_train], 0)
        x_test =  np.concatenate([x_mnist_test, x_usps_test])
        y_test =  np.concatenate([y_mnist_test, y_usps_test])
        
    n_labels = np.max(y_train) + 1

    split_train, split_test = split_image_data_complete(x_train, x_test, y_train, y_test, n_clients=nclients, classes_per_client=classes_pc, verbose=verbose)
    
    split_tmp = shuffle_list(split_train)
    split_test_tmp = shuffle_list(split_test)

    array = []
    for i in range(len(split_tmp)):
        client_sample = len(split_tmp[i][0]) + len(split_test_tmp[i][1])
        array.append(client_sample)

    r1 = np.average(array)
    r2 = np.sqrt(np.mean((array - np.mean(array)) ** 2))
    r3 = np.mean((array - np.mean(array)) ** 2)

    if verbose:
        print_image_data_stats(x_train, y_train, x_test, y_test)
        #print(len(split_tmp[0][0]), len(split_tmp[0][1]))
        #plot_image_data(x_train, y_train)
        #plot_image_data(split_tmp[1][0], split_tmp[1][1])
        #plot_image_data(split_tmp[2][0], split_tmp[2][1])

        print(" -- Mean: ", r1) 
        print(" -- std: ", r2)
        print(" -- Variance: ", r3)
    
    # client_loaders = [torch.utils.data.DataLoader(CustomImageDataset(x, y), batch_size=batch_size, shuffle=True) for x, y in split_tmp]

    # test_loader  = torch.utils.data.DataLoader(CustomImageDataset(x_test, y_test), batch_size=100, shuffle=False) 

    # return client_loaders, test_loader
    return split_tmp, split_test_tmp, r1, r2, r3

def generate_data(dataset, nclients = 100, batch_size = 32, classes_pc = 2, verbose=True):

    outfile = "niid_" + dataset + ".mat"
    if os.path.exists(outfile):
        print(outfile, "already generated. Please remove the file before generating!")
        return False
    else:
        train_data, test_data, mean, std, variance = get_data_loaders(dataset, classes_pc = classes_pc, 
                                                                    nclients= nclients, batch_size=batch_size, verbose=verbose)

        sio.savemat('./' + outfile, {'train_data':train_data, 'test_data': test_data, "stat": [mean, std, variance]})
    
    return outfile
    
if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="mnist", choices=["mnist", "mnistm", "usps"], help="Select source dataset")
    parser.add_argument("--numuser", type=int, default=100, help="Total source domain users")
    parser.add_argument("--label", type=int, default=2, help="Number of classes per user")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--verbose", type=int, default=1, help="Print logs while generating data" )

    args = parser.parse_args()

    print("=" * 80)
    print("Summary of generating process:")
    print("Dataset          : {}".format(args.dataset))
    print("Number of Users  : {}".format(args.numuser))
    print("Labels per User  : {}".format(args.label))
    print("Batch Size       : {}".format(args.batch_size))
    print("Verbose          : {}".format(args.verbose))
    print("=" * 80)

    print("Generating Data:")
    outfile = generate_data(args.dataset, args.numuser, args.batch_size, args.label, (args.verbose == 1) and True or False)
    if outfile:
        print("Sucessfully Generate ", outfile)
        print("Finished!")
    else:
        print("Unsuccessfully Generating. Please try again!")
