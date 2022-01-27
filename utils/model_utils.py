import json
import numpy as np
import os
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data.dataset import Dataset
from tqdm import trange
import random
from scipy.io import loadmat
import sys
import pickle
# from sklearn.datasets import fetch_mldata
from sklearn.model_selection import train_test_split

## Ignore Warnings
import warnings
warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning) 

IMAGE_SIZE = 28
IMAGE_PIXELS = IMAGE_SIZE * IMAGE_SIZE
NUM_CHANNELS = 1

IMAGE_SIZE_CIFAR = 32
NUM_CHANNELS_CIFAR = 3

def suffer_data(data):
    data_x = data['x']
    data_y = data['y']
        # randomly shuffle data
    np.random.seed(100)
    rng_state = np.random.get_state()
    np.random.shuffle(data_x)
    np.random.set_state(rng_state)
    np.random.shuffle(data_y)
    return (data_x, data_y)
    
def batch_data(data, batch_size):
    '''
    data is a dict := {'x': [numpy array], 'y': [numpy array]} (on one client)
    returns x, y, which are both numpy array of length: batch_size
    '''
    data_x = data['x']
    data_y = data['y']

    # randomly shuffle data
    np.random.seed(100)
    rng_state = np.random.get_state()
    np.random.shuffle(data_x)
    np.random.set_state(rng_state)
    np.random.shuffle(data_y)

    # loop through mini-batches
    for i in range(0, len(data_x), batch_size):
        batched_x = data_x[i:i+batch_size]
        batched_y = data_y[i:i+batch_size]
        yield (batched_x, batched_y)

def get_random_batch_sample(data_x, data_y, batch_size):
    num_parts = len(data_x)//batch_size + 1
    if(len(data_x) > batch_size):
        batch_idx = np.random.choice(list(range(num_parts +1)))
        sample_index = batch_idx*batch_size
        if(sample_index + batch_size > len(data_x)):
            return (data_x[sample_index:], data_y[sample_index:])
        else:
            return (data_x[sample_index: sample_index+batch_size], data_y[sample_index: sample_index+batch_size])
    else:
        return (data_x,data_y)

def get_batch_sample(data, batch_size):
    data_x = data['x']
    data_y = data['y']

    np.random.seed(100)
    rng_state = np.random.get_state()
    np.random.shuffle(data_x)
    np.random.set_state(rng_state)
    np.random.shuffle(data_y)

    batched_x = data_x[0:batch_size]
    batched_y = data_y[0:batch_size]
    return (batched_x, batched_y)

def read_mnist_data():
    import scipy
    #mnist = loadmat(base_dir + '/Mnist/data/mldata/mnist.mat')
    mu = np.mean(mnist.data.astype(np.float32), 0)
    sigma = np.std(mnist.data.astype(np.float32), 0)
    mnist.data = (mnist.data.astype(np.float32) - mu)/(sigma+0.001)
    NUM_USERS = 1
    ###### CREATE USER DATA SPLIT #######
    # Assign 100 samples to each user
    X = [[] for _ in range(NUM_USERS)]
    y = [[] for _ in range(NUM_USERS)]
    idx = np.zeros(10, dtype=np.int64)
    for user in range(NUM_USERS):
            #l = (2*user+j)%10
        X[user] = mnist.data.tolist()
        y[user] =  mnist.target.tolist()

    # Create data structure
    train_data = {'users': [], 'user_data':{}, 'num_samples':[]}
    test_data = {'users': [], 'user_data':{}, 'num_samples':[]}

    # Setup 5 users
    # for i in trange(5, ncols=120):
    # Setup 5 users
    # for i in trange(5, ncols=120):
    for i in range(NUM_USERS):
        uname = i
        X_train, X_test, y_train, y_test = train_test_split(X[i], y[i], train_size=0.75, stratify=y[i])

        train_data["user_data"][uname] = {'x': X_train, 'y': y_train}
        train_data['users'].append(uname)
        train_data['num_samples'].append(len(y_train))
        
        test_data['users'].append(uname)
        test_data["user_data"][uname] = {'x': X_test, 'y': y_test}
        test_data['num_samples'].append(len(y_test))

    return train_data['users'], 0 , train_data['user_data'], test_data['user_data']

def read_cifa_data():
    # transform_train = transforms.Compose([transforms.Resize((32,32)),  #resises the image so it can be perfect for our model.
    #                                   transforms.RandomHorizontalFlip(), # FLips the image w.r.t horizontal axis
    #                                   transforms.RandomRotation(10),     #Rotates the image to a specified angel
    #                                   transforms.RandomAffine(0, shear=10, scale=(0.8,1.2)), #Performs actions like zooms, change shear angles.
    #                                   transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2), # Set the color params
    #                                   transforms.ToTensor(), # comvert the image to tensor so that it can work with torch
    #                                   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) #Normalize all the images
    #                            ])
 
 
    # transform = transforms.Compose([transforms.Resize((32,32)),
    #                             transforms.ToTensor(),
    #                             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    #                             ])

    transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,download=True, transform=transform)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False,download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=len(trainset.data),shuffle=False)
    testloader = torch.utils.data.DataLoader(testset, batch_size=len(testset.data),shuffle=False)

    for _, train_data in enumerate(trainloader,0):
        trainset.data, trainset.targets = train_data
    for _, train_data in enumerate(testloader,0):
        testset.data, testset.targets = train_data

    random.seed(1)
    np.random.seed(1)
    NUM_USERS = 20 # should be muitiple of 10
    NUM_LABELS = 3
    # Setup directory for train/test data
    train_path = './data/train/cifa_train_100.json'
    test_path = './data/test/cifa_test_100.json'
    dir_path = os.path.dirname(train_path)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    dir_path = os.path.dirname(test_path)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    cifa_data_image = []
    cifa_data_label = []

    cifa_data_image.extend(trainset.data.cpu().detach().numpy())
    cifa_data_image.extend(testset.data.cpu().detach().numpy())
    cifa_data_label.extend(trainset.targets.cpu().detach().numpy())
    cifa_data_label.extend(testset.targets.cpu().detach().numpy())
    cifa_data_image = np.array(cifa_data_image)
    cifa_data_label = np.array(cifa_data_label)

    cifa_data = []
    for i in trange(10):
        idx = cifa_data_label==i
        cifa_data.append(cifa_data_image[idx])


    print("\nNumb samples of each label:\n", [len(v) for v in cifa_data])
    
    ###### CREATE USER DATA SPLIT #######
    # Assign 100 samples to each user
    X = [[] for _ in range(NUM_USERS)]
    y = [[] for _ in range(NUM_USERS)]
    idx = np.zeros(10, dtype=np.int64)
    for user in range(NUM_USERS):
        for j in range(NUM_LABELS + 1):
            #l = (2*user+j)%10
            l = (user + j) % 10
            print("L:", l)
            X[user] += cifa_data[l][idx[l]:idx[l]+10].tolist()
            y[user] += (l*np.ones(10)).tolist()
            idx[l] += 10

    # Assign remaining sample by power law
    user = 0
    props = np.random.lognormal(
        0, 2., (10, NUM_USERS, 5))  # last 5 is 5 labels
    props = np.array([[[len(v)-NUM_USERS]] for v in cifa_data]) * \
        props/np.sum(props, (1, 2), keepdims=True)
    for user in trange(NUM_USERS):
        for j in range(5):
            l = (user + j) % 10
            num_samples = int(props[l, user//int(NUM_USERS/10), j])
            numran1 = random.randint(100, 200)
            num_samples = (num_samples)  + numran1 #+ 200
            if(NUM_USERS <= 20): 
                num_samples = num_samples * 2
            if idx[l] + num_samples < len(cifa_data[l]):
                X[user] += cifa_data[l][idx[l]:idx[l]+num_samples].tolist()
                y[user] += (l*np.ones(num_samples)).tolist()
                idx[l] += num_samples

    # Create data structure
    train_data = {'users': [], 'user_data':{}, 'num_samples':[]}
    test_data = {'users': [], 'user_data':{}, 'num_samples':[]}

    # Setup 5 users
    # for i in trange(5, ncols=120):
    data_all = []
    for i in range(NUM_USERS):
        uname = i
        combined = list(zip(X[i], y[i]))
        random.shuffle(combined)
        X[i][:], y[i][:] = zip(*combined)

        num_samples = len(X[i])
        print(num_samples)
        train_len = int(0.75*num_samples)
        test_len = num_samples - train_len

        #X_train, X_test, y_train, y_test = train_test_split(X[i], y[i], train_size=0.75, stratify=y[i])\
        X_train, y_train, X_test, y_test = X[i][test_len:], y[i][test_len:],X[i][:test_len],y[i][:test_len]
        X_train = torch.Tensor(X_train).view(-1, NUM_CHANNELS_CIFAR, IMAGE_SIZE_CIFAR, IMAGE_SIZE_CIFAR).type(torch.float32)
        y_train = torch.Tensor(y_train).type(torch.int64)
        X_test = torch.Tensor(X_test).view(-1, NUM_CHANNELS_CIFAR, IMAGE_SIZE_CIFAR, IMAGE_SIZE_CIFAR).type(torch.float32)
        y_test = torch.Tensor(y_test).type(torch.int64)
        train_data_res = [(x, y) for x, y in zip(X_train, y_train)]
        test_data_res = [(x, y) for x, y in zip(X_test, y_test)]
        data_all.append([train_data_res,test_data_res])
    print("Finish generate Cifar10")
    return data_all

def read_data(dataset):
    '''parses data in given train and test data directories

    assumes:
    - the data in the input directories are .json files with 
        keys 'users' and 'user_data'
    - the set of train set users is the same as the set of test set users

    Return:
        clients: list of client ids
        groups: list of group ids; empty list if none found
        train_data: dictionary of train data
        test_data: dictionary of test data
    '''

    if(dataset == "Cifar10"):
        return read_cifa_data()

    train_data_dir = os.path.join('data',dataset,'data', 'train')
    test_data_dir = os.path.join('data',dataset,'data', 'test')
    clients = []
    groups = []
    train_data = {}
    test_data = {}

    train_files = os.listdir(train_data_dir)
    train_files = [f for f in train_files if f.endswith('.json')]
    for f in train_files:
        file_path = os.path.join(train_data_dir, f)
        with open(file_path, 'r') as inf:
            cdata = json.load(inf)
        clients.extend(cdata['users'])
        if 'hierarchies' in cdata:
            groups.extend(cdata['hierarchies'])
        train_data.update(cdata['user_data'])

    test_files = os.listdir(test_data_dir)
    test_files = [f for f in test_files if f.endswith('.json')]
    for f in test_files:
        file_path = os.path.join(test_data_dir, f)
        with open(file_path, 'r') as inf:
            cdata = json.load(inf)
        test_data.update(cdata['user_data'])

    clients = list(sorted(train_data.keys()))

    return clients, groups, train_data, test_data

def read_user_data(index,data,dataset):
    id = data[0][index]
    train_data = data[2][id]
    test_data = data[3][id]
    X_train, y_train, X_test, y_test = train_data['x'], train_data['y'], test_data['x'], test_data['y']
    if(dataset == "Mnist"):
        X_train, y_train, X_test, y_test = train_data['x'], train_data['y'], test_data['x'], test_data['y']
        X_train = torch.Tensor(X_train).view(-1, NUM_CHANNELS, IMAGE_SIZE, IMAGE_SIZE).type(torch.float32)
        y_train = torch.Tensor(y_train).type(torch.int64)
        X_test = torch.Tensor(X_test).view(-1, NUM_CHANNELS, IMAGE_SIZE, IMAGE_SIZE).type(torch.float32)
        y_test = torch.Tensor(y_test).type(torch.int64)
    elif(dataset == "Cifar10"):
        X_train, y_train, X_test, y_test = train_data['x'], train_data['y'], test_data['x'], test_data['y']
        X_train = torch.Tensor(X_train).view(-1, NUM_CHANNELS_CIFAR, IMAGE_SIZE_CIFAR, IMAGE_SIZE_CIFAR).type(torch.float32)
        y_train = torch.Tensor(y_train).type(torch.int64)
        X_test = torch.Tensor(X_test).view(-1, NUM_CHANNELS_CIFAR, IMAGE_SIZE_CIFAR, IMAGE_SIZE_CIFAR).type(torch.float32)
        y_test = torch.Tensor(y_test).type(torch.int64)
    else:
        X_train = torch.Tensor(X_train).type(torch.float32)
        y_train = torch.Tensor(y_train).type(torch.int64)
        X_test = torch.Tensor(X_test).type(torch.float32)
        y_test = torch.Tensor(y_test).type(torch.int64)
    
    train_data = [(x, y) for x, y in zip(X_train, y_train)]
    test_data = [(x, y) for x, y in zip(X_test, y_test)]
    return id, train_data, test_data

base_dir = './data'

def dataset_read(domain_name):
    if domain_name == 'svhn':
        train, test = load_svhn_2(0,0)
    elif domain_name == 'mnist':
        train, test = load_mnist_2(0,0)
    elif domain_name == 'usps':
        train, test = load_usps_2()
    elif domain_name == 'mnistm':
        train, test = load_mnistm_2(0,0)
    elif domain_name == 'synth':
        train, test = load_syntraffic()
    elif domain_name == 'gtsrb':
        train, test = load_gtsrb()
    elif domain_name == 'syn':
        train, test = load_syn()
    elif domain_name == 'amazon':
        train, test = load_amazon()
    elif domain_name == 'dslr':
        train, test = load_dslr()
    elif domain_name == 'webcam':
        train, test = load_webcam()
    elif domain_name == 'Caltech10':
        train, test = load_caltech()
    else:
        return 
    return train, test

def load_amazon():
    #amazon_train = loadmat(base_dir + '/Office_Caltech10/amazon_SURF_L10.mat')
    #amazon_test = loadmat(base_dir + '/Office_Caltech10/amazon_decaf.mat')
    amazon_train = loadmat(base_dir + '/Office_Caltech10/caltech_decaf.mat')
    amazon_test = loadmat(base_dir + '/Office_Caltech10/Caltech10_SURF_L10.mat')

    amazon_train_im = amazon_train['X']
    amazon_train_im = np.array(amazon_train_im.transpose(3, 2, 0, 1).astype(np.float32), dtype=np.float32).reshape(-1, 2352)
    amazon_label = dense_to_one_hot(amazon_train['y'])

    amazon_test_im = amazon_test['X']
    amazon_test_im = np.array(amazon_test_im.transpose(3, 2, 0, 1).astype(np.float32), dtype=np.float32).reshape(-1, 2352)
    amazon_label_test = dense_to_one_hot(amazon_test['y'])

    amazon_all =     np.concatenate((amazon_train_im, amazon_test_im), axis=0)
    mu = np.mean(amazon_all, 0)
    sigma = np.std(amazon_all, 0)
    amazon_train_im = (amazon_train_im.astype(np.float32) - mu)/(sigma+0.001)
    amazon_test_im = (amazon_test_im.astype(np.float32) - mu)/(sigma+0.001)
    amazon_train_im = amazon_train_im.reshape(-1, 3, 28, 28).astype(np.float32)
    amazon_test_im = amazon_test_im.reshape(-1, 3, 28, 28).astype(np.float32)

    amazon_train_im = amazon_train_im[:25000]
    amazon_label = amazon_label[:25000]
    amazon_test_im = amazon_test_im[:9000]
    amazon_label_test = amazon_label_test[:9000]
    print('amazon train X shape->',  amazon_train_im.shape)
    print('amazon train y shape->',  amazon_label.shape)
    print('amazon test X shape->',  amazon_test_im.shape)
    print('amazon test y shape->', amazon_label_test.shape)

    train_data = [(x, y) for x, y in zip(amazon_train_im, amazon_label)]
    test_data = [(x, y) for x, y in zip(amazon_test_im, amazon_label_test)]

    return train_data, test_data

def load_svhn():
    svhn_train = loadmat(base_dir + '/msda/svhn_train_28x28.mat')
    svhn_test = loadmat(base_dir + '/msda/svhn_test_28x28.mat')

    svhn_train_im = svhn_train['X']
    svhn_train_im = np.array(svhn_train_im.transpose(3, 2, 0, 1).astype(np.float32), dtype=np.float32).reshape(-1, 2352)
    svhn_label = dense_to_one_hot(svhn_train['y'])

    svhn_test_im = svhn_test['X']
    svhn_test_im = np.array(svhn_test_im.transpose(3, 2, 0, 1).astype(np.float32), dtype=np.float32).reshape(-1, 2352)
    svhn_label_test = dense_to_one_hot(svhn_test['y'])

    svhn_all =     np.concatenate((svhn_train_im, svhn_test_im), axis=0)
    mu = np.mean(svhn_all, 0)
    sigma = np.std(svhn_all, 0)
    svhn_train_im = (svhn_train_im.astype(np.float32) - mu)/(sigma+0.001)
    svhn_test_im = (svhn_test_im.astype(np.float32) - mu)/(sigma+0.001)
    svhn_train_im = svhn_train_im.reshape(-1, 3, 28, 28).astype(np.float32)
    svhn_test_im = svhn_test_im.reshape(-1, 3, 28, 28).astype(np.float32)

    svhn_train_im = svhn_train_im[:25000]
    svhn_label = svhn_label[:25000]
    svhn_test_im = svhn_test_im[:9000]
    svhn_label_test = svhn_label_test[:9000]
    print('svhn train X shape->',  svhn_train_im.shape)
    print('svhn train y shape->',  svhn_label.shape)
    print('svhn test X shape->',  svhn_test_im.shape)
    print('svhn test y shape->', svhn_label_test.shape)

    train_data = [(x, y) for x, y in zip(svhn_train_im, svhn_label)]
    test_data = [(x, y) for x, y in zip(svhn_test_im, svhn_label_test)]

    return train_data, test_data

def load_mnist(scale=True, usps=False, all_use=False):
    mnist_data = loadmat(base_dir + '/msda/mnist_data.mat')
    mnist_train = np.array(mnist_data['train_28'], dtype=np.float32).reshape(-1, 784)
    mnist_test =  np.array(mnist_data['test_28'], dtype=np.float32).reshape(-1, 784)
    mnist_all =     np.concatenate((mnist_train, mnist_test), axis=0)

    # normalized data
    mu = np.mean(mnist_all, 0)
    sigma = np.std(mnist_all, 0)
    mnist_train = (mnist_train.astype(np.float32) - mu)/(sigma+0.001)
    mnist_test = (mnist_test.astype(np.float32) - mu)/(sigma+0.001)

    mnist_labels_train = mnist_data['label_train']
    mnist_labels_test = mnist_data['label_test']


    mnist_train = mnist_train.reshape(-1, 1, 28, 28).astype(np.float32)
    mnist_test = mnist_test.reshape(-1, 1, 28, 28).astype(np.float32)

    mnist_train = np.concatenate([mnist_train, mnist_train, mnist_train], 1)
    mnist_test = np.concatenate([mnist_test, mnist_test, mnist_test], 1)
    
    train_label = np.argmax(mnist_labels_train, axis=1)
    inds = np.random.permutation(mnist_train.shape[0])
    mnist_train = mnist_train[inds]
    train_label = train_label[inds]
    test_label = np.argmax(mnist_labels_test, axis=1)
    
    mnist_train = mnist_train[:25000]
    train_label = train_label[:25000]
    mnist_test = mnist_test[:9000]
    test_label = test_label[:9000]
    # print('sssss')
    print('mnist train X shape->',  mnist_train.shape)
    print('mnist train y shape->',  train_label.shape)
    print('mnist test X shape->',  mnist_test.shape)
    print('mnist test y shape->', test_label.shape)

    train_data = [(x, y) for x, y in zip(mnist_train, train_label)]
    test_data = [(x, y) for x, y in zip(mnist_test, test_label)]
    return train_data, test_data

def load_usps():
    dataset  = loadmat(base_dir + '/msda/usps_28x28.mat')
    data_set = dataset['dataset']
    img_train = data_set[0][0]
    label_train = data_set[0][1]
    img_test = data_set[1][0]
    label_test = data_set[1][1]
    inds = np.random.permutation(img_train.shape[0])
    img_train = img_train[inds]
    label_train = label_train[inds]
    
    #img_train = img_train * 255
    #img_test = img_test * 255

    #img_train = img_train.reshape((img_train.shape[0], 1, 28, 28))
    #img_test = img_test.reshape((img_test.shape[0], 1, 28, 28))

    label_train = dense_to_one_hot(label_train)
    label_test = dense_to_one_hot(label_test)
    img_train = np.concatenate([ img_train, img_train, img_train], 1)
    img_test = np.concatenate([img_test, img_test,img_test],1)
    
    print('usps train X shape->',  img_train.shape)
    print('usps train y shape->',  label_train.shape)
    print('usps test X shape->',  img_test.shape)
    print('usps test y shape->', label_test.shape)
    train_data = [(x, y) for x, y in zip(img_train, label_train)]
    test_data = [(x, y) for x, y in zip(img_test, label_test)]
    return train_data, test_data

def load_mnistm(scale=True, usps=False, all_use=False):
    mnistm_data = loadmat(base_dir + '/msda/mnistm_with_label.mat')
    mnistm_train =  np.array(mnistm_data['train'], dtype=np.float32).reshape(-1, 2352)
    mnistm_test =  np.array(mnistm_data['test'], dtype=np.float32).reshape(-1, 2352)
    mnistm_all =     np.concatenate((mnistm_train, mnistm_test), axis=0)

    #normalized data
    #mnistm_train = mnistm_train.transpose(0, 3, 1, 2).astype(np.float32)
    #mnistm_test = mnistm_test.transpose(0, 3, 1, 2).astype(np.float32)

    mu = np.mean(mnistm_all, 0)
    sigma = np.std(mnistm_all, 0)
    mnistm_train = (mnistm_train.astype(np.float32) - mu)/(sigma+0.001)
    mnistm_test = (mnistm_test.astype(np.float32) - mu)/(sigma+0.001)

    mnistm_train = mnistm_train.reshape(-1, 3, 28, 28).astype(np.float32)
    mnistm_test = mnistm_test.reshape(-1, 3, 28, 28).astype(np.float32)

    mnistm_labels_train = mnistm_data['label_train']
    mnistm_labels_test = mnistm_data['label_test']

    train_label = np.argmax(mnistm_labels_train, axis=1)
    inds = np.random.permutation(mnistm_train.shape[0])
    mnistm_train = mnistm_train[inds]
    train_label = train_label[inds]
    test_label = np.argmax(mnistm_labels_test, axis=1)
    
    mnistm_train = mnistm_train[:25000]
    train_label = train_label[:25000]
    mnistm_test = mnistm_test[:9000]
    test_label = test_label[:9000]
    print('mnist_m train X shape->',  mnistm_train.shape)
    print('mnist_m train y shape->',  train_label.shape)
    print('mnist_m test X shape->',  mnistm_test.shape)
    print('mnist_m test y shape->', test_label.shape)
    train_data = [(x, y) for x, y in zip(mnistm_train, train_label)]
    test_data = [(x, y) for x, y in zip(mnistm_test, test_label)]
    return train_data, test_data

def load_syn(scale=True, usps=False, all_use=False):
    syn_train = loadmat(base_dir + '/msda/synth_train_28x28.mat')
    syn_test = loadmat(base_dir + '/msda/synth_test_28x28.mat')

    syn_train_im = syn_train['X']
    syn_train_im = np.array(syn_train_im.transpose(3, 2, 0, 1).astype(np.float32), dtype=np.float32).reshape(-1, 2352)
    train_label = dense_to_one_hot(syn_train['y'])
    
    syn_test_im = syn_test['X']
    syn_test_im = np.array(syn_test_im.transpose(3, 2, 0, 1).astype(np.float32), dtype=np.float32).reshape(-1, 2352)
    test_label = dense_to_one_hot(syn_test['y'])

    syn_all =     np.concatenate((syn_train_im, syn_test_im), axis=0)
    mu = np.mean(syn_all, 0)
    sigma = np.std(syn_all, 0)
    syn_train_im = (syn_train_im.astype(np.float32) - mu)/(sigma+0.001)
    syn_test_im = (syn_test_im.astype(np.float32) - mu)/(sigma+0.001)
    
    syn_train_im = syn_train_im.reshape(-1, 3, 28, 28).astype(np.float32)
    syn_test_im = syn_test_im.reshape(-1, 3, 28, 28).astype(np.float32)

    syn_train_im = syn_train_im[:25000]
    train_label = train_label[:25000]
    syn_test_im = syn_test_im[:9000]
    test_label = test_label[:9000]

    print('syn number train X shape->',  syn_train_im.shape)
    print('syn number train y shape->',  train_label.shape)
    print('syn number test X shape->',  syn_test_im.shape)
    print('syn number test y shape->', test_label.shape)

    train_data = [(x, y) for x, y in zip(syn_train_im, train_label)]
    test_data = [(x, y) for x, y in zip(syn_test_im, test_label)]
    return train_data, test_data

def load_syntraffic():
    data_source = pkl.load(open('../data/data_synthetic'))
    source_train = np.random.permutation(len(data_source['image']))
    data_s_im = data_source['image'][source_train[:len(data_source['image'])], :, :, :]
    data_s_im_test = data_source['image'][source_train[len(data_source['image']) - 2000:], :, :, :]
    data_s_label = data_source['label'][source_train[:len(data_source['image'])]]
    data_s_label_test = data_source['label'][source_train[len(data_source['image']) - 2000:]]
    data_s_im = data_s_im.transpose(0, 3, 1, 2).astype(np.float32)
    data_s_im_test = data_s_im_test.transpose(0, 3, 1, 2).astype(np.float32)
    train_data = [(x, y) for x, y in zip(data_s_im, data_s_label)]
    test_data = [(x, y) for x, y in zip(data_s_im_test, data_s_label_test)]
    return train_data, test_data

def load_gtsrb():
    data_target = pkl.load(open('../data/data_gtsrb'))
    target_train = np.random.permutation(len(data_target['image']))
    data_t_im = data_target['image'][target_train[:31367], :, :, :]
    data_t_im_test = data_target['image'][target_train[31367:], :, :, :]
    data_t_label = data_target['label'][target_train[:31367]] + 1
    data_t_label_test = data_target['label'][target_train[31367:]] + 1
    data_t_im = data_t_im.transpose(0, 3, 1, 2).astype(np.float32)
    data_t_im_test = data_t_im_test.transpose(0, 3, 1, 2).astype(np.float32)
    train_data = [(x, y) for x, y in zip(data_t_im, data_t_label)]
    test_data = [(x, y) for x, y in zip(data_t_im_test, data_t_label_test)]
    return train_data, test_data


def dense_to_one_hot(labels_dense):
    """Convert class labels from scalars to one-hot vectors."""
    labels_one_hot = np.zeros((len(labels_dense),))
    labels_dense = list(labels_dense)
    for i, t in enumerate(labels_dense):
        if t == 10:
            t = 0
            labels_one_hot[i] = t
        else:
            labels_one_hot[i] = t
    return labels_one_hot

##### 
def load_mnist_2(num_train, num_test):

    print("-- loading mnist data")

    with open(base_dir + '/msda/mnist.pkl', "br") as fh:
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

def load_mnistm_2(num_train, num_test):
    print("-- loading mnist-m data")

    mnistm_train = loadmat(base_dir + '/msda/mnistm_train.mat')
    mnistm_test = loadmat(base_dir + '/msda/mnistm_test.mat')

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


def load_usps_2():
    print("-- loading usps data")
    
    dataset  = loadmat(base_dir + '/msda/usps_28x28.mat')
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

def load_svhn_2(num_train, num_test):

    print("-- loading svhn data")

    svhn_train = loadmat(base_dir + '/msda/svhn_train_28x28.mat')
    svhn_test = loadmat(base_dir + '/msda/svhn_test_28x28.mat')

    svhn_train_im = svhn_train['X']
    svhn_train_im = np.array(svhn_train_im.transpose(3, 2, 0, 1).astype(np.float32), dtype=np.float32).reshape(-1, 2352)
    svhn_label = dense_to_one_hot(svhn_train['y'])

    svhn_test_im = svhn_test['X']
    svhn_test_im = np.array(svhn_test_im.transpose(3, 2, 0, 1).astype(np.float32), dtype=np.float32).reshape(-1, 2352)
    svhn_label_test = dense_to_one_hot(svhn_test['y'])

    svhn_all =     np.concatenate((svhn_train_im, svhn_test_im), axis=0)
    mu = np.mean(svhn_all, 0)
    sigma = np.std(svhn_all, 0)
    svhn_train_im = (svhn_train_im.astype(np.float32) - mu)/(sigma+0.001)
    svhn_test_im = (svhn_test_im.astype(np.float32) - mu)/(sigma+0.001)
    svhn_train_im = svhn_train_im.reshape(-1, 3, 28, 28).astype(np.float32)
    svhn_test_im = svhn_test_im.reshape(-1, 3, 28, 28).astype(np.float32)

    if (num_train != 0 and num_test != 0):
        svhn_train_im = svhn_train_im[:num_train]
        svhn_label = svhn_label[:num_train]
        svhn_test_im = svhn_test_im[:num_test]
        svhn_label_test = svhn_label_test[:num_test]

    print_image_data_stats(svhn_train_im, svhn_label, svhn_test_im, svhn_label_test)

    train_data = [(x, y) for x, y in zip(svhn_train_im, svhn_label)]
    test_data = [(x, y) for x, y in zip(svhn_test_im, svhn_label_test)]

    return train_data, test_data

######


#########################################
####### Domain Adaptation Utils #########
#########################################

### Base Folder
DATA_FOLDER = './data/fiveDigit/original_data'
BASE_FOLDER = './data/fiveDigit'

### Data Statistics
def print_image_data_stats(data_train, labels_train, data_test, labels_test):
    print("\nData: ")
    print(" -- Train Set: ({},{}), Range: [{:.3f}, {:.3f}], Labels: {},..,{}".format(
        data_train.shape, labels_train.shape, np.min(data_train), np.max(data_train),
        np.min(labels_train), np.max(labels_train)))
    print(" -- Test Set: ({},{}), Range: [{:.3f}, {:.3f}], Labels: {},..,{}".format(
        data_test.shape, labels_test.shape, np.min(data_train), np.max(data_train),
        np.min(labels_test), np.max(labels_test)))
  
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

    mnistm_train = loadmat(DATA_FOLDER + '/mnistm_train.mat')
    mnistm_test = loadmat(DATA_FOLDER + '/mnistm_test.mat')

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
    
    dataset  = loadmat(DATA_FOLDER + '/usps_28x28.mat')
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

def read_domain_data(dataset):
    '''
    Read Domain Datasets for running experiments
    '''

    print("- dataset to read :", dataset)

    data_all = []

    if(dataset == "mnist2mnistm"):

        # mnist_train, mnist_test = get_data_loaders(dataset="mnist", classes_pc = classes_pc, nclients= nclients-1, batch_size=32, verbose=True)        
        mnist_data = loadmat(BASE_FOLDER + "/niid_mnist.mat")
        mnist_train = mnist_data["train_data"]
        mnist_test = mnist_data["test_data"]
        mnistm_train, mnistm_test = get_target_mnistm_full(0,0)       
 
        clients = len(mnist_train)

        for i in range(clients):
            client_train = [(x, y) for x, y in zip(mnist_train[i][0], mnist_train[i][1][0])] 
            client_test = [(x, y) for x, y in zip(mnist_test[i][0], mnist_test[i][1][0])] 
            data_all.append([client_train, client_test])
        
        data_all.append([mnistm_train, mnistm_test])

        return data_all

    elif(dataset == "mnist2usps"):

        # mnist_train, mnist_test = get_data_loaders(dataset="mnist", classes_pc = classes_pc, nclients= nclients-1, batch_size=32, verbose=True)
        mnist_data = loadmat(BASE_FOLDER + "/niid_mnist.mat")
        mnist_train = mnist_data["train_data"]
        mnist_test = mnist_data["test_data"]

        usps_train, usps_test = get_target_usps()

        clients = len(mnist_train)

        for i in range(clients):
            client_train = [(x, y) for x, y in zip(mnist_train[i][0], mnist_train[i][1][0])] 
            client_test = [(x, y) for x, y in zip(mnist_test[i][0], mnist_test[i][1][0])] 
            data_all.append([client_train, client_test])
        
        data_all.append([usps_train, usps_test])

        return data_all

    elif(dataset == "mnistm2usps"):

        # mnistm_train, mnistm_test = get_data_loaders(dataset="mnistm", classes_pc = classes_pc, nclients= nclients-1, batch_size=32, verbose=True)
        mnistm_data = loadmat(BASE_FOLDER + "/niid_mnistm.mat")
        mnistm_train = mnistm_data["train_data"]
        mnistm_test = mnistm_data["test_data"]

        usps_train, usps_test = get_target_usps()

        clients = len(mnistm_train)

        for i in range(clients):
            client_train = [(x, y) for x, y in zip(mnistm_train[i][0], mnistm_train[i][1][0])] 
            client_test = [(x, y) for x, y in zip(mnistm_test[i][0], mnistm_test[i][1][0])] 
            data_all.append([client_train, client_test])
        
        data_all.append([usps_train, usps_test])

        return data_all

    elif(dataset == "mnistm2mnist"):
 
        # mnistm_train, mnistm_test = get_data_loaders(dataset="mnistm", classes_pc = classes_pc, nclients= nclients-1, batch_size=32, verbose=True)
        mnistm_data = loadmat(BASE_FOLDER + "/niid_mnistm.mat")
        mnistm_train = mnistm_data["train_data"]
        mnistm_test = mnistm_data["test_data"]

        mnist_train, mnist_test = get_target_mnist_pkl(0,0)

        print(mnistm_train[0][1].shape)
        clients = len(mnistm_train)

        for i in range(clients):
            client_train = [(x, y) for x, y in zip(mnistm_train[i][0], mnistm_train[i][1][0])] 
            client_test = [(x, y) for x, y in zip(mnistm_test[i][0], mnistm_test[i][1][0])] 
            data_all.append([client_train, client_test])
        
        data_all.append([mnist_train, mnist_test])

        return data_all

    elif(dataset == "usps2mnist"):
       
        # usps_train, usps_test = get_data_loaders(dataset="usps", classes_pc = classes_pc, nclients= nclients-1, batch_size=32, verbose=True)
        
        usps_data = loadmat(BASE_FOLDER + "/niid_usps.mat")
        usps_train = usps_data["train_data"]
        usps_test = usps_data["test_data"]

        mnist_train, mnist_test = get_target_mnist_pkl(0,0)

        clients = len(usps_train)

        for i in range(clients):
            client_train = [(x, y) for x, y in zip(usps_train[i][0], usps_train[i][1][0])] 
            client_test = [(x, y) for x, y in zip(usps_test[i][0], usps_test[i][1][0])] 
            data_all.append([client_train, client_test])
        
        data_all.append([mnist_train, mnist_test])

        return data_all

    elif (dataset == "usps2mnistm"):

        # usps_train, usps_test = get_data_loaders(dataset="usps", classes_pc = classes_pc, nclients= nclients-1, batch_size=32, verbose=True)
        usps_data = loadmat(BASE_FOLDER + "/niid_usps.mat")
        usps_train = usps_data["train_data"]
        usps_test = usps_data["test_data"]

        mnistm_train, mnistm_test = get_target_mnistm_full(0,0)

        clients = len(usps_train)

        for i in range(clients):
            client_train = [(x, y) for x, y in zip(usps_train[i][0], usps_train[i][1][0])] 
            client_test = [(x, y) for x, y in zip(usps_test[i][0], usps_test[i][1][0])] 
            data_all.append([client_train, client_test])
        
        data_all.append([mnistm_train, mnistm_test])

        return data_all
    
    elif(dataset == "msda1"):
        domain_all = ['mnist', 'svhn', 'usps']
        for domain_name in domain_all:
            data_all.append(dataset_read(domain_name))
        return data_all

    elif(dataset == "msda2"):
        domain_all = ['mnist', 'usps', 'svhn']
        for domain_name in domain_all:
            data_all.append(dataset_read(domain_name))
        return data_all

    elif(dataset == "msda3"):
        domain_all = ['svhn', 'usps', 'mnist']
        for domain_name in domain_all:
            data_all.append(dataset_read(domain_name))
        return data_all

    #########################
    ##### Original Code #####
    #########################
    elif(dataset == "fiveDigit"):
        domain_all = ['mnist', 'mnistm', 'svhn', 'syn', 'usps']
        for domain_name in domain_all:
            data_all.append(dataset_read(domain_name))
        return data_all

    elif(dataset == "Office_Caltech10"):
        domain_all = ['amazon', 'dslr', 'webcam', 'Caltech10']
        for domain_name in domain_all:
            data_all.append(dataset_read(domain_name))
        return data_all

    else:
        if(dataset == "Cifar10"):
            return read_cifa_data()

        #if(dataset == "Mnist"):
        #    clients, groups, train_data, test_data = read_mnist_data()
        #    return clients, groups, train_data, test_data
            
        train_data_dir = os.path.join('data',dataset,'data', 'train')
        test_data_dir = os.path.join('data',dataset,'data', 'test')
        clients = []
        groups = []
        train_data = {}
        test_data = {}

        train_files = os.listdir(train_data_dir)
        train_files = [f for f in train_files if f.endswith('.json')]
        for f in train_files:
            file_path = os.path.join(train_data_dir, f)
            with open(file_path, 'r') as inf:
                cdata = json.load(inf)
            clients.extend(cdata['users'])
            if 'hierarchies' in cdata:
                groups.extend(cdata['hierarchies'])
            train_data.update(cdata['user_data'])

        test_files = os.listdir(test_data_dir)
        test_files = [f for f in test_files if f.endswith('.json')]
        for f in test_files:
            file_path = os.path.join(test_data_dir, f)
            with open(file_path, 'r') as inf:
                cdata = json.load(inf)
            test_data.update(cdata['user_data'])

        clients = list(sorted(train_data.keys()))

        for id in clients:
            train_data_i = train_data[id]
            test_data_i = test_data[id]
            X_train, y_train, X_test, y_test = train_data_i['x'], train_data_i['y'], test_data_i['x'], test_data_i['y']
            if(dataset == "Mnist" or dataset == "Emnist"):
                X_train, y_train, X_test, y_test = train_data_i['x'], train_data_i['y'], test_data_i['x'], test_data_i['y']
                X_train = torch.Tensor(X_train).view(-1, NUM_CHANNELS, IMAGE_SIZE, IMAGE_SIZE).type(torch.float32)
                y_train = torch.Tensor(y_train).type(torch.int64)
                X_test = torch.Tensor(X_test).view(-1, NUM_CHANNELS, IMAGE_SIZE, IMAGE_SIZE).type(torch.float32)
                y_test = torch.Tensor(y_test).type(torch.int64)
            elif(dataset == "Cifar10"):
                X_train, y_train, X_test, y_test = train_data_i['x'], train_data_i['y'], test_data_i['x'], test_data_i['y']
                X_train = torch.Tensor(X_train).view(-1, NUM_CHANNELS_CIFAR, IMAGE_SIZE_CIFAR, IMAGE_SIZE_CIFAR).type(torch.float32)
                y_train = torch.Tensor(y_train).type(torch.int64)
                X_test = torch.Tensor(X_test).view(-1, NUM_CHANNELS_CIFAR, IMAGE_SIZE_CIFAR, IMAGE_SIZE_CIFAR).type(torch.float32)
                y_test = torch.Tensor(y_test).type(torch.int64)
            else:
                X_train = torch.Tensor(X_train).type(torch.float32)
                y_train = torch.Tensor(y_train).type(torch.int64)
                X_test = torch.Tensor(X_test).type(torch.float32)
                y_test = torch.Tensor(y_test).type(torch.int64)
            
            train_data_res = [(x, y) for x, y in zip(X_train, y_train)]
            test_data_res = [(x, y) for x, y in zip(X_test, y_test)]
            data_all.append([train_data_res,test_data_res])
        return data_all