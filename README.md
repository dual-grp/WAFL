# ON THE GENERALIZATION OF WASSERSTEIN ROBUST FEDERATED LEARNING
This repository implements all experiments in the paper the *ON THE GENERALIZATION OF WASSERSTEIN ROBUST FEDERATED LEARNING*.
  
Authors: 

# Software requirements:
- numpy, scipy, torch, Pillow, matplotlib, tqdm, pandas, h5py, comet_ml, scikit-learn==0.20.3
- To download the dependencies: **pip3 install -r requirements.txt**
- To download the dependencies for optimal transport dataset distance: **pip3 install -r otdd_requirements.txt**
- The code can be run on any pc, doesn't require GPU.
  
# Datasets:

- MNIST (100 clients): generated by runing <pre><code>python3 data/Mnist/generate_niid_100users.py</code></pre> 
- CIFAR (20 clients): This dataset will be downloaded and generated automatically when runing algorithms.
- Three-digit Datasets (MNIST, SVHN, USPS): A subset of Digit-Five Dataset

Download Link: https://drive.google.com/drive/folders/1ltpI-OejTE__FamAHJ7a2h6QHjgvOPfk?usp=sharing

All dataset after downloading must be stored at folder ./data


# Adversarial Training 
<pre><code>
    python3 main.py --dataset Mnist --model mclr --batch_size 64 --learning_rate 0.001 --robust 0 --gamma 0.05 --num_global_iters 200 --local_epochs 2 --algorithm WAFL --subusers 0.1 --numusers 100 --times 1
    python3 main.py --dataset Mnist --model mclr --batch_size 64 --learning_rate 0.001 --robust 0 --num_global_iters 200 --local_epochs 2 --algorithm FedAvg --subusers 0.1 --numusers 100 --times 1
    python3 main.py --dataset Mnist --model mclr --batch_size 64 --learning_rate 0.001 --robust 0 --num_global_iters 200 --local_epochs 2 --algorithm FedAFL --subusers 0.1 --numusers 100 --times 1
    python3 main.py --dataset Mnist --model mclr --batch_size 64 --learning_rate 0.001 --robust 0 --num_global_iters 200 --local_epochs 2 --algorithm FedDRFA --subusers 0.1 --numusers 100 --times 1
    python3 main.py --dataset Mnist --model mclr --batch_size 64 --learning_rate 0.001 --robust 0 --num_global_iters 200 --local_epochs 2 --algorithm FedPGD --subusers 0.1 --numusers 100 --times 1
    python3 main.py --dataset Mnist --model mclr --batch_size 64 --learning_rate 0.001 --robust 0 --num_global_iters 200 --local_epochs 2 --algorithm FedFGSM --subusers 0.1 --numusers 100 --times 1

    python3 main.py --dataset Mnist --model mclr --batch_size 64 --learning_rate 0.001 --robust 0.2 --gamma 0.05 --num_global_iters 200 --local_epochs 2 --algorithm WAFL --subusers 0.1 --numusers 100 --times 1
    python3 main.py --dataset Mnist --model mclr --batch_size 64 --learning_rate 0.001 --robust 0.2 --num_global_iters 200 --local_epochs 2 --algorithm FedAvg --subusers 0.1 --numusers 100 --times 1
    python3 main.py --dataset Mnist --model mclr --batch_size 64 --learning_rate 0.001 --robust 0.2 --num_global_iters 200 --local_epochs 2 --algorithm FedAFL --subusers 0.1 --numusers 100 --times 1
    python3 main.py --dataset Mnist --model mclr --batch_size 64 --learning_rate 0.001 --robust 0.2 --num_global_iters 200 --local_epochs 2 --algorithm FedDRFA --subusers 0.1 --numusers 100 --times 1
    python3 main.py --dataset Mnist --model mclr --batch_size 64 --learning_rate 0.001 --robust 0.2 --num_global_iters 200 --local_epochs 2 --algorithm FedPGD --subusers 0.1 --numusers 100 --times 1
    python3 main.py --dataset Mnist --model mclr --batch_size 64 --learning_rate 0.001 --robust 0.2 --num_global_iters 200 --local_epochs 2 --algorithm FedFGSM --subusers 0.1 --numusers 100 --times 1

    python3 main.py --dataset Mnist --model mclr --batch_size 64 --learning_rate 0.001 --robust 0.4 --gamma 0.05 --num_global_iters 200 --local_epochs 2 --algorithm WAFL --subusers 0.1 --numusers 100 --times 1
    python3 main.py --dataset Mnist --model mclr --batch_size 64 --learning_rate 0.001 --robust 0.4 --num_global_iters 200 --local_epochs 2 --algorithm FedAvg --subusers 0.1 --numusers 100 --times 1
    python3 main.py --dataset Mnist --model mclr --batch_size 64 --learning_rate 0.001 --robust 0.4 --num_global_iters 200 --local_epochs 2 --algorithm FedAFL --subusers 0.1 --numusers 100 --times 1
    python3 main.py --dataset Mnist --model mclr --batch_size 64 --learning_rate 0.001 --robust 0.4 --num_global_iters 200 --local_epochs 2 --algorithm FedDRFA --subusers 0.1 --numusers 100 --times 1
    python3 main.py --dataset Mnist --model mclr --batch_size 64 --learning_rate 0.001 --robust 0.4 --num_global_iters 200 --local_epochs 2 --algorithm FedPGD --subusers 0.1 --numusers 100 --times 1
    python3 main.py --dataset Mnist --model mclr --batch_size 64 --learning_rate 0.001 --robust 0.4 --num_global_iters 200 --local_epochs 2 --algorithm FedFGSM --subusers 0.1 --numusers 100 --times 1

    python3 main.py --dataset Mnist --model mclr --batch_size 64 --learning_rate 0.001 --robust 0.6 --gamma 0.05 --num_global_iters 200 --local_epochs 2 --algorithm WAFL --subusers 0.1 --numusers 100 --times 1
    python3 main.py --dataset Mnist --model mclr --batch_size 64 --learning_rate 0.001 --robust 0.6 --num_global_iters 200 --local_epochs 2 --algorithm FedAvg --subusers 0.1 --numusers 100 --times 1
    python3 main.py --dataset Mnist --model mclr --batch_size 64 --learning_rate 0.001 --robust 0.6 --num_global_iters 200 --local_epochs 2 --algorithm FedAFL --subusers 0.1 --numusers 100 --times 1
    python3 main.py --dataset Mnist --model mclr --batch_size 64 --learning_rate 0.001 --robust 0.6 --num_global_iters 200 --local_epochs 2 --algorithm FedDRFA --subusers 0.1 --numusers 100 --times 1
    python3 main.py --dataset Mnist --model mclr --batch_size 64 --learning_rate 0.001 --robust 0.6 --num_global_iters 200 --local_epochs 2 --algorithm FedPGD --subusers 0.1 --numusers 100 --times 1
    python3 main.py --dataset Mnist --model mclr --batch_size 64 --learning_rate 0.001 --robust 0.6 --num_global_iters 200 --local_epochs 2 --algorithm FedFGSM --subusers 0.1 --numusers 100 --times 1
    
    python3 main.py --dataset Mnist --model mclr --batch_size 64 --learning_rate 0.001 --robust 0.8 --gamma 0.05 --num_global_iters 200 --local_epochs 2 --algorithm WAFL --subusers 0.1 --numusers 100 --times 1
    python3 main.py --dataset Mnist --model mclr --batch_size 64 --learning_rate 0.001 --robust 0.8 --num_global_iters 200 --local_epochs 2 --algorithm FedAvg --subusers 0.1 --numusers 100 --times 1
    python3 main.py --dataset Mnist --model mclr --batch_size 64 --learning_rate 0.001 --robust 0.8 --num_global_iters 200 --local_epochs 2 --algorithm FedAFL --subusers 0.1 --numusers 100 --times 1
    python3 main.py --dataset Mnist --model mclr --batch_size 64 --learning_rate 0.001 --robust 0.8 --num_global_iters 200 --local_epochs 2 --algorithm FedDRFA --subusers 0.1 --numusers 100 --times 1
    python3 main.py --dataset Mnist --model mclr --batch_size 64 --learning_rate 0.001 --robust 0.8 --num_global_iters 200 --local_epochs 2 --algorithm FedPGD --subusers 0.1 --numusers 100 --times 1
    python3 main.py --dataset Mnist --model mclr --batch_size 64 --learning_rate 0.001 --robust 0.8 --num_global_iters 200 --local_epochs 2 --algorithm FedFGSM --subusers 0.1 --numusers 100 --times 1
</code></pre>

## Cifar10
<pre><code>
    python3 main.py --dataset Cifar10 --model cnn --batch_size 64 --learning_rate 0.05 --robust 0 --gamma 0.5 --alpha 1 --num_global_iters 200 --local_epochs 2 --algorithm WAFL --subusers 0.5 --numusers 20 --times 1
    python3 main.py --dataset Cifar10 --model cnn --batch_size 64 --learning_rate 0.05 --robust 0 --num_global_iters 200 --local_epochs 2 --algorithm FedAvg --subusers 0.5 --numusers 20 --times 1
    python3 main.py --dataset Cifar10 --model cnn --batch_size 64 --learning_rate 0.05 --robust 0 --num_global_iters 200 --local_epochs 2 --algorithm FedAFL --subusers 0.5 --numusers 20 --times 1
    python3 main.py --dataset Cifar10 --model cnn --batch_size 64 --learning_rate 0.05 --robust 0 --num_global_iters 200 --local_epochs 2 --algorithm FedDRFA --subusers 0.5 --numusers 20 --times 1
    python3 main.py --dataset Cifar10 --model cnn --batch_size 64 --learning_rate 0.05 --robust 0 --num_global_iters 200 --local_epochs 2 --algorithm FedPGD --subusers 0.5 --numusers 20 --times 1
    python3 main.py --dataset Cifar10 --model cnn --batch_size 64 --learning_rate 0.05 --robust 0 --num_global_iters 200 --local_epochs 2 --algorithm FedFGSM --subusers 0.5 --numusers 20 --times 1

    python3 main.py --dataset Cifar10 --model cnn --batch_size 64 --learning_rate 0.05 --robust 0.2 --gamma 0.5 --alpha 1 --num_global_iters 200 --local_epochs 2 --algorithm WAFL --subusers 0.5 --numusers 20 --times 1
    python3 main.py --dataset Cifar10 --model cnn --batch_size 64 --learning_rate 0.05 --robust 0.2 --num_global_iters 200 --local_epochs 2 --algorithm FedAvg --subusers 0.5 --numusers 20 --times 1
    python3 main.py --dataset Cifar10 --model cnn --batch_size 64 --learning_rate 0.05 --robust 0.2 --num_global_iters 200 --local_epochs 2 --algorithm FedAFL --subusers 0.5 --numusers 20 --times 1
    python3 main.py --dataset Cifar10 --model cnn --batch_size 64 --learning_rate 0.05 --robust 0.2 --num_global_iters 200 --local_epochs 2 --algorithm FedDRFA --subusers 0.5 --numusers 20 --times 1
    python3 main.py --dataset Cifar10 --model cnn --batch_size 64 --learning_rate 0.05 --robust 0.2 --num_global_iters 200 --local_epochs 2 --algorithm FedPGD --subusers 0.5 --numusers 20 --times 1
    python3 main.py --dataset Cifar10 --model cnn --batch_size 64 --learning_rate 0.05 --robust 0.2 --num_global_iters 200 --local_epochs 2 --algorithm FedFGSM --subusers 0.5 --numusers 20 --times 1

    python3 main.py --dataset Cifar10 --model cnn --batch_size 64 --learning_rate 0.05 --robust 0.4 --gamma 0.5 --alpha 1 --num_global_iters 200 --local_epochs 2 --algorithm WAFL --subusers 0.5 --numusers 20 --times 1
    python3 main.py --dataset Cifar10 --model cnn --batch_size 64 --learning_rate 0.05 --robust 0.4 --num_global_iters 200 --local_epochs 2 --algorithm FedAvg --subusers 0.5 --numusers 20 --times 1
    python3 main.py --dataset Cifar10 --model cnn --batch_size 64 --learning_rate 0.05 --robust 0.4 --num_global_iters 200 --local_epochs 2 --algorithm FedAFL --subusers 0.5 --numusers 20 --times 1
    python3 main.py --dataset Cifar10 --model cnn --batch_size 64 --learning_rate 0.05 --robust 0.4 --num_global_iters 200 --local_epochs 2 --algorithm FedDRFA --subusers 0.5 --numusers 20 --times 1
    python3 main.py --dataset Cifar10 --model cnn --batch_size 64 --learning_rate 0.05 --robust 0.4 --num_global_iters 200 --local_epochs 2 --algorithm FedPGD --subusers 0.5 --numusers 20 --times 1
    python3 main.py --dataset Cifar10 --model cnn --batch_size 64 --learning_rate 0.05 --robust 0.4 --num_global_iters 200 --local_epochs 2 --algorithm FedFGSM --subusers 0.5 --numusers 20 --times 1
    
    python3 main.py --dataset Cifar10 --model cnn --batch_size 64 --learning_rate 0.05 --robust 0.6 --gamma 0.5 --alpha 1 --num_global_iters 200 --local_epochs 2 --algorithm WAFL --subusers 0.5 --numusers 20 --times 1
    python3 main.py --dataset Cifar10 --model cnn --batch_size 64 --learning_rate 0.05 --robust 0.6 --num_global_iters 200 --local_epochs 2 --algorithm FedAvg --subusers 0.5 --numusers 20 --times 1
    python3 main.py --dataset Cifar10 --model cnn --batch_size 64 --learning_rate 0.05 --robust 0.6 --num_global_iters 200 --local_epochs 2 --algorithm FedAFL --subusers 0.5 --numusers 20 --times 1
    python3 main.py --dataset Cifar10 --model cnn --batch_size 64 --learning_rate 0.05 --robust 0.6 --num_global_iters 200 --local_epochs 2 --algorithm FedDRFA --subusers 0.5 --numusers 20 --times 1
    python3 main.py --dataset Cifar10 --model cnn --batch_size 64 --learning_rate 0.05 --robust 0.6 --num_global_iters 200 --local_epochs 2 --algorithm FedPGD --subusers 0.5 --numusers 20 --times 1
    python3 main.py --dataset Cifar10 --model cnn --batch_size 64 --learning_rate 0.05 --robust 0.6 --num_global_iters 200 --local_epochs 2 --algorithm FedFGSM --subusers 0.5 --numusers 20 --times 1

    python3 main.py --dataset Cifar10 --model cnn --batch_size 64 --learning_rate 0.05 --robust 0.8 --gamma 0.5 --alpha 1 --num_global_iters 200 --local_epochs 2 --algorithm WAFL --subusers 0.5 --numusers 200 --times 1
    python3 main.py --dataset Cifar10 --model cnn --batch_size 64 --learning_rate 0.05 --robust 0.8 --num_global_iters 200 --local_epochs 2 --algorithm FedAvg --subusers 0.5 --numusers 20 --times 1
    python3 main.py --dataset Cifar10 --model cnn --batch_size 64 --learning_rate 0.05 --robust 0.8 --num_global_iters 200 --local_epochs 2 --algorithm FedAFL --subusers 0.5 --numusers 20 --times 1
    python3 main.py --dataset Cifar10 --model cnn --batch_size 64 --learning_rate 0.05 --robust 0.8 --num_global_iters 200 --local_epochs 2 --algorithm FedDRFA --subusers 0.5 --numusers 20 --times 1
    python3 main.py --dataset Cifar10 --model cnn --batch_size 64 --learning_rate 0.05 --robust 0.8 --num_global_iters 200 --local_epochs 2 --algorithm FedPGD --subusers 0.5 --numusers 20 --times 1
    python3 main.py --dataset Cifar10 --model cnn --batch_size 64 --learning_rate 0.05 --robust 0.8 --num_global_iters 200 --local_epochs 2 --algorithm FedFGSM --subusers 0.5 --numusers 20 --times 1
cnnode></pre>

Evaluate different values of the robust parameter gamma on MNIST
<pre><code>
    # under 40% attack
    python3 main.py --dataset Mnist --model mclr --batch_size 64 --learning_rate 0.001 --robust 0.4 --gamma 0.01 --alpha 1 --num_global_iters 200 --local_epochs 2 --algorithm WAFL --subusers 0.1 --commet 1 --numusers 100 --times 1
    python3 main.py --dataset Mnist --model mclr --batch_size 64 --learning_rate 0.001 --robust 0.4 --gamma 0.05 --alpha 1 --num_global_iters 200 --local_epochs 2 --algorithm WAFL --subusers 0.1 --commet 1 --numusers 100 --times 1
    python3 main.py --dataset Mnist --model mclr --batch_size 64 --learning_rate 0.001 --robust 0.4 --gamma 0.1 --alpha 1 --num_global_iters 200 --local_epochs 2 --algorithm WAFL --subusers 0.1 --commet 1 --numusers 100 --times 1
    python3 main.py --dataset Mnist --model mclr --batch_size 64 --learning_rate 0.001 --robust 0.4 --gamma 0.5 --alpha 1 --num_global_iters 200 --local_epochs 2 --algorithm WAFL --subusers 0.1 --commet 1 --numusers 100 --times 1
    python3 main.py --dataset Mnist --model mclr --batch_size 64 --learning_rate 0.001 --robust 0.4 --gamma 1 --alpha 1 --num_global_iters 200 --local_epochs 2 --algorithm WAFL --subusers 0.1 --commet 1 --numusers 100 --times 1
    python3 main.py --dataset Mnist --model mclr --batch_size 64 --learning_rate 0.001 --robust 0.4 --gamma 5 --alpha 1 --num_global_iters 200 --local_epochs 2 --algorithm FedAvg --subusers 0.1 --numusers 100 --times 1 --commet 1
</code></pre>

Evaluate different values of robust parameter gamma on CIFAR10
<pre><code>
    # under 40% attack
    python3 main.py --dataset Cifar10 --model cnn --batch_size 64 --learning_rate 0.05 --robust 0.4 --gamma 0.05 --alpha 1 --num_global_iters 200 --local_epochs 2 --algorithm WAFL --subusers 0.5 --numusers 20 --times 1
    python3 main.py --dataset Cifar10 --model cnn --batch_size 64 --learning_rate 0.05 --robust 0.4 --gamma 0.1 --alpha 1 --num_global_iters 200 --local_epochs 2 --algorithm WAFL --subusers 0.5 --numusers 20 --times 1
    python3 main.py --dataset Cifar10 --model cnn --batch_size 64 --learning_rate 0.05 --robust 0.4 --gamma 0.5 --alpha 1 --num_global_iters 200 --local_epochs 2 --algorithm WAFL --subusers 0.5 --numusers 20 --times 1
    python3 main.py --dataset Cifar10 --model cnn --batch_size 64 --learning_rate 0.05 --robust 0.4 --gamma 1 --alpha 1 --num_global_iters 200 --local_epochs 2 --algorithm WAFL --subusers 0.5 --numusers 20 --times 1
    python3 main.py --dataset Cifar10 --model cnn --batch_size 64 --learning_rate 0.05 --robust 0.4 --gamma 5 --alpha 1 --num_global_iters 200 --local_epochs 2 --algorithm WAFL --subusers 0.5 --numusers 20 --times 1
    python3 main.py --dataset Cifar10 --model cnn --batch_size 64 --learning_rate 0.05 --robust 0.4 --gamma 10 --alpha 1 --num_global_iters 200 --local_epochs 2 --algorithm WAFL --subusers 0.5 --numusers 20 --times 1
</code></pre>

 


## Multi-Source Domain Adaptation
   
#### MNIST + SVHN to USPS
<pre><code>
    python3 main.py --dataset msda1 --model mclr --batch_size 64 --learning_rate 0.01 --robust -1 --gamma 0.1 --num_global_iters 10 --local_epochs 2 --algorithm FedAvg --subusers 1 --numusers 3 --times 1
    
    python3 main.py --dataset msda1 --model mclr --batch_size 64 --learning_rate 0.01 --robust -1 --gamma 0.1 --num_global_iters 10 --local_epochs 2 --algorithm FedAFL --subusers 1 --numusers 3 --times 1

    python3 main.py --dataset msda1 --model mclr --batch_size 64 --learning_rate 0.01 --robust -1 --gamma 0.1 --num_global_iters 10 --local_epochs 2 --algorithm FedDRFA --subusers 1 --numusers 3 --times 1
    
    python3 main.py --dataset msda1 --model mclr --batch_size 64 --learning_rate 0.01 --robust -1 --gamma 0.1 --num_global_iters 10 --local_epochs 2 --algorithm DA --subusers 1 --numusers 3 --times 1
</code></pre>

#### MNIST + USPS to SVHN
<pre><code>
    python3 main.py --dataset msda2 --model mclr --batch_size 64 --learning_rate 0.01 --robust -1 --gamma 0.1 --num_global_iters 10 --local_epochs 2 --algorithm FedAvg --subusers 1 --numusers 3 --times 1
    
    python3 main.py --dataset msda2 --model mclr --batch_size 64 --learning_rate 0.01 --robust -1 --gamma 0.1 --num_global_iters 10 --local_epochs 2 --algorithm FedAFL --subusers 1 --numusers 3 --times 1

    python3 main.py --dataset msda2 --model mclr --batch_size 64 --learning_rate 0.01 --robust -1 --gamma 0.1 --num_global_iters 10 --local_epochs 2 --algorithm FedDRFA --subusers 1 --numusers 3 --times 1
    
    python3 main.py --dataset msda2 --model mclr --batch_size 64 --learning_rate 0.01 --robust -1 --gamma 0.1 --num_global_iters 10 --local_epochs 2 --algorithm DA --subusers 1 --numusers 3 --times 1
</code></pre>


#### SVHN + USPS to MNIST
<pre><code>
    python3 main.py --dataset msda3 --model mclr --batch_size 64 --learning_rate 0.01 --robust -1 --gamma 0.1 --num_global_iters 10 --local_epochs 2 --algorithm FedAvg --subusers 1 --numusers 3 --times 1
    
    python3 main.py --dataset msda3 --model mclr --batch_size 64 --learning_rate 0.01 --robust -1 --gamma 0.1 --num_global_iters 10 --local_epochs 2 --algorithm FedAFL --subusers 1 --numusers 3 --times 1

    python3 main.py --dataset msda3 --model mclr --batch_size 64 --learning_rate 0.01 --robust -1 --gamma 0.1 --num_global_iters 10 --local_epochs 2 --algorithm FedDRFA --subusers 1 --numusers 3 --times 1
    
    python3 main.py --dataset msda3 --model mclr --batch_size 64 --learning_rate 0.01 --robust -1 --gamma 0.1 --num_global_iters 10 --local_epochs 2 --algorithm DA --subusers 1 --numusers 3 --times 1
</code></pre>
