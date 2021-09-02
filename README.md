# WASSERSTEIN DISTRIBUTIONALLY ROBUST OPTIMIZATION FOR FEDERATED LEARNING
This repository implements all experiments in the paper the *WASSERSTEIN DISTRIBUTIONALLY ROBUST OPTIMIZATION FOR FEDERATED LEARNING*.
  
Authors: 

# Software requirements:
- numpy, scipy, torch, Pillow, matplotlib, tqdm, pandas, h5py, comet_ml, scikit-learn==0.20.3
- To download the dependencies: **pip3 install -r requirements.txt**
- The code can be run on any pc, doesn't require GPU.
  
# Datasets:

- Human Activity Recognition (30 clients)
- Vehicle Sensor (23 clients)
- MNIST (100 clients)
- CIFAR (20 clients): This dataset will be downloaded and generated automatically when runing algorithms.
- Five-digit(5 clients in total): 4 for source domain, one for target domain.

Download Link: 

All dataset after downloading must be stored at folder \data


# Training Adversarial
    - For Mnist dataset: Before running experiment, need to run generate_niid_100users.py to generate MNIST dataset

<pre><code>
    python3 main.py --dataset Mnist --model mclr --batch_size 64 --learning_rate 0.001 --robust 0 --gamma 0.05 --num_global_iters 200 --local_epochs 2 --algorithm FedRob --subusers 0.1 --numusers 100 --times 1
    python3 main.py --dataset Mnist --model mclr --batch_size 64 --learning_rate 0.001 --robust 0 --num_global_iters 200 --local_epochs 2 --algorithm FedAvg --subusers 0.1 --numusers 100 --times 1
    python3 main.py --dataset Mnist --model mclr --batch_size 64 --learning_rate 0.001 --robust 0 --num_global_iters 200 --local_epochs 2 --algorithm FedPGD --subusers 0.1 --numusers 100 --times 1
    python3 main.py --dataset Mnist --model mclr --batch_size 64 --learning_rate 0.001 --robust 0 --num_global_iters 200 --local_epochs 2 --algorithm FedFGSM --subusers 0.1 --numusers 100 --times 1

    python3 main.py --dataset Mnist --model mclr --batch_size 64 --learning_rate 0.001 --robust 0.2 --gamma 0.05 --num_global_iters 200 --local_epochs 2 --algorithm FedRob --subusers 0.1 --numusers 100 --times 1
    python3 main.py --dataset Mnist --model mclr --batch_size 64 --learning_rate 0.001 --robust 0.2 --num_global_iters 200 --local_epochs 2 --algorithm FedAvg --subusers 0.1 --numusers 100 --times 1
    python3 main.py --dataset Mnist --model mclr --batch_size 64 --learning_rate 0.001 --robust 0.2 --num_global_iters 200 --local_epochs 2 --algorithm FedPGD --subusers 0.1 --numusers 100 --times 1
    python3 main.py --dataset Mnist --model mclr --batch_size 64 --learning_rate 0.001 --robust 0.2 --num_global_iters 200 --local_epochs 2 --algorithm FedFGSM --subusers 0.1 --numusers 100 --times 1

    python3 main.py --dataset Mnist --model mclr --batch_size 64 --learning_rate 0.001 --robust 0.4 --gamma 0.05 --num_global_iters 200 --local_epochs 2 --algorithm FedRob --subusers 0.1 --numusers 100 --times 1
    python3 main.py --dataset Mnist --model mclr --batch_size 64 --learning_rate 0.001 --robust 0.4 --num_global_iters 200 --local_epochs 2 --algorithm FedAvg --subusers 0.1 --numusers 100 --times 1
    python3 main.py --dataset Mnist --model mclr --batch_size 64 --learning_rate 0.001 --robust 0.4 --num_global_iters 200 --local_epochs 2 --algorithm FedPGD --subusers 0.1 --numusers 100 --times 1
    python3 main.py --dataset Mnist --model mclr --batch_size 64 --learning_rate 0.001 --robust 0.4 --num_global_iters 200 --local_epochs 2 --algorithm FedFGSM --subusers 0.1 --numusers 100 --times 1

    python3 main.py --dataset Mnist --model mclr --batch_size 64 --learning_rate 0.001 --robust 0.6 --gamma 0.05 --num_global_iters 200 --local_epochs 2 --algorithm FedRob --subusers 0.1 --numusers 100 --times 1
    python3 main.py --dataset Mnist --model mclr --batch_size 64 --learning_rate 0.001 --robust 0.6 --num_global_iters 200 --local_epochs 2 --algorithm FedAvg --subusers 0.1 --numusers 100 --times 1
    python3 main.py --dataset Mnist --model mclr --batch_size 64 --learning_rate 0.001 --robust 0.6 --num_global_iters 200 --local_epochs 2 --algorithm FedPGD --subusers 0.1 --numusers 100 --times 1
    python3 main.py --dataset Mnist --model mclr --batch_size 64 --learning_rate 0.001 --robust 0.6 --num_global_iters 200 --local_epochs 2 --algorithm FedFGSM --subusers 0.1 --numusers 100 --times 1
    
    python3 main.py --dataset Mnist --model mclr --batch_size 64 --learning_rate 0.001 --robust 0.8 --gamma 0.05 --num_global_iters 200 --local_epochs 2 --algorithm FedRob --subusers 0.1 --numusers 100 --times 1
    python3 main.py --dataset Mnist --model mclr --batch_size 64 --learning_rate 0.001 --robust 0.8 --num_global_iters 200 --local_epochs 2 --algorithm FedAvg --subusers 0.1 --numusers 100 --times 1
    python3 main.py --dataset Mnist --model mclr --batch_size 64 --learning_rate 0.001 --robust 0.8 --num_global_iters 200 --local_epochs 2 --algorithm FedPGD --subusers 0.1 --numusers 100 --times 1
    python3 main.py --dataset Mnist --model mclr --batch_size 64 --learning_rate 0.001 --robust 0.8 --num_global_iters 200 --local_epochs 2 --algorithm FedFGSM --subusers 0.1 --numusers 100 --times 1

    python3 main.py --dataset Mnist --model mclr --batch_size 64 --learning_rate 0.001 --robust 0.9 --gamma 0.05 --num_global_iters 200 --local_epochs 2 --algorithm FedRob --subusers 0.1 --numusers 100 --times 1
    python3 main.py --dataset Mnist --model mclr --batch_size 64 --learning_rate 0.001 --robust 0.9 --num_global_iters 200 --local_epochs 2 --algorithm FedAvg --subusers 0.1 --numusers 100 --times 1
    python3 main.py --dataset Mnist --model mclr --batch_size 64 --learning_rate 0.001 --robust 0.9 --num_global_iters 200 --local_epochs 2 --algorithm FedPGD --subusers 0.1 --numusers 100 --times 1
    python3 main.py --dataset Mnist --model mclr --batch_size 64 --learning_rate 0.001 --robust 0.9 --num_global_iters 200 --local_epochs 2 --algorithm FedFGSM --subusers 0.1 --numusers 100 --times 1


</code></pre>

Evaluate different value of parameter gamma on MNIST
<pre><code>
    # clean data
    python3 main.py --dataset Mnist --model mclr --batch_size 64 --learning_rate 0.001 --robust 0 --num_global_iters 200 --local_epochs 2 --algorithm FedAvg --subusers 0.1 --numusers 100 --times 1 --commet 1
    python3 main.py --dataset Mnist --model mclr --batch_size 64 --learning_rate 0.001 --robust 0 --gamma 0.01 --num_global_iters 200 --local_epochs 2 --algorithm FedRob --subusers 0.1 --commet 1 --numusers 100 --times 1
    python3 main.py --dataset Mnist --model mclr --batch_size 64 --learning_rate 0.001 --robust 0 --gamma 0.05 --num_global_iters 200 --local_epochs 2 --algorithm FedRob --subusers 0.1 --commet 1 --numusers 100 --times 1
    python3 main.py --dataset Mnist --model mclr --batch_size 64 --learning_rate 0.001 --robust 0 --gamma 0.1 --num_global_iters 200 --local_epochs 2 --algorithm FedRob --subusers 0.1 --commet 1 --numusers 100 --times 1
    python3 main.py --dataset Mnist --model mclr --batch_size 64 --learning_rate 0.001 --robust 0 --gamma 0.15 --num_global_iters 200 --local_epochs 2 --algorithm FedRob --subusers 0.1 --commet 1 --numusers 100 --times 1
    python3 main.py --dataset Mnist --model mclr --batch_size 64 --learning_rate 0.001 --robust 0 --gamma 0.2 --num_global_iters 200 --local_epochs 2 --algorithm FedRob --subusers 0.1 --commet 1 --numusers 100 --times 1

    # under 40% attack
    python3 main.py --dataset Mnist --model mclr --batch_size 64 --learning_rate 0.001 --robust 0.4 --gamma 0.01 --num_global_iters 200 --local_epochs 2 --algorithm FedRob --subusers 0.1 --commet 1 --numusers 100 --times 1
    python3 main.py --dataset Mnist --model mclr --batch_size 64 --learning_rate 0.001 --robust 0.4 --gamma 0.05 --num_global_iters 200 --local_epochs 2 --algorithm FedRob --subusers 0.1 --commet 1 --numusers 100 --times 1
    python3 main.py --dataset Mnist --model mclr --batch_size 64 --learning_rate 0.001 --robust 0.4 --gamma 0.1 --num_global_iters 200 --local_epochs 2 --algorithm FedRob --subusers 0.1 --commet 1 --numusers 100 --times 1
    python3 main.py --dataset Mnist --model mclr --batch_size 64 --learning_rate 0.001 --robust 0.4 --gamma 0.15 --num_global_iters 200 --local_epochs 2 --algorithm FedRob --subusers 0.1 --commet 1 --numusers 100 --times 1
    python3 main.py --dataset Mnist --model mclr --batch_size 64 --learning_rate 0.001 --robust 0.4 --gamma 0.2 --num_global_iters 200 --local_epochs 2 --algorithm FedRob --subusers 0.1 --commet 1 --numusers 100 --times 1
    python3 main.py --dataset Mnist --model mclr --batch_size 64 --learning_rate 0.001 --robust 0.4 --num_global_iters 200 --local_epochs 2 --algorithm FedAvg --subusers 0.1 --numusers 100 --times 1 --commet 1
</code></pre>
## Table comparison for 

                              | Dataset        | Algorithm |         Test accuracy        |
                              |----------------|-----------|---------------|--------------|
                              |                            | Convex        | Non Convex   |
                              |----------------|-----------|---------------|--------------|
                              |                |           |               |              |
                              |                |           |               |              |
                              |                |           |               |              |
                              |                |           |               |              |
                              |                |           |               |              |
                              |                |           |               |              |
                              |                |           |               |              |
                              |                |           |               |              |
                              |----------------|-----------|---------------|--------------|
