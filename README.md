#Transferable Tree-based Ensemble Model for Non-Intrusive Load Monitoring

This code implements  a collaborative model-based transfer learning framework for LightGBM
# How to use the code and example

Directory tree:

``` bash
├── NILM_data_management
│   ├── redd_process.py
│   ├── refit_process.py
│   └── ukdale_process.py
├──LightGBM
│   ├── lightree.py
│   └── TreeBuild.py
├──CNN
│   ├── CNN_TransferLearning.py
│   └── CNNtrain.py
├── GAN
│   ├── GAN_disaggregate_dishwasher.ipynb
│   ├── GAN_disaggregate_fridge.ipynb
│   ├── GAN_disaggregate_washingmachine.ipynb
│   └── GAN_disaggregate_microwave.ipynb
├──treeStructure
│   ├── redd
│   └── ukdale
```
## **Create UK-DALE or REDD dataset**
Datasets are built by the power readings of houses 1, 2, and 3 from REDD, and building 1, 2 from UK-DALE. For all datasets, we sampled the active load every 8 seconds. The commonly-used appliances were chosen to implement model training, such as dishwasher, fridge, washing machine,  and microwave. For each dataset, 80% of samples were used for model training, and the remaining for testing. The scripts contained in NILM_data_management directory allow the user to create CSV files of training dataset across different application scenarios. Any output CSV file represents a dataset owned by a householder.

## **Training**
The lightgbm.py script provides booster class and main training workflow of LightGBM for NILM. It integrates with Seq2point paradigm for NILM data (pairs of multiple samples aggregate data and 1 sample midpoint ground truth). The TreeBuild.py is used to perform transfer learning by revising the generated tree structures.

Training default parameters:

* Windowsize: 19 samples
* Number of boosting round: 100
* Learning rate: 0.23179
* Maximum depth: 10
* Maximum bins: 500
* L1 regularisation: 0.02145
* L2 regularisation: 0.0001

# Dataset
Datasets used can be found:
1. UKDALE: https://jack-kelly.com/data/
2. REDD: http://redd.csail.mit.edu/
