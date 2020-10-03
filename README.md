# Weighted_Bayesian_Bootstrap

We explore bootstrap methods for deep networks to qualify uncertainty by approximately calculating posterior distribution of regression models.
Weighted Bayesian Bootstrap (WBB) is a simulation-based algorithm for assessing uncertainty in machine learning and statistics [1]. In this project, we implement weighted bayesian bootstrap on ​datasets supplied from the ​UCI machine learning repository and make comparisons with the results of probabilistic backpropagation method (PBP) [2] and Dropout’s uncertainty [3]

# Data set

We train and test models on the datasets which are taken from the UCI machine learning repository. Due to the small size of the data, if we ourselves split the data we will most likely get different and non-comparable results. So we keep train on split datasets which are identical to the ones used in Hernández-Lobato's code and make comparisons.
We use the experimental setup proposed by Hernandez-Lobato and Adams [2] for evaluating PBP. Each dataset is split into 20 train-test folds, except for the protein dataset which uses 5 folds and the Year Prediction MSD dataset which uses a single train-test split.

# Experiment

In our experiment, we use the same network architecture: 1-hidden layer neural network containing 50 hidden units with a ReLU activation.
For each splitted train-test folds, we train an ensemble​ ​of 100 networks with different random initializations. We train 300 epochs for each model. Our results are shown in table below, along with the results of PBP method [2] and Dropout method [3].

# References

[1] ​Newton, Michael, Nicholas G. Polson, and Jianeng Xu. "Weighted Bayesian Bootstrap for Scalable Bayes." ​arXiv preprint arXiv:1803.04559​ (2018).
[2] Hernández-Lobato, José Miguel, and Ryan Adams. "Probabilistic backpropagation for scalable learning of bayesian neural networks." ​International Conference on Machine Learning​. 2015.
[3] Gal, Yarin, and Zoubin Ghahramani. "Dropout as a bayesian approximation: Representing model uncertainty in deep learning." ​international conference on machine learning​. 2016.
