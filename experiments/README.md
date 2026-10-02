# Experiments

Code for replicating experiments described in Section VII.C and VII.D.

run\_unlearning.py: Contains code run a multi-step unlearning process. Starts by training an initial model, then selects an ordering of datapoints to be unlearned and applies an unlearning algorithm to remove them. Repeats a specified number of times. Requests can be determined in an adaptive or nonadaptive manner.

Unlearning Algorithms:
* Retrain from scratch
* Relabel
* SCRUB

Deletion Requesters:
* Random
* Loss
* LiRA
* Shapley
* Independent Shapley

The data/ subdirectory contains logits extracted from shadow models used for computations of LiRA scores by the MIA requester, as well as the subsets of CIFAR-10 those shadow models were trained on.
