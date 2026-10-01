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
