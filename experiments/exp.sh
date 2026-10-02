#!/bin/bash

BASE_PARAMS="-p general  --gres gpu:1 --mem 64G -o ./out/slurm-%j.out --time 10:00:00" # Add slurm paramters here (partition, resources, time, etc.)
EXP=3

if [[ $EXP -eq 1 ]]
then
    for requester in random lira loss shapley
    do
        for unlearn_method in retrain scrub relabel
        do 
            for i in 1 2 3 4
            do
                SLURM_PARAMS="$BASE_PARAMS"

                # Nonadaptive requester
                sbatch $SLURM_PARAMS --wrap "CUDA_LAUNCH_BLOCKING=1 uv run python run_unlearning.py $i --unlearn_type $requester --num_unlearned 1000 --num_iters 10 --num_epochs 50 --unlearn_method $unlearn_method --non_adaptive --no_model_save --suffix ${requester}" 

                # Adaptive requester
                sbatch $SLURM_PARAMS --wrap "CUDA_LAUNCH_BLOCKING=1 uv run python run_unlearning.py $i --unlearn_type $requester --num_unlearned 1000 --num_iters 10 --num_epochs 50 --unlearn_method $unlearn_method --no_model_save --suffix ${requester}" 
            done
        done
    done
elif [[ $EXP -eq 2 ]]
then
    for i in 1 2 3 4
    do
        SLURM_PARAMS="$BASE_PARAMS"

        # Random requester
        sbatch $SLURM_PARAMS --wrap "CUDA_LAUNCH_BLOCKING=1 uv run python run_unlearning.py $i --unlearn_type random --num_unlearned 1000 --num_iters 10 --num_epochs 50 --unlearn_method retrain --non_adaptive --no_model_save --suffix ${requester}" 

        # Shapley requester
        sbatch $SLURM_PARAMS --wrap "CUDA_LAUNCH_BLOCKING=1 uv run python run_unlearning.py $i --unlearn_type shapley --num_unlearned 1000 --num_iters 10 --num_epochs 50 --unlearn_method retrain --non_adaptive --no_model_save --suffix ${requester}" 

        # Individual Shapley requester
        sbatch $SLURM_PARAMS --wrap "CUDA_LAUNCH_BLOCKING=1 uv run python run_unlearning.py $i --unlearn_type masked_shapley --num_unlearned 1000 --num_iters 10 --num_epochs 50 --unlearn_method retrain --non_adaptive --no_model_save --suffix ${requester}" 

        # Adaptive Shapley requester
        sbatch $SLURM_PARAMS --wrap "CUDA_LAUNCH_BLOCKING=1 uv run python run_unlearning.py $i --unlearn_type shapley --num_unlearned 1000 --num_iters 10 --num_epochs 50 --unlearn_method retrain --no_model_save --suffix ${requester}" 
    done
elif [[ $EXP -eq 3 ]]
then
    SLURM_PARAMS="$BASE_PARAMS"
	# Single Shapley requester run
	sbatch $SLURM_PARAMS --wrap "CUDA_LAUNCH_BLOCKING=1 uv run python run_unlearning.py 1 --unlearn_type shapley --num_unlearned 1000 --num_iters 10 --num_epochs 50 --unlearn_method retrain --non_adaptive --no_model_save --suffix shapley"
fi
