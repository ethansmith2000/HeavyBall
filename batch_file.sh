#!/bin/bash
#SBATCH --job-name=ethan_cifar_test
#SBATCH --partition=gpu-nvidia-a100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=96:00:00
#SBATCH --output=./slurm/logs/%j-%x.out
#SBATCH --gpus-per-task=8
#SBATCH --cpus-per-task=96
#SBATCH --requeue
#SBATCH --exclusive

export YOUR_NAME=ethan
echo "STARTING SBATCH..."
nodes=( $( scontrol show hostnames $SLURM_JOB_NODELIST ) )
nodes_array=($nodes)
head_node=${nodes_array[0]}
head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)
export HEAD_NODE_IP=$head_node_ip
echo Node IP: $head_node_ip
export LOGLEVEL=INFO
source /opt/conda/bin/activate /efs/ethan-home/.conda/envs/leo-train
OD_SRC=/efs/${YOUR_NAME}-home/HeavyBall
PYTHONPATH=/efs/ethan-home/.conda/envs/leo-train/bin/python:$OD_SRC:$PYTHONPATH
export PATH=/efs/ethan-home/.conda/envs/leo-train/bin:$PATH
FI_LOG_LEVEL=1
NCCL_DEBUG=INFO
PATH=$(echo $PATH | tr ":" "\n" | grep -v '/usr/local/cuda' | paste -sd ":" -)
LD_LIBRARY_PATH=$(echo $LD_LIBRARY_PATH | tr ":" "\n" | grep -v '/usr/local/cuda' | paste -sd ":" -)
echo "LAUNCHING RUN..."
if [ 8 -eq 1 ]; then
    echo "Running on single GPU"
    python $OD_SRC/train.py +experiment=cifar_test experiment.name="${SLURM_JOB_ID}_cifar_test" system.debug=False experiment.slurm_job_num_nodes=$SLURM_JOB_NUM_NODES experiment.slurm_job_id=$SLURM_JOB_ID experiment.your_name=$YOUR_NAME
    exit 0
fi

srun torchrun --nnodes $SLURM_JOB_NUM_NODES --nproc_per_node 8 --rdzv_id $RANDOM --rdzv_backend c10d --rdzv_endpoint $head_node_ip:29500 ./cifar10.py
