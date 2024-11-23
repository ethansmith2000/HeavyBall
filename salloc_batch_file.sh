
source /opt/conda/bin/activate /efs/ethan-home/.conda/envs/leo-train
conda deactivate
conda activate /efs/ethan-home/.conda/envs/leo-train
export YOUR_NAME=ethan
echo "STARTING SBATCH..."
nodes=( $( scontrol show hostnames $SLURM_JOB_NODELIST ) )
nodes_array=($nodes)
head_node=${nodes_array[0]}
head_node_ip=$(nslookup $(hostname) | awk '/^Address: / { print $2 }')
export HEAD_NODE_IP=$head_node_ip
echo Node IP: $head_node_ip
export LOGLEVEL=INFO
OD_SRC=/efs/${YOUR_NAME}-home/HeavyBall
WORKDIR=/efs/${YOUR_NAME}-home/workdir
PYTHONPATH=/efs/ethan-home/.conda/envs/leo-train/bin/python:$OD_SRC:$PYTHONPATH
FI_LOG_LEVEL=1
NCCL_DEBUG=INFO
PATH=$(echo $PATH | tr ":" "\n" | grep -v '/usr/local/cuda' | paste -sd ":" -)
LD_LIBRARY_PATH=$(echo $LD_LIBRARY_PATH | tr ":" "\n" | grep -v '/usr/local/cuda' | paste -sd ":" -)
echo "LAUNCHING RUN..."
if [ 8 -eq 1 ]; then
    echo "Running on single GPU"
    python $OD_SRC/train.py +experiment=cifar_train experiment.name="${SLURM_JOB_ID}_cifar_train" experiment.slurm_job_num_nodes=$SLURM_JOB_NUM_NODES experiment.slurm_job_id=$SLURM_JOB_ID experiment.your_name=$YOUR_NAME
    exit 0
fi
torchrun --nnodes $SLURM_JOB_NUM_NODES --nproc_per_node 8 --rdzv_id $RANDOM --rdzv_backend c10d --rdzv_endpoint $head_node_ip:29500 ./cifar10.py
