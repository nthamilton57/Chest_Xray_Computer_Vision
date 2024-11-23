#!/bin/bash
#SBATCH -A account
#SBATCH -p gpu
#SBATCH -J name
#SBATCH -o out_%A_%a.txt
#SBATCH -e error_%A_%a.txt
#SBATCH --mail-type=ALL
#SBATCH --mail-user=email@email.com
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --mem=200GB
#SBATCH --time=20:00:00
#SBATCH --array=0

module load conda
conda activate triplet
# Load modules and set environment variables
module load python/gpu/3.11.5
module load cudatoolkit/12.2

# Define parameter arrays
batchsize=(32) #(16 32 64 128)
embedding=(128) #(64 128 256 512)
retrain_layer=(223) #(706 700 654 566 453 16)

cd /N/u/nothamil/BigRed200/triplet
TIMESTAMP=$(date +%s)
DATASET_PATH="dataset"

# Calculate indices for each parameter based on SLURM_ARRAY_TASK_ID
id=$SLURM_ARRAY_TASK_ID
retrain_layer_index=$((id % 6))
id=$((id / 6))
embedding_index=$((id % 4))
id=$((id / 4))
batchsize_index=$id

# Get the specific parameters for this job
bs=${batchsize[$batchsize_index]}
em=${embedding[$embedding_index]}
rl=${retrain_layer[$retrain_layer_index]}

# Run the Python script with the specific combination of parameters
python experiment.py \
    --backbone "InceptionV3" \
    --dataset="$DATASET_PATH" \
    --output="/N/u/nothamil/BigRed200/triplet/627/output-${TIMESTAMP}-${SLURM_ARRAY_TASK_ID}.zip" \
    --batch_size $bs \
    --embedding_size $em \
    --retrain_layer_count $rl \
    --learning_rate 0.001 \
    --dropout 0.2 \
    --augmentation_count 4 \
    --augmentation_factor 0.2 \
    --loss_margin 0.5 \
    --train_epochs 50 \
    --save_model False \
    --vote_count 5 \
    --verbose 2 \
    --seed 42
