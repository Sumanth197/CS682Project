#!/bin/bash
#SBATCH --partition=gpuq                    # the DGX only belongs in the 'gpu'  partition
#SBATCH --qos=gpu                           # need to select 'gpu' QoS
#SBATCH --job-name=shape_bbox_blip_model
## Separate output and error messages into 2 files.
## NOTE: %u=userID, %x=jobName, %N=nodeID, %j=jobID
#SBATCH --output=%x-%N-%j.out  # Output file
#SBATCH --error=%x-%N-%j.err   # Error file
## Slurm can send you updates via email
#SBATCH --mail-type=ALL         # ALL,NONE,BEGIN,END,FAIL,REQUEUE,..
#SBATCH --mail-user=smanduru@gmu.edu     # Put your GMU email address here
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1                 # up to 128; 
#SBATCH --gres=gpu:A100.80gb:1              # up to 8; only request what you need
#SBATCH --mem=100GB                 # memory per CORE; total memory is 1 TB (1,000,000 MB)
#SBATCH --export=ALL 
#SBATCH --time=3-0:00:00                   # set to 1hr; please choose carefully

set echo
umask 0027

# to see ID and state of GPUs assigned
nvidia-smi

module load gnu10
source /home/smanduru/CS682Project/cvp/bin/activate
# cd /home/smanduru/HighSpeedRL
# cd /scratch/smanduru/ML_688/DropNet
cd /home/smanduru/CS682Project/code


# PYTHONPATH=$(pwd) python3 rl_runners/aoa_runners/tune_ppo.py
# PYTHONPATH=$(pwd) python3 model_testing/graph_visual.py

python3 blip_model_color.py