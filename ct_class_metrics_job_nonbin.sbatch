#!/bin/bash
###############################################################################
##                                                                           ##
##                           Campus Cluster                                  ##
##                   Sample Python Job Batch Script                          ##
##                                                                           ##
## SLURM Options (To view, run the following command below)                  ##
##                                                                           ##
##     man sbatch                                                            ##
##                                                                           ##
###############################################################################
#
#SBATCH --time=00:05:00                  # Job run time (hh:mm:ss)
#SBATCH --nodes=1                        # Number of nodes
#SBATCH --ntasks-per-node=16             # Number of task (cores/ppn) per node
#SBATCH --job-name=python_job            # Name of batch job
#SBATCH --partition=secondary            # Partition (queue)
#SBATCH --account=ACCT_NAME              # Bactch account to use
#SBATCH --output=python_job.o%j          # Name of batch job output file
##SBATCH --error=python_job.e%j          # Name of batch job error file
##SBATCH --mail-user=NetID@illinois.edu  # Send email notifications
##SBATCH --mail-type=BEGIN,END           # Type of email notifications to send
#
###############################################################################
# Change to the directory from which the batch job was submitted
# Note: SLURM defaults to running jobs in the directory where
# they are submitted, no need for cd'ing to $SLURM_SUBMIT_DIR

#cd ${SLURM_SUBMIT_DIR}


# Load Python/Anaconda module (Enable Python in batch job environment)
# by loading a Python or Anaconda modulefile (Only one modulfile should be
# loaded. To use the python modulefile below, uncomment that "module load ..."
# line and comment out the anaconda "module load ..." line.)

source activate cellsighter_UNI_env

# Run the python script(code)
python get_metrics.py --model_path /projects/illinois/vetmed/cb/kwang222/cellsighter_testing/shirui_code/CellSighter/experiment_results/new_full_class_testing_results/results_ver3_mask_aug_55epochs/checkpoint_epoch_55.pth --model resnet --config /projects/illinois/vetmed/cb/kwang222/cellsighter_testing/shirui_code/CellSighter/src/config_new_more_cts_eval.json