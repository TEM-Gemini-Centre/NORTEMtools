#!/bin/bash
#SBATCH --partition=CPUQ
#SBATCH --cpus-per-task=5
#SBATCH --time=0-00:20:00
#SBATCH --job-name="CalCheck"
#SBATCH --mem=64000
#SBATCH --mail-type=END,FAIL,START
#SBATCH --mail-user=emil.christiansen@ntnu.no
#SBATCH --array=0-4
echo "This is job $SLURM_JOB_ID with name $SLURM_JOB_NAME submitted from $SLURM_SUBMIT_DIR"
echo "This job was run on $SLURM_JOB_NUM_NODES nodes using $SLURM_CPUS_ON_NODE cores per node with a total of $SLURM_NTASKS cores on these nodes: $SLURM_JOB_NODELIST"

# Load modules
module purge

# **CHANGE TO SUIT YOUR WORK**

# Parameters 
n_random=10 #Number of sampling points to pick from data
n=5 #The number of calibrations to test
start=0.6 #The relative calibration value to start with
end=1.2 #The relative 
GUESS=(0.0127931 0.0106398 0.00918478 0.00756774 0.00593451) #Tabulated calibration values for 8, 10, 12, 15, 20 cm camera lengths
NAME=("8cm" "10cm" "12cm" "15cm" "20cm") #Names for output files

# Data paths
datafilepath='/cluster/projects/itea_lille-nv-fys-tem/DemoData/SPED_Ag/SPED_cropped_centered.hspy'
templatepath='/cluster/projects/itea_lille-nv-fys-tem/DemoData/SPED_Ag/Ag.pkl'

# ** DO NOT CHANGE CODE AFTER THIS LINE**

# Script paths
pythonpath='/cluster/projects/itea_lille-nv-fys-tem/miniforge3/envs/pyxem0.21.0/bin/python'
pythonscriptpath='./calibration_check.py'

#Run Script
${pythonpath} ${pythonscriptpath} ${datafilepath} -v -n ${n} --start ${start} --end ${end} --n_random ${n_random} --guess ${GUESS[$SLURM_ARRAY_TASK_ID]} --name ${NAME[$SLURM_ARRAY_TASK_ID]} --template_file ${templatepath}

# Display job details
scontrol show job ${SLURM_JOB_ID} -d
