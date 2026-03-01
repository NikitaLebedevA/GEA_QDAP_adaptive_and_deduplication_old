#!/bin/bash
#SBATCH --time=10-10:30
#SBATCH --job-name=generalized_quadratic # Job name
#SBATCH --error=generalized_quadratic-%j.err # File to output errors
#SBATCH --output=generalized_quadratic-%j.log # File to output results
#SBATCH --ntasks 1 # Number of MPI processes
#SBATCH --cpus-per-task 1 # Number of CPUs per task
#SBATCH --mail-user=msohrabi@hse.ru # Enter your email to send notifications
#SBATCH --mail-type=END,FAIL # Events that require notification

module load matlab/r2020a # Loading a MATLAB module
# Run MATLAB with Run.m file
matlab -nodisplay -nosplash -r "run('Run_c201575_Mutation_Sensitivity.m'); exit" > result.dat