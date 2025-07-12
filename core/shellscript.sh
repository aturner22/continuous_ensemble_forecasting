#!/bin/bash
#PBS -N gibbs_abc_gpu
#PBS -q gpu72
#PBS -l select=1:ncpus=4:mem=64gb:ngpus=1:gpu_type=L40S
#PBS -l walltime=03:00:00
#PBS -m abe
#PBS -k n
#PBS -j oe

module load tools/prod
module load SciPy-bundle/2022.05-foss-2022a

cd "$PBS_O_WORKDIR"

rm -f output_gpu.log error_gpu.log

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export TORCH_NUM_THREADS=1
export N_WORKERS=1

source venv/bin/activate

python rfp_gibbs_main.py configs/predict/deterministic-iterative-6h.json \
  --device cuda \
  > output_gpu.log 2> error_gpu.log
