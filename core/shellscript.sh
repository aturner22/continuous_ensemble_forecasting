# ── submit.sh ─────────────────────────────────────────────────────
#!/bin/bash
#PBS -N gibbs_abc_job
#PBS -q standard
#PBS -l ncpus=16,mem=8gb
#PBS -m be
#PBS -j oe                         # merge stdout / stderr
#PBS -V                            # export caller's environment

cd "$PBS_O_WORKDIR"

export N_WORKERS=16                # tell Python how many workers to spin up
export PARALLEL_BACKEND=process    # safer on pure-CPU nodes; use 'thread' for GPU

python abc_gibbs_crps_main.py configs/predict/deterministic-iterative-6h.json