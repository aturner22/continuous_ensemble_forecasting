#!/usr/bin/env python3
"""
Verbose wrapper for ABC‑Gibbs‑RFP.  Keeps original CLI but adds:
 • live matplotlib figure of α‑trace & joint‑CRPS
 • per‑step dump of coverage/spread
"""

import matplotlib.pyplot as plt
import numpy as np, torch, argparse, importlib, sys, time
from pathlib import Path
from rfp_gibbs_main import main as inner_main           # your existing driver
from core import gibbs_abc_threaded_rfp as gibbs

# ── hook: after every Gibbs step ────────────────────────────────────────────
def _after_step_hook(state):
    step      = state["step"]
    a_mean    = state["posterior_samples"][:step+1,:,0].mean(0)
    j_trace   = state["step_mean_crps"][:step+1]

    plt.clf(); plt.subplot(2,1,1)
    plt.plot(j_trace, marker='o'); plt.title("joint‑CRPS"); plt.grid(True)
    plt.subplot(2,1,2); plt.plot(a_mean, marker='x'); plt.title("α mean"); plt.grid(True)
    plt.pause(0.05)

    cov = state.get("coverage", None)
    if cov is not None:
        print(f"step {step+1}: cover={cov:.3f}")

# patch it in
gibbs._AFTER_STEP_CALLBACK = _after_step_hook    # type: ignore[attr-defined]

# ── run the normal script ───────────────────────────────────────────────────
if __name__ == "__main__":
    plt.ion(); inner_main(); plt.ioff(); plt.show()
