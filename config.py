# core/config.py

import json
from pathlib import Path
from typing import Any
from datetime import datetime, timezone

class Config:
    def __init__(self, config_path: str | Path, timestamp: str | None = None):
        with open(config_path, "r") as f:
            raw = json.load(f)

        self.config_path = str(config_path)

        self.raw: dict[str, Any] = raw

        if timestamp is None:
            timestamp = datetime.now(tz=timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')

        self.name: str = raw["name"] + "_" + timestamp
        self.sample_size: int = raw["sample_size"]
        self.ensemble_size: int = raw["ensemble_size"]
        self.n_gibbs_steps: int = raw["n_gibbs_steps"]
        self.n_proposals_per_variable: int = raw["n_proposals_per_variable"]
        self.proposal_scale: float = raw["proposal_scale"]
        self.variable_names: list[str] = raw["variable_names"]
        self.num_variables: int = len(self.variable_names)
        self.num_static_fields: int = raw["num_static_fields"]
        self.max_horizon: int = raw["max_horizon"]
        self.spacing: int = raw["spacing"]
        self.t_max: int = raw["t_max"]
        self.t_direct: int = raw["t_direct"]
        self.t_iter: int = raw["t_iter"]
        self.n_ens: int = raw["n_ens"]
        self.SEED: int = raw["seed"]

        # paths
        self.data_directory: Path = Path(raw["data_directory"])
        self.model_directory: Path = Path(raw["model_directory"])
        self.result_directory: Path = Path(raw["result_directory"]) / self.name
        self.result_directory.mkdir(parents=True, exist_ok=True)

    def as_dict(self) -> dict[str, Any]:
        return self.raw
