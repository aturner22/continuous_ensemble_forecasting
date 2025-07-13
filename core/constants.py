from pathlib import Path
SAMPLE_SIZE = 500
ENSEMBLE_SIZE = 50
N_GIBBS_STEPS = 100
N_PROPOSALS_PER_VARIABLE = 10

VARIABLE_NAMES = ["z500", "t850", "t2m", "u10", "v10"]
NUM_VARIABLES = len(VARIABLE_NAMES)
NUM_STATIC_FIELDS = 2
MAX_HORIZON = 240

DATA_DIRECTORY = Path("./data")
MODEL_DIRECTORY = Path("./models")
RESULT_DIRECTORY = Path("./results/abc_gibbs_crps/random_testing")
RESULT_DIRECTORY.mkdir(parents=True, exist_ok=True)
