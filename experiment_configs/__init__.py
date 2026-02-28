# Experiment config modules — each file exposes:
#   EXPERIMENTS: dict[str, ExperimentConfig]
#
# Drop a new .py file here and it will be auto-discovered by experiments.py.
# Each file can also be run standalone:
#   python experiment_configs/transformer.py --list
#   python experiment_configs/transformer.py transformer_debug
