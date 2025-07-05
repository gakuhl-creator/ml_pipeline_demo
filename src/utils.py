import os
import yaml

def load_config():
    config_path = os.path.join(os.path.dirname(__file__), "./config.yaml")
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def resolve_path(relative_path):
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    return os.path.join(project_root, relative_path)
