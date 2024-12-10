# AI Preferences
This files contains a non-exhaustive list of my preferred patterns and tools to use.


## Argument parsing 
Use the Simple Parsing library for argument parsing
```python
from dataclasses import dataclass
import simple_parsing as sp

@dataclass
class Config:
    debug: bool = False  # Debug flag for quick testing
    n_debug_samples: int = 5  # Number of samples to use in debug mode
    models_dir: str = "models"  # Directory containing the scorer models
    wandb_entity: str = "c-metrics"  # W&B entity

config = sp.parse(Config)
# Optionally enable the ability to pass a config file too
# config = sp.parse(Config, config_path="configs/hallu.yaml")
```
