from pathlib import Path

import yaml  # type: ignore[import-untyped]
from pydantic import BaseModel, Field


class Dataset(BaseModel):
    """Dataset model."""

    dataset_id: str = Field(
        default="benitomartin/styles_with_images_384_512", description="Name of the dataset"
    )
    split: str = Field(default="train", description="Data split (train/val/test)")
    n_samples: int = Field(default=75750, description="Number of samples in the dataset")
    seed: int = Field(default=42, description="Random seed for shuffling")

    @classmethod
    def load(cls, config_path: str | None = None) -> "Dataset":
        """
        Load dataset configuration from config.yaml.
        """
        if config_path is None:
            config_path = str(Path(__file__).parent.parent / "configs" / "config_450M.yaml")
        with open(config_path) as f:
            config = yaml.safe_load(f)

        # Map config keys to model field names, filtering out None values
        data = {
            "dataset_id": config.get("dataset_id"),
            "split": config.get("split"),
            "n_samples": config.get("n_samples"),
            "seed": config.get("seed"),
        }
        # Remove None values so Pydantic can use defaults
        data = {k: v for k, v in data.items() if v is not None}
        return cls(**data)
