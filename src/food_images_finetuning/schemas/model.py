from pathlib import Path

import yaml  # type: ignore[import-untyped]
from pydantic import BaseModel, Field


class Model(BaseModel):
    """Model configuration."""

    model_id: str = Field(default="LiquidAI/LFM2-VL-450M", description="Name of the model")

    @classmethod
    def load(cls, config_path: str | None = None) -> "Model":
        """
        Load model configuration from YAML file.
        """
        if config_path is None:
            config_path = str(Path(__file__).parent.parent / "configs" / "config_450M.yaml")
        with open(config_path) as f:
            config = yaml.safe_load(f)

        # Map config keys to model field names, filtering out None values
        data = {
            "model_id": config.get("model_id"),
        }
        # Remove None values so Pydantic can use defaults
        data = {k: v for k, v in data.items() if v is not None}
        return cls(**data)
