from typing import Literal

import yaml  # type: ignore[import-untyped]
from pydantic import BaseModel, Field


class LoRA(BaseModel):
    r: int = Field(default=8)
    lora_alpha: int = Field(default=16)
    lora_dropout: float = Field(default=0.05)
    bias: Literal["none", "all", "lora_only"] = Field(default="none")
    target_modules: list[str] = Field(
        default_factory=lambda: [
            "q_proj",
            "v_proj",
            "fc1",
            "fc2",
            "gate_proj",
            "up_proj",
            "down_proj",
            "linear",
        ]
    )


class SFT(BaseModel):
    output_dir: str = Field(default="/model_output/lfm2-vl-food")
    num_train_epochs: int = Field(default=1)
    per_device_train_batch_size: int = Field(default=1)
    gradient_accumulation_steps: int = Field(default=16)
    learning_rate: float = Field(default=5e-4)
    warmup_ratio: float = Field(default=0.1)
    weight_decay: float = Field(default=0.01)
    max_grad_norm: float = Field(default=1.0)
    logging_steps: int = Field(default=10)
    logging_first_step: bool = Field(default=True)
    optim: str = Field(default="adamw_8bit")
    max_length: int = Field(default=512)
    report_to: str | None = Field(default=None)
    eval_strategy: str = Field(default="steps")
    eval_steps: int = Field(default=50)
    save_strategy: str = Field(default="steps")
    save_steps: int = Field(default=50)
    save_total_limit: int = Field(default=3)
    load_best_model_at_end: bool = Field(default=True)
    metric_for_best_model: str = Field(default="eval_loss")
    lr_scheduler_type: str = Field(default="cosine")


class Runtime(BaseModel):
    gpu: str = Field(default="L40S")
    timeout: int = Field(default=7200)
    volume_name: str = Field(default="food-classifier-models")
    min_containers: int = Field(default=1)
    max_retries: int = Field(default=1)


class ProcessorModel(BaseModel):
    max_image_tokens: int = Field(default=256)
    dtype: str = Field(default="bfloat16")
    device_map: str = Field(default="auto")


class TrainData(BaseModel):
    class_list: list[str] = Field(
        default_factory=lambda: [
            "hamburger",
            "garlic_bread",
            "hot_dog",
            "ceviche",
            "carrot_cake",
        ]
    )
    samples_per_class: int = Field(default=750)
    test_size: float = Field(default=0.2)
    seed: int = Field(default=40)


class Inference(BaseModel):
    max_new_tokens: int = Field(default=50)
    do_sample: bool = Field(default=False)
    temperature: float | None = Field(default=None)
    top_p: float | None = Field(default=None)


class Prompts(BaseModel):
    system_message: str = Field(
        default=(
            "You are a food classifier.\n"
            "Given an image of a food item, you identify the food category."
        )
    )
    user_message: str = Field(
        default=(
            "What food type from the following list do you see in the picture?\n\n"
            "\n\n{class_list}\n\n"
            "Provide your answer as a single food type from the list"
            "without any additional text."   
        )
    )


class FinetuneSettings(BaseModel):
    lora: LoRA = Field(default_factory=LoRA)
    sft: SFT = Field(default_factory=SFT)
    runtime: Runtime = Field(default_factory=Runtime)
    processor_model: ProcessorModel = Field(default_factory=ProcessorModel)
    train_data: TrainData = Field(default_factory=TrainData)
    inference: Inference = Field(default_factory=Inference)
    prompts: Prompts = Field(default_factory=Prompts)

    @classmethod
    def load(cls, config_path: str) -> "FinetuneSettings":
        """
        Load finetune settings from YAML file.
        """

        with open(config_path) as f:
            data = yaml.safe_load(f) or {}

        # Nested models: ignore missing keys so defaults apply
        return cls(
            lora=data.get("lora", {}),
            sft=data.get("sft", {}),
            runtime=data.get("runtime", {}),
            processor_model=data.get("processor_model", {}),
            train_data=data.get("train_data", {}),
            inference=data.get("inference", {}),
            prompts=data.get("prompts", {}),
        )
