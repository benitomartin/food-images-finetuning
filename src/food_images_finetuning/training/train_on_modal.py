import os
from collections.abc import Callable
from io import BytesIO
from typing import Any

# Modal imports
import modal
import pandas as pd
import torch
from datasets import Dataset, DatasetDict
from peft import LoraConfig, get_peft_model
from PIL import Image

# ML imports
from transformers import AutoModelForImageTextToText, AutoProcessor
from trl import SFTConfig, SFTTrainer

# Local imports
from food_images_finetuning.loaders.load_dataset import load_dataset
from food_images_finetuning.training.utils import get_docker_image, get_retries, get_volume
from food_images_finetuning.schemas.finetune import FinetuneSettings
from food_images_finetuning.schemas.model import Model as ModelConfig

# Modal configuration
app = modal.App("food-classifier-training")

docker_image = get_docker_image()
volume = get_volume("food-classifier-models")


def ensure_rgb(image: dict | Image.Image) -> Image.Image:
    """Convert image (possibly dict) to RGB PIL Image."""
    pil_image: Image.Image
    if isinstance(image, dict):
        if "bytes" in image and image["bytes"] is not None:
            pil_image = Image.open(BytesIO(image["bytes"]))
        else:
            raise ValueError("Image dict missing 'bytes' key.")
    else:
        pil_image = image
    if pil_image.mode != "RGB":
        pil_image = pil_image.convert("RGB")
    return pil_image


def format_sample_for_training(sample: dict, system_message: str, user_message: str) -> list[dict]:
    """Format a dataset sample into chat template format for fine-tuning."""

    ground_truth = sample["label"]

    image = ensure_rgb(sample["image"])

    return [
        {"role": "system", "content": [{"type": "text", "text": system_message}]},
        {
            "role": "user",
            "content": [{"type": "image", "image": image}, {"type": "text", "text": user_message}],
        },
        {
            "role": "assistant",
            "content": [{"type": "text", "text": ground_truth}],
        },
    ]


def create_collate_fn(processor: Any) -> Callable[[list[dict]], dict[str, Any]]:
    """
    Collate function with proper padding/truncation.
    """

    def collate_fn(sample: list[dict]) -> dict[str, Any]:
        # Apply chat template with proper settings
        batch = processor.apply_chat_template(
            sample,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            padding=True,  # Changed from 'max_length' to True
            truncation=True,
            max_length=512,
        )

        # Create labels by cloning input_ids and masking padding
        labels = batch["input_ids"].clone()
        labels[labels == processor.tokenizer.pad_token_id] = -100

        batch["labels"] = labels
        return batch

    return collate_fn


@app.function(
    image=docker_image,
    gpu="L40S",
    timeout=7200,
    volumes={"/model_output": volume},
    retries=get_retries(max_retries=1),
    min_containers=1,
)
def train_model(
    model_id: str = "LiquidAI/LFM2-VL-450M",
    dataset_id: str = "ethz/food101",
    split: str = "train",
    n_samples: int | None = None,
    seed: int = 42,
    config_path: str | None = None,
) -> dict[str, Any]:
    """
    Train the food classifier.
    """

    ##################################
    # DATA LOADING AND PREPARATION
    ##################################

    if not config_path:
        raise ValueError("config_path is required and must point to a finetune YAML file")

    from pathlib import Path

    container_config_path = Path("/root") / Path(config_path).relative_to("src")
    settings = FinetuneSettings.load(str(container_config_path))

    print("📥 Loading dataset...")
    full_dataset = load_dataset(
        name=dataset_id,
        split=split,
        n_samples=n_samples,
        seed=seed,
    )

    label_names = full_dataset.features["label"].names

    df = full_dataset.to_pandas()

    df["label"] = df["label"].apply(lambda x: label_names[x])

    food_list = settings.train_data.class_list

    # Create balanced dataset
    samples_per_class = settings.train_data.samples_per_class
    filtered_subsets = []
    for food in food_list:
        sub_df = df[df["label"] == food].sample(
            n=min(samples_per_class, sum(df["label"] == food)),
            random_state=settings.train_data.seed,
        )
        filtered_subsets.append(sub_df)

    balanced_df = pd.concat(filtered_subsets, ignore_index=True)
    balanced_dataset = Dataset.from_pandas(balanced_df).shuffle(seed=settings.train_data.seed)

    print(f"📊 Balanced dataset created: {len(balanced_dataset)} samples")
    print("   Distribution:")
    for food in food_list:
        count = sum(1 for _, row in balanced_df.iterrows() if row["label"] == food)
        print(f"      {food:15s}: {count}")

    dataset_split: DatasetDict = balanced_dataset.train_test_split(
        test_size=settings.train_data.test_size,
        seed=settings.train_data.seed,
    )

    # Build prompts from settings; support {class_list} replacement
    class_lines = "\n".join([f"- {c}" for c in food_list])
    system_message = settings.prompts.system_message
    user_message = settings.prompts.user_message.replace("{class_list}", class_lines)

    # Format datasets
    train_split: Dataset = dataset_split["train"]
    test_split: Dataset = dataset_split["test"]

    train_dataset = [
        format_sample_for_training(sample, system_message=system_message, user_message=user_message)
        for sample in train_split
    ]
    test_dataset = [
        format_sample_for_training(sample, system_message=system_message, user_message=user_message)
        for sample in test_split
    ]

    print("\n✅ Training data formatted:")
    print(f"   📚 Train: {len(train_dataset)} samples")
    print(f"   🧪 Test: {len(test_dataset)} samples")

    ##################################
    # MODEL AND PROCESSOR LOADING
    ##################################

    print("\n📚 Loading base processor...")
    processor = AutoProcessor.from_pretrained(
        model_id,
        max_image_tokens=settings.processor_model.max_image_tokens,
    )

    print("🧠 Loading base model...")
    dtype = torch.bfloat16 if settings.processor_model.dtype == "bfloat16" else torch.float16
    model = AutoModelForImageTextToText.from_pretrained(
        model_id,
        dtype=dtype,
        device_map=settings.processor_model.device_map,
    )

    ##################################
    # LORA CONFIGURATION
    ##################################

    peft_config = LoraConfig(
        lora_alpha=settings.lora.lora_alpha,
        lora_dropout=settings.lora.lora_dropout,
        r=settings.lora.r,
        bias=settings.lora.bias,
        target_modules=settings.lora.target_modules,
        task_type="CAUSAL_LM",
    )

    model = get_peft_model(model, peft_config)  # type: ignore
    model.print_trainable_parameters()  # type: ignore

    print("\n✅ LoRA configuration:")
    print(f"   Rank: {peft_config.r}")
    print(f"   Alpha: {peft_config.lora_alpha}")
    print(f"   Dropout: {peft_config.lora_dropout}")

    ##################################
    # SFT TRAINER CONFIGURATION
    ##################################

    sft = settings.sft
    sft_config = SFTConfig(
        output_dir=sft.output_dir,
        num_train_epochs=sft.num_train_epochs,
        per_device_train_batch_size=sft.per_device_train_batch_size,
        gradient_accumulation_steps=sft.gradient_accumulation_steps,
        learning_rate=sft.learning_rate,
        warmup_ratio=sft.warmup_ratio,
        weight_decay=sft.weight_decay,
        max_grad_norm=sft.max_grad_norm,
        logging_steps=sft.logging_steps,
        logging_first_step=sft.logging_first_step,
        optim=sft.optim,
        max_length=sft.max_length,
        dataset_kwargs={"skip_prepare_dataset": True},
        report_to=sft.report_to,
        eval_strategy=sft.eval_strategy,
        eval_steps=sft.eval_steps,
        save_strategy=sft.save_strategy,
        save_steps=sft.save_steps,
        save_total_limit=sft.save_total_limit,
        load_best_model_at_end=sft.load_best_model_at_end,
        metric_for_best_model=sft.metric_for_best_model,
        lr_scheduler_type=sft.lr_scheduler_type,
    )

    print("\n✅ Training configuration:")
    print(f"   Epochs: {sft_config.num_train_epochs}")
    print(f"   Learning rate: {sft_config.learning_rate}")
    print(f"   Eval frequency: every {sft_config.eval_steps} steps")
    print(f"   Save frequency: every {sft_config.save_steps} steps")

    ###################################
    # DATA COLLATOR AND TRAINER SETUP
    ###################################

    collate_fn = create_collate_fn(processor)

    sft_trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        data_collator=collate_fn,
        processing_class=processor.tokenizer,
    )

    ##################################
    # TRAINING LOOP START
    ##################################

    print("\n" + "=" * 70)
    print("Starting training")
    print("=" * 70 + "\n")

    train_result = sft_trainer.train()

    print("\nTraining completed")
    print(f"   Final train loss: {train_result.training_loss:.4f}")

    if sft_trainer.state.best_metric is not None:
        print(f"   Best eval loss: {sft_trainer.state.best_metric:.4f}")
        if sft_trainer.state.best_model_checkpoint:
            print(f"   Best checkpoint: {sft_trainer.state.best_model_checkpoint}")

    # Save the final model
    print(f"\n💾 Saving final model to: {sft_config.output_dir}")
    sft_trainer.save_model()

    # Save LoRA adapter
    adapter_path = f"{sft_config.output_dir}/lora_adapter"
    print(f"💾 Saving LoRA adapter to: {adapter_path}")
    model.save_pretrained(adapter_path)
    processor.save_pretrained(adapter_path)
    print(f"✅ LoRA adapter saved to: {adapter_path}")

    print("💾 Committing volume (this may take a moment)...")
    volume.commit()
    print("✅ Volume committed - model persisted!")

    return {
        "status": "success",
        "output_dir": sft_config.output_dir,
        "train_samples": len(train_dataset),
        "eval_samples": len(test_dataset),
        "final_train_loss": train_result.training_loss,
        "lora_rank": settings.lora.r,
        "learning_rate": settings.sft.learning_rate,
    }


@app.function(
    image=docker_image,
    volumes={"/model_output": volume},
)
def download_model_files(model_path: str = "/model_output/lfm2-vl-food/lora_adapter") -> bytes:
    """Download model files from Modal volume to local filesystem."""
    import tarfile
    from io import BytesIO

    print(f"📦 Creating tarball of model at {model_path}...")

    tar_buffer = BytesIO()
    with tarfile.open(fileobj=tar_buffer, mode="w:gz") as tar:
        tar.add(model_path, arcname=os.path.basename(model_path))

    tar_buffer.seek(0)
    print(f"✅ Tarball created ({len(tar_buffer.getvalue()) / 1e6:.2f} MB)")

    return tar_buffer.getvalue()


@app.local_entrypoint()
def main(
    download_after_training: bool = True,
    finetune_config: str | None = None,
    config: str | None = None,
) -> None:
    """Main entry point for local execution."""
    import time

    print("Starting Food Classifier Training on Modal")
    if not finetune_config:
        raise ValueError("Please provide --finetune-config path to a finetune YAML file")
    if not config:
        raise ValueError("Please provide --config path to a model/dataset YAML file")

    print("\nUsing finetune config:", finetune_config)
    print("Using config:", config)

    from food_images_finetuning.schemas.dataset import Dataset as DatasetConfig

    model_config = ModelConfig.load(config)
    dataset_config = DatasetConfig.load(config)
    settings = FinetuneSettings.load(finetune_config)

    model_id = model_config.model_id
    dataset_id = dataset_config.dataset_id
    split = dataset_config.split
    n_samples = dataset_config.n_samples
    seed = dataset_config.seed

    print(f"   Model: {model_id}")
    print(f"   Dataset: {dataset_id}")
    print(f"   Split: {split}")
    print(f"   n_samples: {n_samples}")
    print(f"   seed: {seed}")

    print("\n⏳ Spawning training job (non-blocking)...")
    print("   This allows the training to run without connection timeouts")

    # Use spawn() instead of remote() to avoid blocking and timeouts
    # Note: runtime (gpu/timeout/volumes) is defined by the decorator and cannot be changed here
    function_call = train_model.spawn(
        model_id=model_id,
        dataset_id=dataset_id,
        split=split,
        n_samples=n_samples,
        seed=seed,
        config_path=finetune_config,
    )

    print(f"Training job spawned: {function_call.object_id}")
    print("   View logs at: https://modal.com/apps")

    # Poll for completion with status updates
    print("\nWaiting for training to complete...")
    print("   (This may take 15-30 minutes depending on the dataset size)")

    last_status_time = time.time()
    dots = 0

    while True:
        try:
            # Try to get result with a short timeout
            result = function_call.get(timeout=30)
            print("\n\nTraining completed successfully")
            print(f"   Result: {result}")
            break
        except TimeoutError:
            # Job still running - print status update
            current_time = time.time()
            if current_time - last_status_time >= 30:
                dots = (dots + 1) % 4
                print(f"\r   Still training{'.' * dots}{' ' * (3 - dots)}", end="", flush=True)
                last_status_time = current_time
        except Exception as e:
            print(f"\n❌ Error while waiting for training: {e}")
            print("   Check Modal dashboard for full logs")
            return None

    if download_after_training:
        print("\n📥 Downloading trained model to local filesystem...")

        import tarfile
        from io import BytesIO

        modal_adapter_path = f"{settings.sft.output_dir}/lora_adapter"
        model_bytes = download_model_files.remote(model_path=modal_adapter_path)

        model_name = os.path.basename(settings.sft.output_dir)
        local_model_dir = f"./models/{model_name}"
        os.makedirs(local_model_dir, exist_ok=True)

        print(f"📂 Extracting model to {local_model_dir}...")
        tar_buffer = BytesIO(model_bytes)
        with tarfile.open(fileobj=tar_buffer, mode="r:gz") as tar:
            tar.extractall(path=local_model_dir)

        print(f"Model downloaded to: {local_model_dir}")
        print(f"   Adapter path: {local_model_dir}/lora_adapter")

    # return result


if __name__ == "__main__":
    print("🔧 Running in local mode (without Modal)")
    print("To run on Modal, use: modal run train_on_modal_optimized.py")
