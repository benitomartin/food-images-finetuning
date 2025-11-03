"""
Evaluation script for fine-tuned Foood classifier using plain text output format.
"""

from io import BytesIO

import pandas as pd
import torch
from datasets import Dataset
from peft import PeftModel
from PIL import Image
from tqdm import tqdm
from transformers import AutoModelForImageTextToText, AutoProcessor

from food_images_finetuning.schemas.dataset import Dataset as DatasetConfig
from food_images_finetuning.schemas.finetune import FinetuneSettings
from food_images_finetuning.schemas.model import Model as ModelConfig


class FoodEvaluator:
    """Evaluator for food classification model."""

    def __init__(
        self,
        base_model_id: str = "LiquidAI/LFM2-VL-450M",
        lora_adapter_path: str = "models/lfm2-vl-food/lora_adapter",
        device: str = "auto",
        finetune_config: str | None = None,
    ):
        """
        Initialize the evaluator with base model and LoRA adapter.
        """
        self.base_model_id = base_model_id
        self.lora_adapter_path = lora_adapter_path
        self.device = device

        if not finetune_config:
            raise ValueError("finetune_config path is required")
        self.settings = FinetuneSettings.load(finetune_config)
        self.food_list = self.settings.train_data.class_list

        print(f"📚 Loading processor... for model: {base_model_id}")
        self.processor = AutoProcessor.from_pretrained(
            base_model_id,
            max_image_tokens=self.settings.processor_model.max_image_tokens,
        )

        print("🧠 Loading base model...")
        dtype = (
            torch.bfloat16 if self.settings.processor_model.dtype == "bfloat16" else torch.float16
        )
        self.base_model = AutoModelForImageTextToText.from_pretrained(
            base_model_id,
            dtype=dtype,
            device_map=device,
        )

        print("🔧 Loading LoRA adapter...")
        self.model = PeftModel.from_pretrained(
            self.base_model,
            lora_adapter_path,
            dtype=torch.bfloat16,
        )
        self.model.eval()

        print("✅ Merged model loaded successfully!")

    def load_balanced_dataset(
        self,
        dataset_name: str = "ethz/food101",
        samples_per_class: int | None = None,
        test_size: float | None = None,
        seed: int | None = None,
    ) -> dict:
        """Load and prepare balanced dataset with same approach as training."""
        print(f"📥 Loading dataset: {dataset_name}")

        from food_images_finetuning.loaders.load_dataset import load_dataset

        full_dataset = load_dataset(
            name=dataset_name,
            split="train",
            n_samples=None,
        )

        label_names = full_dataset.features["label"].names

        df = full_dataset.to_pandas()
        df["label"] = df["label"].apply(lambda x: label_names[x])

        print(f"DF: {df.head()}")
        df = df[["label", "image"]]

        if samples_per_class is None:
            samples_per_class = self.settings.train_data.samples_per_class
        if test_size is None:
            test_size = self.settings.train_data.test_size
        if seed is None:
            seed = self.settings.train_data.seed

        print(f"📊 Creating balanced dataset with {samples_per_class} samples per class...")

        filtered_subsets = []
        for food in self.food_list:
            sub_df = df[df["label"] == food].sample(
                n=min(samples_per_class, sum(df["label"] == food)), random_state=42
            )
            filtered_subsets.append(sub_df)

        balanced_df = pd.concat(filtered_subsets, ignore_index=True)
        balanced_dataset = Dataset.from_pandas(balanced_df).shuffle(seed=seed)

        print(f"   Total samples: {len(balanced_dataset)}")
        print("   Food type distribution:")
        for food in self.food_list:
            count = sum(1 for _, row in balanced_df.iterrows() if row["label"] == food)
            print(f"      {food:15s}: {count}")

        split = balanced_dataset.train_test_split(test_size=test_size, seed=seed)

        print("\n✅ Dataset prepared:")
        print(f"   📚 Train: {len(split['train'])} samples")
        print(f"   🧪 Test: {len(split['test'])} samples")

        return split

    def ensure_rgb(self, image: dict | Image.Image) -> Image.Image:
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

    def evaluate_sample(self, sample: dict, show_prediction: bool = False) -> dict:
        """
        Evaluate a single sample using plain text output format.
        """
        ground_truth = sample["label"]

        # Prepare image
        image = self.ensure_rgb(sample["image"])

        # Build prompt from finetune settings; support {class_list}
        class_lines = "\n".join([f"- {c}" for c in self.food_list])
        prompt_text = self.settings.prompts.user_message.replace("{class_list}", class_lines)

        # Format as conversation
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt_text},
                ],
            }
        ]

        # Apply chat template
        try:
            inputs = self.processor.apply_chat_template(
                messages,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
                add_generation_prompt=True,
            ).to(self.model.device)

            # Generate prediction
            with torch.no_grad():
                gen = self.settings.inference
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=gen.max_new_tokens,
                    do_sample=gen.do_sample,
                    temperature=gen.temperature,
                    top_p=gen.top_p,
                )

            # Decode the output
            generated_ids = outputs[0][inputs["input_ids"].shape[1] :]
            prediction_text = self.processor.tokenizer.decode(
                generated_ids, skip_special_tokens=True
            ).strip()

            # Clean up prediction (remove any extra text)
            prediction = prediction_text.split("\n")[0].strip()

            # Check if prediction is in valid list
            if prediction not in self.food_list:
                # Try to find closest match
                prediction_lower = prediction.lower()
                for food in self.food_list:
                    if food.lower() in prediction_lower or prediction_lower in food.lower():
                        prediction = food
                        break

            if show_prediction:
                match_symbol = "✅" if prediction.lower() == ground_truth.lower() else "❌"
                print(f" {prediction:15s} vs {ground_truth:15s} {match_symbol}")

            return {
                "prediction": prediction,
                "ground_truth": ground_truth,
                "success": True,
                "raw_output": prediction_text,
            }
        except Exception as e:
            if show_prediction:
                print(f"  ERROR - {str(e)}")
            return {
                "prediction": None,
                "ground_truth": ground_truth,
                "success": False,
                "error": str(e),
            }

    def evaluate_dataset(self, test_dataset: Dataset, show_first: int = 5) -> dict:
        """Evaluate the entire test dataset."""
        print("🧪 Running evaluation on test set...")
        print(f"   Test samples: {len(test_dataset)}\n")

        results = []
        correct = 0
        total_success = 0

        for i, sample in enumerate(tqdm(test_dataset, desc="Evaluating")):
            result = self.evaluate_sample(sample, show_prediction=(i < show_first))
            results.append(result)

            if result["success"]:
                total_success += 1
                if result["prediction"].lower() == result["ground_truth"].lower():
                    correct += 1

        # Calculate metrics
        print("\n" + "=" * 70)
        print("📊 EVALUATION RESULTS")
        print("=" * 70)

        print(
            f"\n✅ Successful predictions: \n"
            f"{total_success}/{len(test_dataset)} ({100 * total_success / len(test_dataset):.1f}%)"
        )

        if total_success > 0:
            accuracy = 100 * correct / total_success
            print("\n📈 Accuracy:")
            print(f"   Exact match:   {correct}/{total_success} ({accuracy:.1f}%)")

        # Per-class accuracy
        print("\n📊 Per-Class Accuracy:")
        for food in self.food_list:
            food_samples = [r for r in results if r["ground_truth"] == food and r["success"]]
            if food_samples:
                correct_class = sum(
                    1 for r in food_samples if r["prediction"].lower() == food.lower()
                )
                acc = 100 * correct_class / len(food_samples)
                print(f"   {food:15s}: {correct_class:3d}/{len(food_samples):3d} ({acc:5.1f}%)")

        # Show wrong predictions
        wrong_predictions = [
            r
            for r in results
            if r["success"] and r["prediction"].lower() != r["ground_truth"].lower()
        ]
        if wrong_predictions:
            print(f"\n❌ Wrong Predictions ({len(wrong_predictions)} samples):")
            for r in wrong_predictions:
                print(f" PRED: {r['prediction']:15s} vs TRUE: {r['ground_truth']:15s} ❌")

        # Show some example raw outputs
        print("\n📝 Sample raw outputs:")
        for i, r in enumerate(results[:3]):
            if r["success"]:
                print(f"   Sample {i + 1}: '{r.get('raw_output', 'N/A')}'")

        print("\nEvaluation completed")

        return {
            "results": results,
            "total_samples": len(test_dataset),
            "successful_predictions": total_success,
            "correct_predictions": correct,
            "accuracy": accuracy if total_success > 0 else 0.0,
        }


def main() -> None:
    """Run evaluation on test set."""
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate food classifier on test set")
    parser.add_argument("--config", type=str, required=True, help="Path to model/dataset YAML")
    parser.add_argument("--finetune-config", type=str, required=True, help="Path to finetune YAML")

    args = parser.parse_args()

    # Initialize evaluator
    model_id = ModelConfig.load(args.config).model_id
    dataset_name = DatasetConfig.load(args.config).dataset_id
    settings = FinetuneSettings.load(args.finetune_config)

    # Use local path where model was downloaded from Modal
    model_name = settings.sft.output_dir.split("/")[-1]
    adapter_path = f"models/{model_name}/lora_adapter"

    print(f"Loading adapter from: {adapter_path}")

    evaluator = FoodEvaluator(
        base_model_id=model_id,
        lora_adapter_path=adapter_path,
        device="auto",
        finetune_config=args.finetune_config,
    )

    # Load balanced dataset
    split = evaluator.load_balanced_dataset(dataset_name=dataset_name)

    # Run evaluation
    evaluator.evaluate_dataset(test_dataset=split["test"], show_first=5)


if __name__ == "__main__":
    main()
