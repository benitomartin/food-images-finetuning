import datasets
from loguru import logger


def load_dataset(
    name: str = "benitomartin/styles_with_images_384_512",
    split: str = "train",
    n_samples: int | None = 75750,
    seed: int = 42,
) -> datasets.Dataset:
    """Load dataset from Hugging Face.

    Args:
        name: Dataset identifier.
        split: Dataset split to load.
        n_samples: Number of samples to select. If None, loads all samples.
        seed: Random seed for shuffling.

    Returns:
        Dataset: Loaded dataset.
    """
    logger.info(f"Loading dataset '{name}' with split '{split}'")
    # Load the dataset from Hugging Face
    # Disable progress bars to avoid Jupyter context issues
    dataset = datasets.load_dataset(
        path=name,
        split=split,
        # download_mode=datasets.DownloadMode.REUSE_DATASET_IF_EXISTS,
    ).shuffle(seed=seed)
    logger.info(f"Dataset loaded with {len(dataset)} samples.")

    # Optionally limit the number of samples
    if n_samples is not None and n_samples < len(dataset):
        dataset = dataset.select(range(n_samples))
        logger.info(f"Dataset truncated to {n_samples} samples.")
        logger.info(f"Length of the dataset: {len(dataset)}")

    return dataset


if __name__ == "__main__":
    from food_images_finetuning.schemas.dataset import Dataset

    dataset_config = Dataset()

    dataset_id = dataset_config.dataset_id
    split = dataset_config.split
    n_samples = dataset_config.n_samples

    dataset_config = Dataset(dataset_id=dataset_id, split=split, n_samples=n_samples)
    logger.info(f"Dataset configuration: {dataset_config}")

    # Example usage

    dataset = load_dataset(
        name=dataset_config.dataset_id,
        split=dataset_config.split,
        n_samples=dataset_config.n_samples,
    )
    logger.info(f"First sample: {dataset[0]}")
