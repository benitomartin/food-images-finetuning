import torch
from loguru import logger
from transformers import AutoModelForImageTextToText, AutoProcessor

from food_images_finetuning.schemas.model import Model


def load_vl_model(
    model_id: str,
    *,
    dtype: str | None = None,
    device_map: str | None = None,
    max_image_tokens: int | None = None,
) -> tuple[AutoModelForImageTextToText, AutoProcessor]:
    """Load the model and processor from Hugging Face.
    Args:
        model_id (str): The model identifier from Hugging Face.
    Returns:
        tuple: A tuple containing the model and processor.
    """

    resolved_dtype = dtype or "bfloat16"
    torch_dtype = torch.bfloat16 if resolved_dtype == "bfloat16" else torch.float16
    resolved_device_map = device_map or "auto"

    model = AutoModelForImageTextToText.from_pretrained(
        model_id,
        device_map=resolved_device_map,
        dtype=torch_dtype,
    )

    processor = AutoProcessor.from_pretrained(
        model_id,
        max_image_tokens=max_image_tokens or 256,
    )

    logger.info(f"Model and processor loaded from {model_id}")

    # # Log model details
    # logger.debug(f"Model config: {model.config}")
    # total_params = sum(p.numel() for p in model.parameters())
    # trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # logger.debug(f"Total parameters: {total_params / 1e6:.2f}M")
    # logger.debug(f"Trainable parameters: {trainable_params / 1e6:.2f}M")
    # logger.info(
    #     f"Percentage of trainable model parameters: {100 * trainable_params / total_params:.2f}%"
    # )

    # # # # Log processor details
    # logger.debug(f"Processor details: {processor}")
    # logger.info(f"Processor vocabulary size: {processor.tokenizer.vocab_size}")
    # logger.info(f"Processor min image tokens: {processor.image_processor.min_image_tokens}")
    # logger.info(f"Processor max image tokens: {processor.image_processor.max_image_tokens}")

    # # --- Additional logging for image tiling limits ---
    # tile_size = processor.image_processor.tile_size
    # max_tiles = processor.image_processor.max_tiles
    # downsample_factor = processor.image_processor.downsample_factor

    # if tile_size and max_tiles:
    #     # pixels per tile (e.g., 512 * 512)
    #     pixels_per_tile = tile_size * tile_size
    #     total_pixels = pixels_per_tile * max_tiles

    #     # convert to megapixels
    #     total_megapixels = total_pixels / 1_000_000

    #     # approximate max image dimension if square
    #     input_image_dim = 2048
    #     output_image_dim = int(total_pixels / input_image_dim)

    #     logger.info(f"Tile size: {tile_size}x{tile_size}px")
    #     logger.info(f"Max tiles: {max_tiles}")
    #     logger.info(f"Downsample factor: {downsample_factor}")
    #     logger.info(
    #         f"Approx. max pixel capacity: {total_pixels:,} pixels (~{total_megapixels:.2f} MP)"
    #     )
    #     logger.info(f"Example size w/o downsampling: {input_image_dim}x{output_image_dim}px")
    # else:
    #     logger.warning("Tile or max_tiles info not available in processor.")

    return model, processor  # type: ignore


if __name__ == "__main__":
    from food_images_finetuning.schemas.model import Model

    # --- Configurations ---
    model_config = Model.load()

    model_id = model_config.model_id
    logger.info(f"Model and processor loaded from {model_id}")

    model, processor = load_vl_model(model_id)
