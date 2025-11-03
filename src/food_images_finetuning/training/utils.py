import modal


def get_docker_image() -> modal.Image:
    """
    Returns a Modal Docker image with all the required Python dependencies installed.
    """
    docker_image = (
        modal.Image.debian_slim(python_version="3.12")
        .uv_pip_install(
            "bitsandbytes>=0.48.1",
            "datasets>=4.2.0",
            "loguru>=0.7.3",
            "modal>=1.2.0",
            "peft>=0.17.1",
            "pydantic>=2.12.3",
            "pyyaml>=6.0.3",
            "torch==2.8.0",
            "torchao==0.13.0",
            "torchvision>=0.23.0",
            "transformers>=4.57.1",
            "trl>=0.24.0",
        )
        .env({"HF_HOME": "/model_cache"})
        .add_local_python_source("food_images_finetuning", ignore=[])
    )

    return docker_image


def get_volume(name: str) -> modal.Volume:
    """
    Returns a Modal volume object for the given name.
    """
    return modal.Volume.from_name(name, create_if_missing=True)


def get_retries(max_retries: int) -> modal.Retries:
    """
    Returns the retry policy for failed tasks.
    """
    return modal.Retries(initial_delay=0.0, max_retries=max_retries)
