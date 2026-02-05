"""
Model download utilities for ComfyUI-VideoMaMa
"""

import os
from pathlib import Path
from typing import Optional

# Check huggingface_hub availability
try:
    from huggingface_hub import snapshot_download, hf_hub_download
    HF_HUB_AVAILABLE = True
except ImportError:
    HF_HUB_AVAILABLE = False
    print("huggingface_hub not available. Auto-download disabled.")


def download_model(
    repo_id: str,
    local_dir: str,
    description: str = "model"
) -> str:
    """
    Download a model from Hugging Face Hub.

    Args:
        repo_id: Hugging Face repository ID (e.g., "stabilityai/stable-video-diffusion-img2vid-xt")
        local_dir: Local directory to save the model
        description: Human-readable description for logging

    Returns:
        Path to the downloaded model

    Raises:
        RuntimeError: If download fails or huggingface_hub is not available
    """
    if os.path.exists(local_dir):
        return local_dir

    print(f"{description} not found at {local_dir}")

    if not HF_HUB_AVAILABLE:
        raise RuntimeError(
            f"{description} not found at {local_dir}\n"
            f"Please install huggingface_hub: pip install huggingface_hub\n"
            f"Or download manually from: https://huggingface.co/{repo_id}"
        )

    print(f"Downloading {description} from Hugging Face...")
    try:
        downloaded_path = snapshot_download(
            repo_id=repo_id,
            local_dir=local_dir,
            local_dir_use_symlinks=False,
        )
        print(f"{description} downloaded successfully to {downloaded_path}")
        return downloaded_path
    except Exception as e:
        raise RuntimeError(
            f"Failed to download {description}: {e}\n"
            f"Please download manually from: https://huggingface.co/{repo_id}"
        )


def download_file(
    repo_id: str,
    filename: str,
    local_dir: str,
    local_filename: Optional[str] = None,
    description: str = "file"
) -> str:
    """
    Download a single file from Hugging Face Hub.

    Args:
        repo_id: Hugging Face repository ID
        filename: Filename in the repository
        local_dir: Local directory to save the file
        local_filename: Optional different filename for local storage
        description: Human-readable description for logging

    Returns:
        Path to the downloaded file

    Raises:
        RuntimeError: If download fails or huggingface_hub is not available
    """
    local_filename = local_filename or filename
    local_path = os.path.join(local_dir, local_filename)

    if os.path.exists(local_path):
        return local_path

    print(f"{description} not found at {local_path}")

    if not HF_HUB_AVAILABLE:
        raise RuntimeError(
            f"{description} not found at {local_path}\n"
            f"Please install huggingface_hub: pip install huggingface_hub\n"
            f"Or download manually from: https://huggingface.co/{repo_id}"
        )

    print(f"Downloading {description} from Hugging Face...")
    try:
        os.makedirs(local_dir, exist_ok=True)
        downloaded_path = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            local_dir=local_dir,
            local_dir_use_symlinks=False,
        )
        print(f"{description} downloaded successfully to {local_path}")
        return local_path
    except Exception as e:
        raise RuntimeError(
            f"Failed to download {description}: {e}\n"
            f"Please download manually from: https://huggingface.co/{repo_id}"
        )


# SAM2 model configurations
SAM2_MODELS = {
    "large": {
        "repo_id": "facebook/sam2.1-hiera-large",
        "checkpoint": "sam2.1_hiera_large.pt",
        "config": "sam2.1_hiera_l.yaml",
    },
    "base_plus": {
        "repo_id": "facebook/sam2.1-hiera-base-plus",
        "checkpoint": "sam2.1_hiera_base_plus.pt",
        "config": "sam2.1_hiera_b+.yaml",
    },
    "small": {
        "repo_id": "facebook/sam2.1-hiera-small",
        "checkpoint": "sam2.1_hiera_small.pt",
        "config": "sam2.1_hiera_s.yaml",
    },
    "tiny": {
        "repo_id": "facebook/sam2.1-hiera-tiny",
        "checkpoint": "sam2.1_hiera_tiny.pt",
        "config": "sam2.1_hiera_t.yaml",
    },
}


def get_sam2_model_info(config_name: str) -> dict:
    """
    Get SAM2 model information based on config name.

    Args:
        config_name: Config file name or path containing model variant info

    Returns:
        Dictionary with repo_id, checkpoint, and config filenames
    """
    config_lower = config_name.lower()

    if "large" in config_lower or "hiera_l" in config_lower:
        return SAM2_MODELS["large"]
    elif "base_plus" in config_lower or "hiera_b+" in config_lower:
        return SAM2_MODELS["base_plus"]
    elif "small" in config_lower or "hiera_s" in config_lower:
        return SAM2_MODELS["small"]
    elif "tiny" in config_lower or "hiera_t" in config_lower:
        return SAM2_MODELS["tiny"]
    else:
        # Default to large
        return SAM2_MODELS["large"]
