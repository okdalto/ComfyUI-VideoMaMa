"""
ComfyUI Custom Nodes for VideoMaMa
Provides video matting capabilities with mask conditioning
"""

import os
import torch
import numpy as np
from PIL import Image
import tempfile
import shutil
from pathlib import Path
from typing import List, Tuple

# Import VideoMaMa components
from .pipeline_svd_mask import VideoInferencePipeline

# For automatic model downloading
try:
    from huggingface_hub import snapshot_download
    HF_HUB_AVAILABLE = True
except ImportError:
    HF_HUB_AVAILABLE = False
    print("huggingface_hub not available. Auto-download disabled.")

# Auto-install SAM2 if not available
def setup_sam2():
    """Automatically download and setup SAM2 if not available"""
    import sys
    import subprocess

    node_dir = Path(__file__).parent
    sam2_dir = node_dir / "sam2"

    # Check if SAM2 is already properly installed
    # Configs are in sam2/sam2/configs/ (inside the package)
    sam2_configs_dir = sam2_dir / "sam2" / "configs"
    if sam2_dir.exists() and (sam2_dir / "sam2" / "__init__.py").exists() and sam2_configs_dir.exists():
        # Valid SAM2 installation, just add to path
        sam2_parent = str(sam2_dir.parent)
        if sam2_parent not in sys.path:
            sys.path.insert(0, sam2_parent)
        print(f"Using existing SAM2 installation at {sam2_dir}")
        return True

    # If directory exists but is invalid, remove it
    if sam2_dir.exists():
        print(f"Removing invalid SAM2 installation at {sam2_dir}")
        try:
            shutil.rmtree(sam2_dir)
        except Exception as e:
            print(f"Warning: Could not remove old SAM2 directory: {e}")
            return False

    # Try to clone SAM2
    print("SAM2 not found. Attempting to download SAM2 automatically...")
    try:
        # Clone SAM2 repository
        subprocess.run(
            ["git", "clone", "https://github.com/facebookresearch/sam2.git", str(sam2_dir)],
            check=True,
            capture_output=True
        )
        print(f"SAM2 cloned successfully to {sam2_dir}")

        # Add to Python path
        sam2_parent = str(sam2_dir.parent)
        if sam2_parent not in sys.path:
            sys.path.insert(0, sam2_parent)

        # Install dependencies (optional, may fail but SAM2 can still work)
        try:
            subprocess.run(
                [sys.executable, "-m", "pip", "install", "-e", str(sam2_dir)],
                check=False,  # Don't fail if pip install fails
                capture_output=True
            )
            print("SAM2 dependencies installed")
        except Exception as e:
            print(f"Warning: Could not install SAM2 dependencies: {e}")
            print("SAM2 may still work if dependencies are already installed")

        return True

    except subprocess.CalledProcessError as e:
        print(f"Failed to clone SAM2: {e}")
        print("Please manually install SAM2:")
        print("  git clone https://github.com/facebookresearch/sam2.git")
        print("  cd sam2 && pip install -e .")
        return False
    except Exception as e:
        print(f"Error setting up SAM2: {e}")
        return False

# Try to import SAM2 (with auto-install)
SAM2_AVAILABLE = False
try:
    import sam2
    from sam2.build_sam import build_sam2_video_predictor

    # Verify this is a valid SAM2 with configs directory
    sam2_path = Path(sam2.__file__).parent
    # SAM2 has configs inside the sam2 package
    sam2_configs_dir = sam2_path / "configs"

    # Check if this is comfyui-rmbg's sam2 (has no configs)
    if "comfyui-rmbg" in str(sam2_path):
        print("WARNING: Detected comfyui-rmbg's SAM2, which is not compatible.")
        print("Attempting to install official SAM2...")
        if setup_sam2():
            # Try importing again
            import sam2
            from sam2.build_sam import build_sam2_video_predictor
            sam2_path = Path(sam2.__file__).parent
            sam2_configs_dir = sam2_path / "configs"
            if sam2_configs_dir.exists():
                SAM2_AVAILABLE = True
                print(f"SAM2 loaded successfully from: {sam2_path}")
    elif not sam2_configs_dir.exists():
        print("WARNING: SAM2 installation missing configs directory.")
        print(f"Looking for configs at: {sam2_configs_dir}")
        print("Attempting to install official SAM2...")
        if setup_sam2():
            # Try importing again
            import sam2
            from sam2.build_sam import build_sam2_video_predictor
            sam2_path = Path(sam2.__file__).parent
            sam2_configs_dir = sam2_path / "configs"
            if sam2_configs_dir.exists():
                SAM2_AVAILABLE = True
                print(f"SAM2 loaded successfully from: {sam2_path}")
    else:
        SAM2_AVAILABLE = True
        print(f"SAM2 loaded successfully from: {sam2_path}")

except ImportError:
    print("SAM2 not found. Attempting to install automatically...")
    if setup_sam2():
        # Try importing again after setup
        try:
            import sam2
            from sam2.build_sam import build_sam2_video_predictor
            sam2_path = Path(sam2.__file__).parent
            sam2_configs_dir = sam2_path / "configs"
            if sam2_configs_dir.exists():
                SAM2_AVAILABLE = True
                print(f"SAM2 loaded successfully from: {sam2_path}")
            else:
                print(f"SAM2 downloaded but configs not found at {sam2_configs_dir}")
        except ImportError as e:
            print(f"Could not import SAM2 after installation: {e}")


class VideoMaMaPipelineLoader:
    """
    Loads the VideoMaMa pipeline with base model and fine-tuned UNet checkpoint.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "base_model_path": ("STRING", {
                    "default": "checkpoints/stabilityai/stable-video-diffusion-img2vid-xt",
                    "multiline": False
                }),
                "unet_checkpoint_path": ("STRING", {
                    "default": "checkpoints/VideoMaMa",
                    "multiline": False
                }),
                "precision": (["fp16", "bf16"], {
                    "default": "fp16"
                }),
                "enable_model_cpu_offload": ("BOOLEAN", {
                    "default": True
                }),
                "vae_encode_chunk_size": ("INT", {
                    "default": 4,
                    "min": 1,
                    "max": 25,
                    "step": 1
                }),
            }
        }

    RETURN_TYPES = ("VIDEOMAMA_PIPELINE",)
    FUNCTION = "load_pipeline"
    CATEGORY = "VideoMaMa"

    def load_pipeline(self, base_model_path: str, unet_checkpoint_path: str, precision: str,
                      enable_model_cpu_offload: bool, vae_encode_chunk_size: int):
        """Load the VideoMaMa inference pipeline"""

        weight_dtype = torch.float16 if precision == "fp16" else torch.bfloat16

        # Get the absolute path relative to this node's directory
        node_dir = Path(__file__).parent
        base_model_path = str(node_dir / base_model_path)
        unet_checkpoint_path = str(node_dir / unet_checkpoint_path)

        # Auto-download base model if not exists
        if not os.path.exists(base_model_path):
            print(f"Base model not found at {base_model_path}")
            if HF_HUB_AVAILABLE:
                print("Downloading Stable Video Diffusion model from Hugging Face...")
                print("This may take several minutes (model size: ~20GB)...")
                try:
                    base_model_path = snapshot_download(
                        repo_id="stabilityai/stable-video-diffusion-img2vid-xt",
                        local_dir=base_model_path,
                        local_dir_use_symlinks=False,
                    )
                    print(f"Base model downloaded successfully to {base_model_path}")
                except Exception as e:
                    raise RuntimeError(
                        f"Failed to download base model: {e}\n"
                        f"Please download manually from: "
                        f"https://huggingface.co/stabilityai/stable-video-diffusion-img2vid-xt"
                    )
            else:
                raise RuntimeError(
                    f"Base model not found at {base_model_path}\n"
                    f"Please install huggingface_hub: pip install huggingface_hub\n"
                    f"Or download manually from: "
                    f"https://huggingface.co/stabilityai/stable-video-diffusion-img2vid-xt"
                )

        # Auto-download UNet checkpoint if not exists
        if not os.path.exists(unet_checkpoint_path):
            print(f"VideoMaMa UNet checkpoint not found at {unet_checkpoint_path}")
            if HF_HUB_AVAILABLE:
                print("Downloading VideoMaMa UNet checkpoint from Hugging Face...")
                print("This may take several minutes (model size: ~3GB)...")
                try:
                    unet_checkpoint_path = snapshot_download(
                        repo_id="SammyLim/VideoMaMa",
                        local_dir=unet_checkpoint_path,
                        local_dir_use_symlinks=False,
                    )
                    print(f"VideoMaMa checkpoint downloaded successfully to {unet_checkpoint_path}")
                except Exception as e:
                    raise RuntimeError(
                        f"Failed to download VideoMaMa checkpoint: {e}\n"
                        f"Please download manually from: "
                        f"https://huggingface.co/SammyLim/VideoMaMa"
                    )
            else:
                raise RuntimeError(
                    f"VideoMaMa UNet checkpoint not found at {unet_checkpoint_path}\n"
                    f"Please install huggingface_hub: pip install huggingface_hub\n"
                    f"Or download manually from: "
                    f"https://huggingface.co/SammyLim/VideoMaMa"
                )

        try:
            print(f"Loading VideoMaMa pipeline...")
            print(f"  Base model: {base_model_path}")
            print(f"  UNet checkpoint: {unet_checkpoint_path}")
            print(f"  Model CPU Offload: {enable_model_cpu_offload}")
            print(f"  VAE Encode Chunk Size: {vae_encode_chunk_size}")

            pipeline = VideoInferencePipeline(
                base_model_path=base_model_path,
                unet_checkpoint_path=unet_checkpoint_path,
                weight_dtype=weight_dtype,
                device="cuda" if torch.cuda.is_available() else "cpu",
                enable_model_cpu_offload=enable_model_cpu_offload,
                vae_encode_chunk_size=vae_encode_chunk_size
            )

            print(f"VideoMaMa pipeline loaded successfully with {precision} precision")
            return (pipeline,)

        except Exception as e:
            raise RuntimeError(f"Failed to load VideoMaMa pipeline: {e}")


class VideoMaMaRun:
    """
    Runs VideoMaMa inference on video frames with mask conditioning.
    Expects ComfyUI IMAGE format: [B, H, W, C] tensors with values in [0, 1].
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "pipeline": ("VIDEOMAMA_PIPELINE",),
                "images": ("IMAGE",),  # ComfyUI format: [N, H, W, C] float32 in [0,1]
                "masks": ("IMAGE",),   # ComfyUI format: [N, H, W, C] float32 in [0,1]
                "seed": ("INT", {
                    "default": 42,
                    "min": 0,
                    "max": 0xffffffffffffffff
                }),
                "mask_cond_mode": (["vae", "interpolate"], {
                    "default": "vae"
                }),
                "fps": ("INT", {
                    "default": 7,
                    "min": 1,
                    "max": 60
                }),
                "motion_bucket_id": ("INT", {
                    "default": 127,
                    "min": 1,
                    "max": 255
                }),
                "noise_aug_strength": ("FLOAT", {
                    "default": 0.0,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01
                }),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "run_inference"
    CATEGORY = "VideoMaMa"

    def run_inference(
        self,
        pipeline,
        images,
        masks,
        seed: int,
        mask_cond_mode: str,
        fps: int,
        motion_bucket_id: int,
        noise_aug_strength: float
    ):
        """Run VideoMaMa inference"""

        # Convert ComfyUI IMAGE format to PIL Images
        # ComfyUI: [N, H, W, C] float32 in [0,1]
        # PIL: needs uint8 RGB

        num_frames = images.shape[0]

        # Check that images and masks have same number of frames
        if masks.shape[0] != num_frames:
            raise ValueError(
                f"Number of image frames ({num_frames}) must match number of mask frames ({masks.shape[0]})"
            )

        # Convert to PIL Images
        cond_frames = []
        mask_frames = []

        for i in range(num_frames):
            # Convert image: [H, W, C] -> PIL RGB
            img_np = (images[i].cpu().numpy() * 255).astype(np.uint8)
            cond_frames.append(Image.fromarray(img_np, mode='RGB'))

            # Convert mask: [H, W, C] -> PIL L (grayscale)
            # Take the first channel or average if multi-channel
            mask_np = (masks[i].cpu().numpy() * 255).astype(np.uint8)
            if mask_np.shape[-1] == 3:
                mask_np = mask_np.mean(axis=-1).astype(np.uint8)
            elif mask_np.shape[-1] == 1:
                mask_np = mask_np.squeeze(-1)
            mask_frames.append(Image.fromarray(mask_np, mode='L'))

        print(f"Running VideoMaMa inference on {num_frames} frames...")

        # Run inference
        try:
            output_frames_pil = pipeline.run(
                cond_frames=cond_frames,
                mask_frames=mask_frames,
                seed=seed,
                mask_cond_mode=mask_cond_mode,
                fps=fps,
                motion_bucket_id=motion_bucket_id,
                noise_aug_strength=noise_aug_strength
            )

            # Convert back to ComfyUI IMAGE format: [N, H, W, C] float32 in [0,1]
            output_images = []
            for frame_pil in output_frames_pil:
                frame_np = np.array(frame_pil).astype(np.float32) / 255.0
                # Ensure 3 channels
                if len(frame_np.shape) == 2:
                    frame_np = np.stack([frame_np] * 3, axis=-1)
                output_images.append(frame_np)

            output_tensor = torch.from_numpy(np.stack(output_images, axis=0))

            print(f"VideoMaMa inference completed: {output_tensor.shape}")
            return (output_tensor,)

        except Exception as e:
            raise RuntimeError(f"VideoMaMa inference failed: {e}")


class SAM2VideoMaskGenerator:
    """
    Generates video masks using SAM2 (Segment Anything 2) video tracking.
    Takes a video and point prompts on the first frame to generate masks for all frames.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),  # ComfyUI format: [N, H, W, C] float32 in [0,1]
                "checkpoint_path": ("STRING", {
                    "default": "checkpoints/sam2/sam2.1_hiera_large.pt",
                    "multiline": False
                }),
                "config_name": ("STRING", {
                    "default": "configs/sam2.1/sam2.1_hiera_l.yaml",
                    "multiline": False
                }),
                "points_x": ("STRING", {
                    "default": "512",
                    "multiline": False
                }),
                "points_y": ("STRING", {
                    "default": "288",
                    "multiline": False
                }),
                "labels": ("STRING", {
                    "default": "1",
                    "multiline": False
                }),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "generate_masks"
    CATEGORY = "VideoMaMa"

    def generate_masks(
        self,
        images,
        checkpoint_path: str,
        config_name: str,
        points_x: str,
        points_y: str,
        labels: str
    ):
        """Generate video masks using SAM2"""

        if not SAM2_AVAILABLE:
            raise RuntimeError(
                "SAM2 is not available. Please install SAM2:\n"
                "git clone https://github.com/facebookresearch/sam2.git && cd sam2 && pip install -e ."
            )

        # Check if we're using the official SAM2 package (not comfyui-rmbg's version)
        try:
            import sam2
            sam2_path = Path(sam2.__file__).parent
            # SAM2 uses sam2_configs directory (not configs)
            sam2_configs_dir = sam2_path / "configs"

            if not sam2_configs_dir.exists():
                raise RuntimeError(
                    f"SAM2 configs directory not found at {sam2_configs_dir}\n"
                    f"It appears you're using a SAM2 version without configs (possibly from comfyui-rmbg).\n"
                    f"Please restart ComfyUI to auto-install SAM2, or install manually:\n"
                    f"  git clone https://github.com/facebookresearch/sam2.git\n"
                    f"  cd sam2\n"
                    f"  pip install -e .\n"
                    f"This will ensure the configs directory is properly installed."
                )
        except Exception as e:
            print(f"Warning: Could not verify SAM2 installation: {e}")

        # Get the absolute path relative to this node's directory for checkpoint
        node_dir = Path(__file__).parent
        checkpoint_path_abs = str(node_dir / checkpoint_path)

        # Determine model variant from config_name
        if "large" in config_name.lower() or "hiera_l" in config_name.lower():
            repo_id = "facebook/sam2.1-hiera-large"
            checkpoint_filename = "sam2.1_hiera_large.pt"
            config_filename = "sam2.1_hiera_l.yaml"
        elif "base_plus" in config_name.lower() or "hiera_b+" in config_name.lower():
            repo_id = "facebook/sam2.1-hiera-base-plus"
            checkpoint_filename = "sam2.1_hiera_base_plus.pt"
            config_filename = "sam2.1_hiera_b+.yaml"
        elif "small" in config_name.lower() or "hiera_s" in config_name.lower():
            repo_id = "facebook/sam2.1-hiera-small"
            checkpoint_filename = "sam2.1_hiera_small.pt"
            config_filename = "sam2.1_hiera_s.yaml"
        elif "tiny" in config_name.lower() or "hiera_t" in config_name.lower():
            repo_id = "facebook/sam2.1-hiera-tiny"
            checkpoint_filename = "sam2.1_hiera_tiny.pt"
            config_filename = "sam2.1_hiera_t.yaml"
        else:
            # Default to large
            repo_id = "facebook/sam2.1-hiera-large"
            checkpoint_filename = "sam2.1_hiera_large.pt"
            config_filename = "sam2.1_hiera_l.yaml"

        # Download config file to local configs directory
        config_local_dir = node_dir / "configs" / "sam2.1"
        config_local_path = config_local_dir / config_filename

        if not os.path.exists(config_local_path):
            print(f"SAM2 config not found at {config_local_path}")
            if HF_HUB_AVAILABLE:
                print("Downloading SAM2 config from Hugging Face...")
                try:
                    from huggingface_hub import hf_hub_download

                    os.makedirs(config_local_dir, exist_ok=True)

                    downloaded_config = hf_hub_download(
                        repo_id=repo_id,
                        filename=config_filename,
                        local_dir=config_local_dir,
                        local_dir_use_symlinks=False,
                    )

                    print(f"SAM2 config downloaded to {config_local_path}")
                except Exception as e:
                    raise RuntimeError(
                        f"Failed to download SAM2 config: {e}\n"
                        f"Please download manually from: https://huggingface.co/{repo_id}"
                    )

        # Auto-download SAM2 checkpoint if not exists
        if not os.path.exists(checkpoint_path_abs):
            print(f"SAM2 checkpoint not found at {checkpoint_path_abs}")
            if HF_HUB_AVAILABLE:
                print("Downloading SAM2 checkpoint from Hugging Face...")
                print("This may take a few minutes (model size: ~900MB)...")
                try:
                    from huggingface_hub import hf_hub_download

                    # Create directory if it doesn't exist
                    os.makedirs(os.path.dirname(checkpoint_path_abs), exist_ok=True)

                    # Download the checkpoint
                    downloaded_path = hf_hub_download(
                        repo_id=repo_id,
                        filename=checkpoint_filename,
                        local_dir=os.path.dirname(checkpoint_path_abs),
                        local_dir_use_symlinks=False,
                    )

                    # If the downloaded filename doesn't match expected, rename it
                    if downloaded_path != checkpoint_path_abs:
                        shutil.move(downloaded_path, checkpoint_path_abs)

                    print(f"SAM2 checkpoint downloaded successfully to {checkpoint_path_abs}")
                except Exception as e:
                    raise RuntimeError(
                        f"Failed to download SAM2 checkpoint: {e}\n"
                        f"Please download manually from: https://huggingface.co/{repo_id}"
                    )
            else:
                raise RuntimeError(
                    f"SAM2 checkpoint not found at {checkpoint_path_abs}\n"
                    f"Please install huggingface_hub: pip install huggingface_hub\n"
                    f"Or download manually from: https://huggingface.co/{repo_id}"
                )

        # Parse point coordinates and labels
        try:
            x_coords = [int(x.strip()) for x in points_x.split(",")]
            y_coords = [int(y.strip()) for y in points_y.split(",")]
            label_list = [int(l.strip()) for l in labels.split(",")]

            if len(x_coords) != len(y_coords) or len(x_coords) != len(label_list):
                raise ValueError("Number of x, y coordinates and labels must match")

            points = [[x, y] for x, y in zip(x_coords, y_coords)]

        except Exception as e:
            raise ValueError(f"Failed to parse points/labels: {e}")

        # Convert ComfyUI IMAGE format to numpy arrays
        num_frames = images.shape[0]
        frames_np = []

        for i in range(num_frames):
            # Convert [H, W, C] float32 [0,1] -> uint8 RGB
            frame_np = (images[i].cpu().numpy() * 255).astype(np.uint8)
            frames_np.append(frame_np)

        print(f"Generating masks for {num_frames} frames using SAM2...")
        print(f"Points: {points}, Labels: {label_list}")

        # Create temporary directory for frames
        temp_dir = Path(tempfile.mkdtemp())
        frames_dir = temp_dir / "frames"
        frames_dir.mkdir(exist_ok=True)

        try:
            # Save frames to temp directory
            for i, frame in enumerate(frames_np):
                frame_path = frames_dir / f"{i:05d}.jpg"
                Image.fromarray(frame).save(frame_path, quality=95)

            # Copy config to SAM2 package's config directory
            # This ensures Hydra can find it when build_sam2_video_predictor is called
            # SAM2 has configs inside the package (sam2/configs/)
            try:
                import sam2

                # Get the actual sam2 package location
                sam2_module_path = Path(sam2.__file__).parent
                print(f"SAM2 module location: {sam2_module_path}")

                # SAM2 has configs inside the package directory
                sam2_config_dest = sam2_module_path / "configs" / "sam2.1" / config_filename
                print(f"Target config location: {sam2_config_dest}")
                print(f"Local config location: {config_local_path}")
                print(f"Config exists at target: {os.path.exists(sam2_config_dest)}")
                print(f"Config exists locally: {os.path.exists(config_local_path)}")

                if not os.path.exists(sam2_config_dest):
                    print(f"Copying config from {config_local_path} to {sam2_config_dest}")
                    os.makedirs(os.path.dirname(sam2_config_dest), exist_ok=True)
                    shutil.copy(str(config_local_path), str(sam2_config_dest))
                    print("Config copied successfully")
                else:
                    print("Config already exists in SAM2 package")
            except Exception as e:
                print(f"Warning: Could not copy config to SAM2 package: {e}")
                import traceback
                traceback.print_exc()
                print("Attempting to use local config...")

            # Initialize SAM2 video predictor
            device = "cuda" if torch.cuda.is_available() else "cpu"
            print(f"Loading SAM2 with config: {config_name}, checkpoint: {checkpoint_path_abs}")

            # Use relative config path for Hydra
            # Hydra expects a path relative to the SAM2 package's config search path
            # The config should already be copied to sam2/configs/sam2.1/ directory
            config_relative_path = f"configs/sam2.1/{config_filename}"

            print(f"Using relative config path: {config_relative_path}")

            # Verify config exists in SAM2 package
            import sam2
            sam2_module_path = Path(sam2.__file__).parent
            config_absolute_path = sam2_module_path / "configs" / "sam2.1" / config_filename
            print(f"Config absolute path: {config_absolute_path}")
            print(f"Config exists: {os.path.exists(config_absolute_path)}")

            predictor = build_sam2_video_predictor(
                config_file=config_relative_path,
                ckpt_path=checkpoint_path_abs,
                device=device
            )

            inference_state = predictor.init_state(video_path=str(frames_dir))

            # Add prompts on first frame
            points_array = np.array(points, dtype=np.float32)
            labels_array = np.array(label_list, dtype=np.int32)

            _, out_obj_ids, out_mask_logits = predictor.add_new_points(
                inference_state=inference_state,
                frame_idx=0,
                obj_id=1,
                points=points_array,
                labels=labels_array,
            )

            # Propagate through video
            masks = []
            for frame_idx, object_ids, mask_logits in predictor.propagate_in_video(inference_state):
                # Get mask for object ID 1
                obj_ids_list = object_ids.tolist() if hasattr(object_ids, 'tolist') else object_ids

                if 1 in obj_ids_list:
                    mask_idx = obj_ids_list.index(1)
                    mask = (mask_logits[mask_idx] > 0.0).cpu().numpy()
                    mask_uint8 = (mask.squeeze() * 255).astype(np.uint8)
                    masks.append(mask_uint8)
                else:
                    # No mask for this frame, use empty mask
                    h, w = frames_np[0].shape[:2]
                    masks.append(np.zeros((h, w), dtype=np.uint8))

            # Convert masks to ComfyUI IMAGE format: [N, H, W, C] float32 in [0,1]
            mask_images = []
            for mask in masks:
                # Convert grayscale mask to 3-channel
                mask_float = mask.astype(np.float32) / 255.0
                mask_rgb = np.stack([mask_float] * 3, axis=-1)
                mask_images.append(mask_rgb)

            output_tensor = torch.from_numpy(np.stack(mask_images, axis=0))

            print(f"Generated {len(masks)} masks with shape {output_tensor.shape}")
            return (output_tensor,)

        except Exception as e:
            raise RuntimeError(f"SAM2 mask generation failed: {e}")

        finally:
            # Clean up temporary directory
            shutil.rmtree(temp_dir, ignore_errors=True)


# Node class mappings
NODE_CLASS_MAPPINGS = {
    "VideoMaMaPipelineLoader": VideoMaMaPipelineLoader,
    "VideoMaMaRun": VideoMaMaRun,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "VideoMaMaPipelineLoader": "VideoMaMa Pipeline Loader",
    "VideoMaMaRun": "VideoMaMa Run",
}

# Add SAM2 node if available
if SAM2_AVAILABLE:
    NODE_CLASS_MAPPINGS["SAM2VideoMaskGenerator"] = SAM2VideoMaskGenerator
    NODE_DISPLAY_NAME_MAPPINGS["SAM2VideoMaskGenerator"] = "SAM2 Video Mask Generator"
