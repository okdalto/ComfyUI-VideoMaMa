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
from typing import List
from comfy.utils import ProgressBar

# Import VideoMaMa components
from .pipeline_svd_mask import VideoInferencePipeline

# Import utilities
from .utils import (
    download_model,
    download_file,
    get_sam2_model_info,
    HF_HUB_AVAILABLE,
    SAM2_AVAILABLE,
    build_sam2_video_predictor,
    get_sam2_module,
)


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

    def load_pipeline(
        self,
        base_model_path: str,
        unet_checkpoint_path: str,
        precision: str,
        enable_model_cpu_offload: bool,
        vae_encode_chunk_size: int
    ):
        """Load the VideoMaMa inference pipeline"""
        weight_dtype = torch.float16 if precision == "fp16" else torch.bfloat16

        # Get absolute paths relative to this node's directory
        node_dir = Path(__file__).parent
        base_model_path = str(node_dir / base_model_path)
        unet_checkpoint_path = str(node_dir / unet_checkpoint_path)

        # Auto-download models if not exists
        base_model_path = download_model(
            repo_id="stabilityai/stable-video-diffusion-img2vid-xt",
            local_dir=base_model_path,
            description="Stable Video Diffusion base model"
        )

        unet_checkpoint_path = download_model(
            repo_id="SammyLim/VideoMaMa",
            local_dir=unet_checkpoint_path,
            description="VideoMaMa UNet checkpoint"
        )

        print(f"Loading VideoMaMa pipeline...")
        print(f"  Base model: {base_model_path}")
        print(f"  UNet checkpoint: {unet_checkpoint_path}")
        print(f"  Model CPU Offload: {enable_model_cpu_offload}")
        print(f"  VAE Encode Chunk Size: {vae_encode_chunk_size}")

        try:
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


class VideoMaMaSampler:
    """
    Runs VideoMaMa inference on video frames with mask conditioning.
    Expects ComfyUI IMAGE format: [B, H, W, C] tensors with values in [0, 1].
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "pipeline": ("VIDEOMAMA_PIPELINE",),
                "images": ("IMAGE",),
                "masks": ("MASK",),
                "seed": ("INT", {
                    "default": 42,
                    "min": 0,
                    "max": 0xffffffffffffffff
                }),
                "max_resolution": ("INT", {
                    "default": 1024,
                    "min": 256,
                    "max": 2048,
                    "step": 8
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

    RETURN_TYPES = ("MASK",)
    FUNCTION = "run_inference"
    CATEGORY = "VideoMaMa"

    @staticmethod
    def _compute_target_size(width: int, height: int, max_resolution: int):
        """Compute target size preserving aspect ratio with longest axis = max_resolution, aligned to 8."""
        if width >= height:
            new_width = max_resolution
            new_height = int(height * max_resolution / width)
        else:
            new_height = max_resolution
            new_width = int(width * max_resolution / height)
        # Align to multiple of 8
        new_width = max((new_width // 8) * 8, 8)
        new_height = max((new_height // 8) * 8, 8)
        return new_width, new_height

    def run_inference(
        self,
        pipeline,
        images,
        masks,
        seed: int,
        max_resolution: int,
        fps: int,
        motion_bucket_id: int,
        noise_aug_strength: float
    ):
        """Run VideoMaMa inference"""
        num_frames = images.shape[0]

        if masks.shape[0] != num_frames:
            raise ValueError(
                f"Number of image frames ({num_frames}) must match "
                f"number of mask frames ({masks.shape[0]})"
            )

        # Compute target resolution preserving aspect ratio
        orig_h, orig_w = images.shape[1], images.shape[2]
        target_w, target_h = self._compute_target_size(orig_w, orig_h, max_resolution)
        print(f"Input resolution: {orig_w}x{orig_h} -> Target resolution: {target_w}x{target_h} (max_resolution={max_resolution})")

        # Convert to PIL Images and resize to target resolution
        cond_frames = []
        mask_frames = []

        for i in range(num_frames):
            img_np = (images[i].cpu().numpy() * 255).astype(np.uint8)
            img_pil = Image.fromarray(img_np, mode='RGB')
            cond_frames.append(img_pil.resize((target_w, target_h), Image.LANCZOS))

            mask_np = (masks[i].cpu().numpy() * 255).astype(np.uint8)
            mask_pil = Image.fromarray(mask_np, mode='L')
            mask_frames.append(mask_pil.resize((target_w, target_h), Image.LANCZOS))

        print(f"Running VideoMaMa inference on {num_frames} frames...")

        # Create progress bar for ComfyUI (4 steps: CLIP encode, VAE encode, UNet, VAE decode)
        pbar = ProgressBar(4)

        try:
            output_frames_pil = pipeline.run(
                cond_frames=cond_frames,
                mask_frames=mask_frames,
                seed=seed,
                fps=fps,
                motion_bucket_id=motion_bucket_id,
                noise_aug_strength=noise_aug_strength,
                pbar=pbar
            )

            # Convert back to ComfyUI MASK format [B, H, W] at original resolution
            output_masks = []
            for frame_pil in output_frames_pil:
                # Resize back to original input resolution
                frame_pil = frame_pil.resize((orig_w, orig_h), Image.LANCZOS)
                frame_np = np.array(frame_pil).astype(np.float32) / 255.0
                # Convert to grayscale if RGB
                if len(frame_np.shape) == 3:
                    frame_np = frame_np.mean(axis=-1)
                output_masks.append(frame_np)

            output_tensor = torch.from_numpy(np.stack(output_masks, axis=0))
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
                "images": ("IMAGE",),
                "checkpoint_path": ("STRING", {
                    "default": "checkpoints/sam2/sam2.1_hiera_large.pt",
                    "multiline": False
                }),
                "config_name": ("STRING", {
                    "default": "configs/sam2.1/sam2.1_hiera_l.yaml",
                    "multiline": False
                }),
            },
            "hidden": {
                "points_x": ("STRING", {"default": "512"}),
                "points_y": ("STRING", {"default": "288"}),
                "labels": ("STRING", {"default": "1"}),
            }
        }

    RETURN_TYPES = ("MASK",)
    FUNCTION = "generate_masks"
    CATEGORY = "VideoMaMa"

    def _verify_sam2(self):
        """Verify SAM2 is available and properly installed."""
        if not SAM2_AVAILABLE:
            raise RuntimeError(
                "SAM2 is not available. Please install SAM2:\n"
                "git clone https://github.com/facebookresearch/sam2.git && "
                "cd sam2 && pip install -e ."
            )

        sam2 = get_sam2_module()
        if sam2 is None:
            raise RuntimeError("SAM2 module not loaded properly.")

        sam2_path = Path(sam2.__file__).parent
        sam2_configs_dir = sam2_path / "configs"

        if not sam2_configs_dir.exists():
            raise RuntimeError(
                f"SAM2 configs directory not found at {sam2_configs_dir}\n"
                f"Please restart ComfyUI to auto-install SAM2, or install manually:\n"
                f"  git clone https://github.com/facebookresearch/sam2.git\n"
                f"  cd sam2 && pip install -e ."
            )

        return sam2, sam2_path

    def _download_sam2_files(self, config_name: str, checkpoint_path: str):
        """Download SAM2 config and checkpoint files."""
        node_dir = Path(__file__).parent
        model_info = get_sam2_model_info(config_name)

        checkpoint_path_abs = str(node_dir / checkpoint_path)
        config_local_dir = node_dir / "configs" / "sam2.1"
        config_local_path = config_local_dir / model_info["config"]

        # Download config
        if not os.path.exists(config_local_path):
            download_file(
                repo_id=model_info["repo_id"],
                filename=model_info["config"],
                local_dir=str(config_local_dir),
                description="SAM2 config"
            )

        # Download checkpoint
        if not os.path.exists(checkpoint_path_abs):
            os.makedirs(os.path.dirname(checkpoint_path_abs), exist_ok=True)
            download_file(
                repo_id=model_info["repo_id"],
                filename=model_info["checkpoint"],
                local_dir=os.path.dirname(checkpoint_path_abs),
                description="SAM2 checkpoint"
            )

        return checkpoint_path_abs, config_local_path, model_info["config"]

    def _copy_config_to_sam2(self, sam2_path: Path, config_local_path: Path, config_filename: str):
        """Copy config file to SAM2 package directory if needed."""
        sam2_config_dest = sam2_path / "configs" / "sam2.1" / config_filename

        if not os.path.exists(sam2_config_dest):
            print(f"Copying config to SAM2 package: {sam2_config_dest}")
            os.makedirs(os.path.dirname(sam2_config_dest), exist_ok=True)
            shutil.copy(str(config_local_path), str(sam2_config_dest))

    def _parse_points(self, points_x: str, points_y: str, labels: str):
        """Parse point coordinates and labels from strings."""
        try:
            x_coords = [int(x.strip()) for x in points_x.split(",")]
            y_coords = [int(y.strip()) for y in points_y.split(",")]
            label_list = [int(l.strip()) for l in labels.split(",")]

            if len(x_coords) != len(y_coords) or len(x_coords) != len(label_list):
                raise ValueError("Number of x, y coordinates and labels must match")

            points = [[x, y] for x, y in zip(x_coords, y_coords)]
            return points, label_list

        except Exception as e:
            raise ValueError(f"Failed to parse points/labels: {e}")

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
        # Convert images to numpy first to get num_frames
        num_frames = images.shape[0]

        # Create progress bar (2 + num_frames steps: prepare, load model, propagate each frame)
        pbar = ProgressBar(2 + num_frames)

        # Verify SAM2 installation
        sam2, sam2_path = self._verify_sam2()

        # Download required files
        checkpoint_path_abs, config_local_path, config_filename = \
            self._download_sam2_files(config_name, checkpoint_path)

        # Copy config to SAM2 package
        self._copy_config_to_sam2(sam2_path, config_local_path, config_filename)

        # Parse points
        points, label_list = self._parse_points(points_x, points_y, labels)

        # Convert images to numpy
        frames_np = [
            (images[i].cpu().numpy() * 255).astype(np.uint8)
            for i in range(num_frames)
        ]

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

            pbar.update(1)  # Preparation complete

            # Initialize SAM2 predictor
            device = "cuda" if torch.cuda.is_available() else "cpu"
            config_relative_path = f"configs/sam2.1/{config_filename}"

            print(f"Loading SAM2 with config: {config_relative_path}")
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

            pbar.update(1)  # Model loading complete

            # Propagate through video
            masks = []
            for frame_idx, object_ids, mask_logits in predictor.propagate_in_video(inference_state):
                obj_ids_list = object_ids.tolist() if hasattr(object_ids, 'tolist') else object_ids

                if 1 in obj_ids_list:
                    mask_idx = obj_ids_list.index(1)
                    mask = (mask_logits[mask_idx] > 0.0).cpu().numpy()
                    mask_uint8 = (mask.squeeze() * 255).astype(np.uint8)
                    masks.append(mask_uint8)
                else:
                    h, w = frames_np[0].shape[:2]
                    masks.append(np.zeros((h, w), dtype=np.uint8))

                pbar.update(1)  # Frame propagation complete

            # Convert to ComfyUI MASK format
            mask_images = [mask.astype(np.float32) / 255.0 for mask in masks]
            output_tensor = torch.from_numpy(np.stack(mask_images, axis=0))

            print(f"Generated {len(masks)} masks with shape {output_tensor.shape}")
            return (output_tensor,)

        except Exception as e:
            raise RuntimeError(f"SAM2 mask generation failed: {e}")

        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)


# Node class mappings
NODE_CLASS_MAPPINGS = {
    "VideoMaMaPipelineLoader": VideoMaMaPipelineLoader,
    "VideoMaMaSampler": VideoMaMaSampler,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "VideoMaMaPipelineLoader": "VideoMaMa Pipeline Loader",
    "VideoMaMaSampler": "VideoMaMa Sampler",
}

# Add SAM2 node if available
if SAM2_AVAILABLE:
    NODE_CLASS_MAPPINGS["SAM2VideoMaskGenerator"] = SAM2VideoMaskGenerator
    NODE_DISPLAY_NAME_MAPPINGS["SAM2VideoMaskGenerator"] = "SAM2 Video Mask Generator"
