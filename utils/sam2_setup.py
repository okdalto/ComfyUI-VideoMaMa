"""
SAM2 setup and import utilities for ComfyUI-VideoMaMa
"""

import sys
import shutil
import subprocess
import importlib
from pathlib import Path
from typing import Tuple, Any, Optional

# Module-level state
SAM2_AVAILABLE = False
build_sam2_video_predictor = None
_sam2_module = None


def setup_sam2() -> bool:
    """
    Automatically download and setup SAM2 if not available.

    Returns:
        True if setup was successful, False otherwise
    """
    node_dir = Path(__file__).parent.parent
    sam2_dir = node_dir / "sam2"

    # Check if SAM2 is already properly installed
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

    # Clone SAM2
    print("SAM2 not found. Attempting to download SAM2 automatically...")
    try:
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

        # Install dependencies
        try:
            subprocess.run(
                [sys.executable, "-m", "pip", "install", "-e", str(sam2_dir)],
                check=False,
                capture_output=True
            )
            print("SAM2 dependencies installed")
        except Exception as e:
            print(f"Warning: Could not install SAM2 dependencies: {e}")
            print("SAM2 may still work if dependencies are already installed")

        # Invalidate import caches so Python can find newly installed modules
        importlib.invalidate_caches()

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


def _try_import_sam2() -> Tuple[bool, Optional[Any], Optional[Any]]:
    """
    Try to import SAM2 module.

    Returns:
        Tuple of (success, sam2_module, build_sam2_video_predictor_func)
    """
    try:
        import sam2
        from sam2.build_sam import build_sam2_video_predictor as build_func

        sam2_path = Path(sam2.__file__).parent
        sam2_configs_dir = sam2_path / "configs"

        # Check for incompatible SAM2 (e.g., from comfyui-rmbg)
        if "comfyui-rmbg" in str(sam2_path):
            print("WARNING: Detected comfyui-rmbg's SAM2, which is not compatible.")
            return False, None, None

        if not sam2_configs_dir.exists():
            print("WARNING: SAM2 installation missing configs directory.")
            print(f"Looking for configs at: {sam2_configs_dir}")
            return False, None, None

        print(f"SAM2 loaded successfully from: {sam2_path}")
        return True, sam2, build_func

    except ImportError:
        return False, None, None


def _try_import_after_setup() -> Tuple[bool, Optional[Any], Optional[Any]]:
    """
    Try to import SAM2 after fresh installation using importlib.

    Returns:
        Tuple of (success, sam2_module, build_sam2_video_predictor_func)
    """
    try:
        importlib.invalidate_caches()

        sam2 = importlib.import_module('sam2')
        build_sam_module = importlib.import_module('sam2.build_sam')
        build_func = build_sam_module.build_sam2_video_predictor

        sam2_path = Path(sam2.__file__).parent
        sam2_configs_dir = sam2_path / "configs"

        if sam2_configs_dir.exists():
            print(f"SAM2 loaded successfully from: {sam2_path}")
            return True, sam2, build_func
        else:
            print(f"SAM2 downloaded but configs not found at {sam2_configs_dir}")
            print("Please restart ComfyUI to complete SAM2 installation.")
            return False, None, None

    except ImportError as e:
        print(f"Could not import SAM2 after installation: {e}")
        print("Please restart ComfyUI to complete SAM2 installation.")
        return False, None, None


def import_sam2() -> Tuple[bool, Optional[Any]]:
    """
    Import SAM2, installing it automatically if needed.

    Returns:
        Tuple of (SAM2_AVAILABLE, build_sam2_video_predictor function or None)
    """
    global SAM2_AVAILABLE, build_sam2_video_predictor, _sam2_module

    # First try: direct import
    success, sam2_mod, build_func = _try_import_sam2()

    if success:
        SAM2_AVAILABLE = True
        _sam2_module = sam2_mod
        build_sam2_video_predictor = build_func
        return True, build_func

    # If incompatible version found, try to install official version
    print("Attempting to install official SAM2...")
    if setup_sam2():
        success, sam2_mod, build_func = _try_import_sam2()
        if success:
            SAM2_AVAILABLE = True
            _sam2_module = sam2_mod
            build_sam2_video_predictor = build_func
            return True, build_func

    # Second try: install and import with importlib
    print("SAM2 not found. Attempting to install automatically...")
    if setup_sam2():
        success, sam2_mod, build_func = _try_import_after_setup()
        if success:
            SAM2_AVAILABLE = True
            _sam2_module = sam2_mod
            build_sam2_video_predictor = build_func
            return True, build_func

    return False, None


def get_sam2_module():
    """Get the imported sam2 module."""
    return _sam2_module


# Initialize SAM2 on module load
SAM2_AVAILABLE, build_sam2_video_predictor = import_sam2()
