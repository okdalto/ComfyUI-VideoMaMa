"""
VideoMaMa ComfyUI Custom Nodes
Video matting with mask conditioning for ComfyUI

Installation:
    cd /path/to/ComfyUI/custom_nodes/
    git clone https://github.com/your-repo/VideoMaMa.git

Then restart ComfyUI and the nodes will appear under the "VideoMaMa" category.
"""

from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']
