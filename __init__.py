"""
VideoMaMa ComfyUI Custom Nodes
Video matting with mask conditioning for ComfyUI

Installation:
    cd /path/to/ComfyUI/custom_nodes/
    git clone https://github.com/your-repo/VideoMaMa.git

Then restart ComfyUI and the nodes will appear under the "VideoMaMa" category.
"""

from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

# Web directory for custom frontend widgets
WEB_DIRECTORY = "./web/js"

# Register server routes for video frame extraction
try:
    import os
    import io
    import folder_paths
    from aiohttp import web
    from server import PromptServer

    @PromptServer.instance.routes.get("/videomama/video_frame")
    async def get_video_frame(request):
        """Extract first frame from a video file and return as JPEG image."""
        filename = request.query.get("filename", "")
        frame_index = int(request.query.get("frame", 0))

        if not filename:
            return web.Response(status=400, text="Missing filename parameter")

        # Find the video file in ComfyUI input directory
        input_dir = folder_paths.get_input_directory()
        video_path = os.path.join(input_dir, filename)

        if not os.path.exists(video_path):
            return web.Response(status=404, text=f"Video file not found: {filename}")

        try:
            import cv2
            import numpy as np
            from PIL import Image

            # Open video and extract frame
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                return web.Response(status=500, text="Failed to open video file")

            # Seek to requested frame
            if frame_index > 0:
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)

            ret, frame = cap.read()
            cap.release()

            if not ret or frame is None:
                return web.Response(status=500, text="Failed to read video frame")

            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Convert to PIL Image and save to bytes
            img = Image.fromarray(frame_rgb)
            img_bytes = io.BytesIO()
            img.save(img_bytes, format='JPEG', quality=90)
            img_bytes.seek(0)

            return web.Response(
                body=img_bytes.read(),
                content_type='image/jpeg',
                headers={
                    'Cache-Control': 'public, max-age=3600',
                    'Access-Control-Allow-Origin': '*'
                }
            )

        except ImportError as e:
            return web.Response(status=500, text=f"Missing dependency: {e}")
        except Exception as e:
            return web.Response(status=500, text=f"Error extracting frame: {e}")

    print("[VideoMaMa] Registered /videomama/video_frame API endpoint")

except Exception as e:
    print(f"[VideoMaMa] Failed to register server routes: {e}")

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS', 'WEB_DIRECTORY']
