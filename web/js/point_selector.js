import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

/**
 * SAM2 Point Selector Extension for ComfyUI
 * Provides a visual interface for selecting positive/negative points on video frames
 */

class SAM2PointEditor {
    constructor(node) {
        this.node = node;

        this.positivePoints = [];
        this.negativePoints = [];
        this.isPositiveMode = true;
        this.image = null;
        this.dialog = null;
        this.canvas = null;
        this.ctx = null;
        this.scale = 1;
        this.imageWidth = 0;
        this.imageHeight = 0;
    }

    async open() {
        // Parse existing points from node's stored data
        this.parseExistingPoints();

        // Load image first to get dimensions
        await this.loadImageFromNode();

        // Create dialog with proper size
        this.createDialog();

        // Draw initial state
        this.draw();
    }

    parseExistingPoints() {
        try {
            // Get stored point data from node
            const pointData = this.node.sam2PointData || {
                points_x: "512",
                points_y: "288",
                labels: "1"
            };

            const xCoords = pointData.points_x.split(",").map(x => parseInt(x.trim())).filter(x => !isNaN(x));
            const yCoords = pointData.points_y.split(",").map(y => parseInt(y.trim())).filter(y => !isNaN(y));
            const labels = pointData.labels.split(",").map(l => parseInt(l.trim())).filter(l => !isNaN(l));

            this.positivePoints = [];
            this.negativePoints = [];

            for (let i = 0; i < Math.min(xCoords.length, yCoords.length, labels.length); i++) {
                if (labels[i] === 1) {
                    this.positivePoints.push({ x: xCoords[i], y: yCoords[i] });
                } else {
                    this.negativePoints.push({ x: xCoords[i], y: yCoords[i] });
                }
            }
        } catch (e) {
            console.log("Could not parse existing points:", e);
            this.positivePoints = [];
            this.negativePoints = [];
        }
    }

    async loadImageFromNode() {
        // Method 1: Try to get image from connected node's preview
        const imageInput = this.node.inputs?.find(i => i.name === "images");

        if (imageInput && imageInput.link !== null) {
            const linkInfo = app.graph.links[imageInput.link];
            if (linkInfo) {
                const sourceNode = app.graph.getNodeById(linkInfo.origin_id);

                // Check if source node has preview images
                if (sourceNode && sourceNode.imgs && sourceNode.imgs.length > 0) {
                    return new Promise((resolve) => {
                        this.image = new Image();
                        this.image.crossOrigin = "anonymous";
                        this.image.onload = () => {
                            this.imageWidth = this.image.naturalWidth;
                            this.imageHeight = this.image.naturalHeight;
                            console.log("Loaded image from node preview:", this.imageWidth, "x", this.imageHeight);
                            resolve();
                        };
                        this.image.onerror = () => {
                            console.log("Failed to load from node preview");
                            this.image = null;
                            resolve();
                        };
                        this.image.src = sourceNode.imgs[0].src;
                    });
                }

                // Method 2: Try to get image from widget (for LoadImage node)
                if (sourceNode && sourceNode.widgets) {
                    const imageWidget = sourceNode.widgets.find(w => w.name === "image");
                    if (imageWidget && imageWidget.value) {
                        const imageName = imageWidget.value;
                        return this.loadImageFromServer(imageName);
                    }
                }
            }
        }

        // Method 3: If this node has been executed, check for cached images
        if (this.node.imgs && this.node.imgs.length > 0) {
            return new Promise((resolve) => {
                this.image = new Image();
                this.image.crossOrigin = "anonymous";
                this.image.onload = () => {
                    this.imageWidth = this.image.naturalWidth;
                    this.imageHeight = this.image.naturalHeight;
                    resolve();
                };
                this.image.src = this.node.imgs[0].src;
            });
        }

        // Fallback: Set default dimensions
        this.imageWidth = 1024;
        this.imageHeight = 576;
        console.log("No image found, using default dimensions");
    }

    async loadImageFromServer(imageName) {
        return new Promise((resolve) => {
            this.image = new Image();
            this.image.crossOrigin = "anonymous";
            this.image.onload = () => {
                this.imageWidth = this.image.naturalWidth;
                this.imageHeight = this.image.naturalHeight;
                console.log("Loaded image from server:", this.imageWidth, "x", this.imageHeight);
                resolve();
            };
            this.image.onerror = () => {
                console.log("Failed to load image from server");
                this.image = null;
                this.imageWidth = 1024;
                this.imageHeight = 576;
                resolve();
            };
            // ComfyUI serves uploaded images from /view endpoint
            this.image.src = `/view?filename=${encodeURIComponent(imageName)}&type=input`;
        });
    }

    createDialog() {
        // Remove existing dialog if any
        const existingOverlay = document.getElementById("sam2-point-editor-overlay");
        if (existingOverlay) {
            existingOverlay.remove();
        }

        // Calculate canvas size to fit viewport while maintaining aspect ratio
        const maxWidth = window.innerWidth * 0.85;
        const maxHeight = window.innerHeight * 0.75;

        this.scale = Math.min(
            maxWidth / this.imageWidth,
            maxHeight / this.imageHeight,
            1.5  // Allow slight upscale for small images
        );

        const canvasWidth = Math.round(this.imageWidth * this.scale);
        const canvasHeight = Math.round(this.imageHeight * this.scale);

        // Create overlay
        const overlay = document.createElement("div");
        overlay.id = "sam2-point-editor-overlay";
        overlay.style.cssText = `
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background: rgba(0, 0, 0, 0.85);
            z-index: 10000;
            display: flex;
            justify-content: center;
            align-items: center;
        `;

        // Create dialog container - size based on image
        this.dialog = document.createElement("div");
        this.dialog.id = "sam2-point-editor-dialog";
        this.dialog.style.cssText = `
            background: #2a2a2a;
            border-radius: 8px;
            padding: 20px;
            display: flex;
            flex-direction: column;
            gap: 12px;
            box-shadow: 0 4px 20px rgba(0, 0, 0, 0.5);
        `;

        // Title
        const title = document.createElement("h3");
        title.textContent = "SAM2 Point Selector";
        title.style.cssText = `
            margin: 0;
            color: #fff;
            font-size: 18px;
        `;

        // Instructions
        const instructions = document.createElement("div");
        instructions.style.cssText = `
            color: #aaa;
            font-size: 12px;
            line-height: 1.4;
        `;
        instructions.innerHTML = `
            <b>Left click:</b> Add positive point (+) &nbsp;|&nbsp;
            <b>Right click:</b> Add negative point (-) &nbsp;|&nbsp;
            <b>Middle click / Ctrl+click:</b> Remove point
        `;

        // Mode indicator
        this.modeIndicator = document.createElement("div");
        this.updateModeIndicator();

        // Canvas container with exact size
        const canvasContainer = document.createElement("div");
        canvasContainer.style.cssText = `
            position: relative;
            width: ${canvasWidth}px;
            height: ${canvasHeight}px;
            border: 2px solid #444;
            border-radius: 4px;
            overflow: hidden;
        `;

        // Canvas - exact size matching image
        this.canvas = document.createElement("canvas");
        this.canvas.width = canvasWidth;
        this.canvas.height = canvasHeight;
        this.canvas.style.cssText = `
            display: block;
            cursor: crosshair;
            width: ${canvasWidth}px;
            height: ${canvasHeight}px;
        `;
        canvasContainer.appendChild(this.canvas);
        this.ctx = this.canvas.getContext("2d");

        // Points info
        this.pointsInfo = document.createElement("div");
        this.pointsInfo.style.cssText = `
            color: #ccc;
            font-size: 13px;
        `;
        this.updatePointsInfo();

        // Buttons
        const buttonContainer = document.createElement("div");
        buttonContainer.style.cssText = `
            display: flex;
            gap: 10px;
            justify-content: flex-end;
        `;

        const clearBtn = this.createButton("Clear All", "#666", () => {
            this.positivePoints = [];
            this.negativePoints = [];
            this.draw();
            this.updatePointsInfo();
        });

        const cancelBtn = this.createButton("Cancel", "#666", () => {
            this.close();
        });

        const saveBtn = this.createButton("Save", "#4a9eff", () => {
            this.save();
            this.close();
        });

        buttonContainer.appendChild(clearBtn);
        buttonContainer.appendChild(cancelBtn);
        buttonContainer.appendChild(saveBtn);

        // Assemble dialog
        this.dialog.appendChild(title);
        this.dialog.appendChild(instructions);
        this.dialog.appendChild(this.modeIndicator);
        this.dialog.appendChild(canvasContainer);
        this.dialog.appendChild(this.pointsInfo);
        this.dialog.appendChild(buttonContainer);

        overlay.appendChild(this.dialog);
        document.body.appendChild(overlay);

        // Event listeners
        this.setupEventListeners(overlay);
    }

    createButton(text, bgColor, onClick) {
        const btn = document.createElement("button");
        btn.textContent = text;
        btn.style.cssText = `
            padding: 8px 20px;
            border: none;
            border-radius: 4px;
            background: ${bgColor};
            color: white;
            cursor: pointer;
            font-size: 14px;
            transition: opacity 0.2s;
        `;
        btn.onmouseover = () => btn.style.opacity = "0.8";
        btn.onmouseout = () => btn.style.opacity = "1";
        btn.onclick = onClick;
        return btn;
    }

    updateModeIndicator() {
        if (!this.modeIndicator) return;
        const color = this.isPositiveMode ? "#4CAF50" : "#f44336";
        const mode = this.isPositiveMode ? "Positive (+)" : "Negative (-)";
        this.modeIndicator.style.cssText = `
            padding: 6px 12px;
            background: ${color};
            color: white;
            border-radius: 4px;
            font-weight: bold;
            text-align: center;
            font-size: 13px;
        `;
        this.modeIndicator.textContent = `Current Mode: ${mode}  (Press + or - to switch)`;
    }

    updatePointsInfo() {
        if (!this.pointsInfo) return;
        const total = this.positivePoints.length + this.negativePoints.length;
        this.pointsInfo.innerHTML = `
            <span style="color: #4CAF50;">Positive: ${this.positivePoints.length}</span> &nbsp;|&nbsp;
            <span style="color: #f44336;">Negative: ${this.negativePoints.length}</span> &nbsp;|&nbsp;
            Total: ${total} points
        `;
    }

    setupEventListeners(overlay) {
        // Canvas click
        this.canvas.addEventListener("mousedown", (e) => this.handleCanvasClick(e));
        this.canvas.addEventListener("contextmenu", (e) => e.preventDefault());

        // Keyboard
        this.keyHandler = (e) => {
            if (e.key === "+" || e.key === "=") {
                this.isPositiveMode = true;
                this.updateModeIndicator();
            } else if (e.key === "-" || e.key === "_") {
                this.isPositiveMode = false;
                this.updateModeIndicator();
            } else if (e.key === "Escape") {
                this.close();
            }
        };
        document.addEventListener("keydown", this.keyHandler);

        // Close on overlay click (but not on dialog click)
        overlay.addEventListener("click", (e) => {
            if (e.target === overlay) {
                this.close();
            }
        });
    }

    handleCanvasClick(e) {
        const rect = this.canvas.getBoundingClientRect();
        // Convert canvas coordinates to original image coordinates
        const x = Math.round((e.clientX - rect.left) / this.scale);
        const y = Math.round((e.clientY - rect.top) / this.scale);

        // Clamp to image bounds
        const clampedX = Math.max(0, Math.min(x, this.imageWidth - 1));
        const clampedY = Math.max(0, Math.min(y, this.imageHeight - 1));

        // Middle click or Ctrl+click = remove nearest point
        if (e.button === 1 || (e.button === 0 && e.ctrlKey)) {
            this.removeNearestPoint(clampedX, clampedY);
        }
        // Right click = negative point
        else if (e.button === 2) {
            this.negativePoints.push({ x: clampedX, y: clampedY });
        }
        // Left click = based on current mode
        else if (e.button === 0) {
            if (this.isPositiveMode) {
                this.positivePoints.push({ x: clampedX, y: clampedY });
            } else {
                this.negativePoints.push({ x: clampedX, y: clampedY });
            }
        }

        this.draw();
        this.updatePointsInfo();
    }

    removeNearestPoint(x, y) {
        let minDist = Infinity;
        let nearestIdx = -1;
        let isPositive = true;

        // Check positive points
        this.positivePoints.forEach((p, i) => {
            const dist = Math.sqrt((p.x - x) ** 2 + (p.y - y) ** 2);
            if (dist < minDist) {
                minDist = dist;
                nearestIdx = i;
                isPositive = true;
            }
        });

        // Check negative points
        this.negativePoints.forEach((p, i) => {
            const dist = Math.sqrt((p.x - x) ** 2 + (p.y - y) ** 2);
            if (dist < minDist) {
                minDist = dist;
                nearestIdx = i;
                isPositive = false;
            }
        });

        // Remove if within threshold (30 pixels in original image space)
        if (minDist < 30 && nearestIdx >= 0) {
            if (isPositive) {
                this.positivePoints.splice(nearestIdx, 1);
            } else {
                this.negativePoints.splice(nearestIdx, 1);
            }
        }
    }

    draw() {
        if (!this.ctx) return;

        // Clear canvas
        this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);

        // Draw image if loaded
        if (this.image && this.image.complete && this.image.naturalWidth > 0) {
            this.ctx.drawImage(this.image, 0, 0, this.canvas.width, this.canvas.height);
        } else {
            // Draw placeholder with grid
            this.ctx.fillStyle = "#1a1a1a";
            this.ctx.fillRect(0, 0, this.canvas.width, this.canvas.height);

            // Grid pattern
            this.ctx.strokeStyle = "#333";
            this.ctx.lineWidth = 1;
            const gridSize = 50;
            for (let x = 0; x < this.canvas.width; x += gridSize) {
                this.ctx.beginPath();
                this.ctx.moveTo(x, 0);
                this.ctx.lineTo(x, this.canvas.height);
                this.ctx.stroke();
            }
            for (let y = 0; y < this.canvas.height; y += gridSize) {
                this.ctx.beginPath();
                this.ctx.moveTo(0, y);
                this.ctx.lineTo(this.canvas.width, y);
                this.ctx.stroke();
            }

            // Info text
            this.ctx.fillStyle = "#666";
            this.ctx.font = "16px Arial";
            this.ctx.textAlign = "center";
            this.ctx.fillText("No image preview available", this.canvas.width / 2, this.canvas.height / 2 - 20);
            this.ctx.fillText("Connect and execute an image source node first", this.canvas.width / 2, this.canvas.height / 2 + 10);
            this.ctx.fillText(`Canvas: ${this.imageWidth} x ${this.imageHeight}`, this.canvas.width / 2, this.canvas.height / 2 + 40);
        }

        // Draw positive points (green)
        this.positivePoints.forEach((p, i) => {
            this.drawPoint(p.x * this.scale, p.y * this.scale, "#4CAF50", "+", i + 1);
        });

        // Draw negative points (red)
        this.negativePoints.forEach((p, i) => {
            this.drawPoint(p.x * this.scale, p.y * this.scale, "#f44336", "-", i + 1);
        });
    }

    drawPoint(x, y, color, symbol, index) {
        const radius = 14;

        // Outer circle with shadow
        this.ctx.shadowColor = "rgba(0, 0, 0, 0.6)";
        this.ctx.shadowBlur = 6;
        this.ctx.shadowOffsetX = 2;
        this.ctx.shadowOffsetY = 2;

        this.ctx.beginPath();
        this.ctx.arc(x, y, radius, 0, Math.PI * 2);
        this.ctx.fillStyle = color;
        this.ctx.fill();

        // Reset shadow
        this.ctx.shadowColor = "transparent";
        this.ctx.shadowBlur = 0;
        this.ctx.shadowOffsetX = 0;
        this.ctx.shadowOffsetY = 0;

        // White border
        this.ctx.strokeStyle = "white";
        this.ctx.lineWidth = 2.5;
        this.ctx.stroke();

        // Symbol
        this.ctx.fillStyle = "white";
        this.ctx.font = "bold 18px Arial";
        this.ctx.textAlign = "center";
        this.ctx.textBaseline = "middle";
        this.ctx.fillText(symbol, x, y);
    }

    save() {
        // Combine all points and create coordinate strings
        const allX = [];
        const allY = [];
        const allLabels = [];

        this.positivePoints.forEach(p => {
            allX.push(Math.round(p.x));
            allY.push(Math.round(p.y));
            allLabels.push(1);
        });

        this.negativePoints.forEach(p => {
            allX.push(Math.round(p.x));
            allY.push(Math.round(p.y));
            allLabels.push(0);
        });

        // Store point data in node for serialization
        if (allX.length > 0) {
            this.node.sam2PointData = {
                points_x: allX.join(","),
                points_y: allY.join(","),
                labels: allLabels.join(",")
            };
        } else {
            // Default center point if no points selected
            const defaultX = Math.round(this.imageWidth / 2);
            const defaultY = Math.round(this.imageHeight / 2);
            this.node.sam2PointData = {
                points_x: String(defaultX),
                points_y: String(defaultY),
                labels: "1"
            };
        }

        console.log("Saved points:", this.node.sam2PointData);
    }

    close() {
        if (this.keyHandler) {
            document.removeEventListener("keydown", this.keyHandler);
        }
        const overlay = document.getElementById("sam2-point-editor-overlay");
        if (overlay) {
            overlay.remove();
        }
        this.dialog = null;
    }
}

// Register the extension
app.registerExtension({
    name: "VideoMaMa.SAM2PointSelector",

    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData?.name !== "SAM2VideoMaskGenerator") {
            return;
        }

        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = async function() {
            const ret = onNodeCreated?.apply(this, arguments);

            // Initialize point data with defaults
            this.sam2PointData = {
                points_x: "512",
                points_y: "288",
                labels: "1"
            };

            // Add the "Select Points" button
            const buttonWidget = this.addWidget("button", "Select Points", null, () => {
                const editor = new SAM2PointEditor(this);
                editor.open();
            });
            buttonWidget.serialize = false;

            return ret;
        };

        // Override serialization to include point data
        const onSerialize = nodeType.prototype.onSerialize;
        nodeType.prototype.onSerialize = function(o) {
            if (onSerialize) {
                onSerialize.apply(this, arguments);
            }
            // Store point data in serialized output
            o.sam2PointData = this.sam2PointData;
        };

        // Override deserialization to restore point data
        const onConfigure = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function(o) {
            if (onConfigure) {
                onConfigure.apply(this, arguments);
            }
            // Restore point data from serialized data
            if (o.sam2PointData) {
                this.sam2PointData = o.sam2PointData;
            }
        };

        // Override getExtraMenuOptions to add point data to execution
        const origGetExtraMenuOptions = nodeType.prototype.getExtraMenuOptions;
        nodeType.prototype.getExtraMenuOptions = function(_, options) {
            if (origGetExtraMenuOptions) {
                origGetExtraMenuOptions.apply(this, arguments);
            }
        };
    },

    // Inject hidden values when node is queued for execution
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData?.name !== "SAM2VideoMaskGenerator") {
            return;
        }
    }
});

// Hook into the prompt serialization to inject point data
const originalQueuePrompt = api.queuePrompt;
api.queuePrompt = async function(number, { output, workflow }) {
    // Find SAM2VideoMaskGenerator nodes and inject point data
    for (const nodeId in output) {
        const nodeData = output[nodeId];
        if (nodeData.class_type === "SAM2VideoMaskGenerator") {
            // Get the node from the graph
            const node = app.graph.getNodeById(parseInt(nodeId));
            if (node && node.sam2PointData) {
                // Inject point data into the node's inputs
                nodeData.inputs.points_x = node.sam2PointData.points_x;
                nodeData.inputs.points_y = node.sam2PointData.points_y;
                nodeData.inputs.labels = node.sam2PointData.labels;
            } else {
                // Use defaults
                nodeData.inputs.points_x = "512";
                nodeData.inputs.points_y = "288";
                nodeData.inputs.labels = "1";
            }
        }
    }
    return originalQueuePrompt.call(this, number, { output, workflow });
};
