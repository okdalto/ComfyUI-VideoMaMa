import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

/**
 * SAM2 Point Selector Extension for ComfyUI
 * Provides a visual interface for selecting positive/negative points on video frames
 */

class SAM2PointEditor {
    constructor(node, imageWidget, posWidget, negWidget, labelsWidget) {
        this.node = node;
        this.imageWidget = imageWidget;
        this.posWidget = posWidget;
        this.negWidget = negWidget;
        this.labelsWidget = labelsWidget;

        this.positivePoints = [];
        this.negativePoints = [];
        this.isPositiveMode = true;
        this.image = null;
        this.dialog = null;
        this.canvas = null;
        this.ctx = null;
        this.scale = 1;
    }

    async open() {
        // Parse existing points
        this.parseExistingPoints();

        // Create dialog
        this.createDialog();

        // Load image from node input
        await this.loadImageFromNode();

        // Draw initial state
        this.draw();
    }

    parseExistingPoints() {
        try {
            const xCoords = this.posWidget.value.split(",").map(x => parseInt(x.trim())).filter(x => !isNaN(x));
            const yCoords = this.negWidget.value.split(",").map(y => parseInt(y.trim())).filter(y => !isNaN(y));
            const labels = this.labelsWidget.value.split(",").map(l => parseInt(l.trim())).filter(l => !isNaN(l));

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

    createDialog() {
        // Remove existing dialog if any
        if (this.dialog) {
            this.dialog.remove();
        }

        // Create overlay
        const overlay = document.createElement("div");
        overlay.id = "sam2-point-editor-overlay";
        overlay.style.cssText = `
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background: rgba(0, 0, 0, 0.8);
            z-index: 10000;
            display: flex;
            justify-content: center;
            align-items: center;
        `;

        // Create dialog container
        this.dialog = document.createElement("div");
        this.dialog.id = "sam2-point-editor-dialog";
        this.dialog.style.cssText = `
            background: #2a2a2a;
            border-radius: 8px;
            padding: 20px;
            max-width: 90vw;
            max-height: 90vh;
            display: flex;
            flex-direction: column;
            gap: 15px;
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
            font-size: 13px;
            line-height: 1.5;
        `;
        instructions.innerHTML = `
            <b>Left click:</b> Add positive point (include) &nbsp;|&nbsp;
            <b>Right click:</b> Add negative point (exclude) &nbsp;|&nbsp;
            <b>Middle click / Ctrl+click:</b> Remove nearest point<br>
            <b>Keyboard:</b> Press <kbd>+</kbd> for positive mode, <kbd>-</kbd> for negative mode
        `;

        // Canvas container
        const canvasContainer = document.createElement("div");
        canvasContainer.style.cssText = `
            position: relative;
            max-width: 100%;
            max-height: 60vh;
            overflow: auto;
            border: 1px solid #444;
            border-radius: 4px;
        `;

        // Canvas
        this.canvas = document.createElement("canvas");
        this.canvas.style.cssText = `
            display: block;
            cursor: crosshair;
        `;
        canvasContainer.appendChild(this.canvas);
        this.ctx = this.canvas.getContext("2d");

        // Mode indicator
        this.modeIndicator = document.createElement("div");
        this.updateModeIndicator();

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
            padding: 8px 15px;
            background: ${color};
            color: white;
            border-radius: 4px;
            font-weight: bold;
            text-align: center;
        `;
        this.modeIndicator.textContent = `Current Mode: ${mode}`;
    }

    updatePointsInfo() {
        if (!this.pointsInfo) return;
        this.pointsInfo.innerHTML = `
            <span style="color: #4CAF50;">Positive points: ${this.positivePoints.length}</span> &nbsp;|&nbsp;
            <span style="color: #f44336;">Negative points: ${this.negativePoints.length}</span>
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

        // Close on overlay click
        overlay.addEventListener("click", (e) => {
            if (e.target === overlay) {
                this.close();
            }
        });
    }

    handleCanvasClick(e) {
        const rect = this.canvas.getBoundingClientRect();
        const x = Math.round((e.offsetX / this.scale));
        const y = Math.round((e.offsetY / this.scale));

        // Middle click or Ctrl+click = remove nearest point
        if (e.button === 1 || (e.button === 0 && e.ctrlKey)) {
            this.removeNearestPoint(x, y);
        }
        // Right click = negative point (or add if in negative mode)
        else if (e.button === 2) {
            this.negativePoints.push({ x, y });
        }
        // Left click = positive point (or mode-based)
        else if (e.button === 0) {
            if (this.isPositiveMode) {
                this.positivePoints.push({ x, y });
            } else {
                this.negativePoints.push({ x, y });
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

        // Remove if within threshold (30 pixels)
        if (minDist < 30 && nearestIdx >= 0) {
            if (isPositive) {
                this.positivePoints.splice(nearestIdx, 1);
            } else {
                this.negativePoints.splice(nearestIdx, 1);
            }
        }
    }

    async loadImageFromNode() {
        // Try to get image from connected node
        const imageInput = this.node.inputs?.find(i => i.name === "images");

        if (imageInput && imageInput.link !== null) {
            const linkInfo = app.graph.links[imageInput.link];
            if (linkInfo) {
                const sourceNode = app.graph.getNodeById(linkInfo.origin_id);
                if (sourceNode && sourceNode.imgs && sourceNode.imgs.length > 0) {
                    // Use the first frame from source node
                    return new Promise((resolve) => {
                        this.image = new Image();
                        this.image.onload = () => {
                            this.setupCanvas();
                            resolve();
                        };
                        this.image.src = sourceNode.imgs[0].src;
                    });
                }
            }
        }

        // Fallback: create placeholder
        this.createPlaceholderCanvas();
    }

    createPlaceholderCanvas() {
        this.canvas.width = 512;
        this.canvas.height = 288;
        this.scale = 1;
        this.ctx.fillStyle = "#333";
        this.ctx.fillRect(0, 0, 512, 288);
        this.ctx.fillStyle = "#888";
        this.ctx.font = "16px Arial";
        this.ctx.textAlign = "center";
        this.ctx.fillText("Connect an image/video input to see preview", 256, 144);
        this.ctx.fillText("You can still add points by clicking", 256, 170);
    }

    setupCanvas() {
        if (!this.image) return;

        // Calculate scale to fit in viewport
        const maxWidth = window.innerWidth * 0.8;
        const maxHeight = window.innerHeight * 0.6;

        this.scale = Math.min(
            maxWidth / this.image.width,
            maxHeight / this.image.height,
            1  // Don't scale up
        );

        this.canvas.width = this.image.width * this.scale;
        this.canvas.height = this.image.height * this.scale;
    }

    draw() {
        if (!this.ctx) return;

        // Clear and draw image
        this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);

        if (this.image) {
            this.ctx.drawImage(this.image, 0, 0, this.canvas.width, this.canvas.height);
        } else {
            this.ctx.fillStyle = "#333";
            this.ctx.fillRect(0, 0, this.canvas.width, this.canvas.height);
        }

        // Draw positive points (green)
        this.positivePoints.forEach(p => {
            this.drawPoint(p.x * this.scale, p.y * this.scale, "#4CAF50", "+");
        });

        // Draw negative points (red)
        this.negativePoints.forEach(p => {
            this.drawPoint(p.x * this.scale, p.y * this.scale, "#f44336", "-");
        });
    }

    drawPoint(x, y, color, symbol) {
        const radius = 12;

        // Outer circle with shadow
        this.ctx.shadowColor = "rgba(0, 0, 0, 0.5)";
        this.ctx.shadowBlur = 4;
        this.ctx.beginPath();
        this.ctx.arc(x, y, radius, 0, Math.PI * 2);
        this.ctx.fillStyle = color;
        this.ctx.fill();
        this.ctx.shadowBlur = 0;

        // White border
        this.ctx.strokeStyle = "white";
        this.ctx.lineWidth = 2;
        this.ctx.stroke();

        // Symbol
        this.ctx.fillStyle = "white";
        this.ctx.font = "bold 16px Arial";
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
            allX.push(p.x);
            allY.push(p.y);
            allLabels.push(1);
        });

        this.negativePoints.forEach(p => {
            allX.push(p.x);
            allY.push(p.y);
            allLabels.push(0);
        });

        // Update node widgets
        if (allX.length > 0) {
            this.posWidget.value = allX.join(",");
            this.negWidget.value = allY.join(",");
            this.labelsWidget.value = allLabels.join(",");
        } else {
            // Default values if no points
            this.posWidget.value = "512";
            this.negWidget.value = "288";
            this.labelsWidget.value = "1";
        }

        // Trigger widget callbacks if any
        this.posWidget.callback?.(this.posWidget.value);
        this.negWidget.callback?.(this.negWidget.value);
        this.labelsWidget.callback?.(this.labelsWidget.value);
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

            // Find the coordinate widgets
            const posWidget = this.widgets.find(w => w.name === "points_x");
            const negWidget = this.widgets.find(w => w.name === "points_y");
            const labelsWidget = this.widgets.find(w => w.name === "labels");

            if (posWidget && negWidget && labelsWidget) {
                // Add the "Select Points" button
                const buttonWidget = this.addWidget("button", "Select Points", null, () => {
                    const editor = new SAM2PointEditor(
                        this,
                        null,
                        posWidget,
                        negWidget,
                        labelsWidget
                    );
                    editor.open();
                });

                // Style the button
                buttonWidget.serialize = false;
            }

            return ret;
        };
    }
});
