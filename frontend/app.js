document.addEventListener("DOMContentLoaded", () => {
    // DOM Elements - Input & Controls
    const dropzone = document.getElementById("dropzone");
    const fileInput = document.getElementById("file-input");
    const dropzonePrompt = document.getElementById("dropzone-prompt");
    const fileInfo = document.getElementById("file-info");
    const fileNameDisplay = document.getElementById("file-name-display");
    const removeFileBtn = document.getElementById("remove-file-btn");
    
    const queryInput = document.getElementById("query-input");
    const vqaInput = document.getElementById("vqa-input");
    const confSlider = document.getElementById("conf-slider");
    const confVal = document.getElementById("conf-val");
    const tileSizeSelect = document.getElementById("tile-size-select");
    const overlapSelect = document.getElementById("overlap-select");
    
    const toggleYolo = document.getElementById("toggle-yolo");
    const toggleGrounding = document.getElementById("toggle-grounding");
    const toggleCaptioning = document.getElementById("toggle-captioning");
    const toggleVqa = document.getElementById("toggle-vqa");
    
    const form = document.getElementById("pipeline-form");
    const btnSubmit = document.getElementById("btn-submit");
    const btnSpinner = document.getElementById("btn-spinner");
    const btnText = document.getElementById("btn-text");
    
    const samplesGallery = document.getElementById("samples-gallery");
    const resetAllBtn = document.getElementById("reset-all-btn");
    
    // Status & Stepper
    const statusMsg = document.getElementById("status-msg");
    const statusTimer = document.getElementById("status-timer");
    const stepNodes = document.querySelectorAll(".step-node");
    
    // Metrics
    const valTiles = document.getElementById("val-tiles");
    const valYolo = document.getElementById("val-yolo");
    const valGrounding = document.getElementById("val-grounding");
    const valTime = document.getElementById("val-time");
    
    // Visualizer Elements & Viewport
    const visualizerCard = document.getElementById("visualizer-card");
    const canvasStage = document.getElementById("canvas-stage");
    const paneOriginal = document.getElementById("pane-original");
    const paneAnnotated = document.getElementById("pane-annotated");
    const wrapperOriginal = document.getElementById("wrapper-original");
    const wrapperAnnotated = document.getElementById("wrapper-annotated");
    const imgOriginal = document.getElementById("img-original");
    const imgAnnotated = document.getElementById("img-annotated");
    const interactiveOverlay = document.getElementById("interactive-overlay");
    const placeholderOriginal = document.getElementById("placeholder-original");
    const placeholderAnnotated = document.getElementById("placeholder-annotated");
    const imageResBadge = document.getElementById("image-res-badge");
    const downloadAnnotatedBtn = document.getElementById("download-annotated-btn");
    
    // Zoom Controls
    const btnZoomIn = document.getElementById("btn-zoom-in");
    const btnZoomOut = document.getElementById("btn-zoom-out");
    const btnZoomFit = document.getElementById("btn-zoom-fit");
    const btnZoomReset = document.getElementById("btn-zoom-reset");
    const zoomLevelBadge = document.getElementById("zoom-level-badge");
    
    const tabBtns = document.querySelectorAll(".tab-btn");
    
    // Insights
    const captionBox = document.getElementById("caption-box");
    const vqaQuestionDisplay = document.getElementById("vqa-question-display");
    const vqaQuestionText = document.getElementById("vqa-question-text");
    const vqaAnswerText = document.getElementById("vqa-answer-text");
    const copyCaptionBtn = document.getElementById("copy-caption-btn");
    const copyVqaBtn = document.getElementById("copy-vqa-btn");
    
    // Table & Inspector
    const filterInput = document.getElementById("filter-input");
    const tableBody = document.getElementById("table-body");
    const exportJsonBtn = document.getElementById("export-json-btn");
    const exportCsvBtn = document.getElementById("export-csv-btn");
    const filterChipsContainer = document.getElementById("filter-chips-container");
    
    // Health
    const gpuStatusText = document.getElementById("gpu-status-text");
    const vramBadge = document.getElementById("vram-badge");
    const healthDot = document.getElementById("health-dot");
    
    // State
    let selectedSampleId = null;
    let selectedFile = null;
    let currentResults = null;
    let timerInterval = null;
    let startTime = 0;
    
    // Pan & Zoom State
    let zoomLevel = 1.0;
    let panX = 0;
    let panY = 0;
    let isPanning = false;
    let startMouseX = 0;
    let startMouseY = 0;
    let activeFilterClass = "ALL";
    let combinedDetections = [];

    // --- Toast Notifications ---
    function showToast(message, type = "info") {
        const container = document.getElementById("toast-container");
        const toast = document.createElement("div");
        toast.className = `toast ${type}`;
        toast.innerHTML = `
            <span>${message}</span>
            <button style="background:none;border:none;color:#fff;cursor:pointer;margin-left:auto;font-size:1.1rem;">&times;</button>
        `;
        toast.querySelector("button").onclick = () => toast.remove();
        container.appendChild(toast);
        setTimeout(() => {
            if (toast.parentNode) toast.remove();
        }, 4000);
    }

    // --- Confidence Slider Listener ---
    if (confSlider && confVal) {
        confSlider.addEventListener("input", (e) => {
            confVal.textContent = parseFloat(e.target.value).toFixed(2);
        });
    }

    // --- Suggestion Chips Click ---
    document.querySelectorAll(".chip-btn").forEach(chip => {
        chip.addEventListener("click", () => {
            const targetId = chip.dataset.target;
            const targetEl = document.getElementById(targetId);
            if (targetEl) {
                targetEl.value = chip.dataset.val;
                targetEl.focus();
                showToast(`Applied preset to ${targetId === "query-input" ? "Grounding" : "VQA"}`, "info");
            }
        });
    });

    // --- Pan & Zoom Implementation ---
    function updateTransform() {
        const transformStr = `translate(${panX}px, ${panY}px) scale(${zoomLevel})`;
        if (wrapperOriginal) wrapperOriginal.style.transform = transformStr;
        if (wrapperAnnotated) wrapperAnnotated.style.transform = transformStr;
        if (zoomLevelBadge) zoomLevelBadge.textContent = `${Math.round(zoomLevel * 100)}%`;
    }

    function setZoom(newZoom) {
        zoomLevel = Math.max(0.2, Math.min(8.0, newZoom));
        updateTransform();
    }

    function resetZoom() {
        zoomLevel = 1.0;
        panX = 0;
        panY = 0;
        updateTransform();
    }

    btnZoomIn.addEventListener("click", () => setZoom(zoomLevel * 1.25));
    btnZoomOut.addEventListener("click", () => setZoom(zoomLevel / 1.25));
    btnZoomReset.addEventListener("click", resetZoom);
    btnZoomFit.addEventListener("click", () => {
        zoomLevel = 1.0;
        panX = 0;
        panY = 0;
        updateTransform();
    });

    // Canvas Mouse Wheel Zoom
    canvasStage.addEventListener("wheel", (e) => {
        if (!imgOriginal.classList.contains("visible") && !imgAnnotated.classList.contains("visible")) return;
        e.preventDefault();
        const delta = e.deltaY < 0 ? 1.15 : 0.87;
        setZoom(zoomLevel * delta);
    }, { passive: false });

    // Drag to Pan
    function onMouseDown(e) {
        if (e.button !== 0) return; // Left click only
        isPanning = true;
        startMouseX = e.clientX - panX;
        startMouseY = e.clientY - panY;
        canvasStage.querySelectorAll(".viewer-pane").forEach(p => p.classList.add("is-panning"));
    }

    function onMouseMove(e) {
        if (!isPanning) return;
        panX = e.clientX - startMouseX;
        panY = e.clientY - startMouseY;
        updateTransform();
    }

    function onMouseUp() {
        if (isPanning) {
            isPanning = false;
            canvasStage.querySelectorAll(".viewer-pane").forEach(p => p.classList.remove("is-panning"));
        }
    }

    canvasStage.addEventListener("mousedown", onMouseDown);
    window.addEventListener("mousemove", onMouseMove);
    window.addEventListener("mouseup", onMouseUp);

    // --- Fetch System Health Status ---
    async function checkHealth() {
        try {
            const res = await fetch("/api/health");
            if (res.ok) {
                const data = await res.json();
                if (data.cuda_available) {
                    gpuStatusText.textContent = `${data.device_name} (GPU Ready)`;
                    vramBadge.textContent = `${data.vram_free_mb} MB Free`;
                    healthDot.className = "health-indicator online";
                } else {
                    gpuStatusText.textContent = "CPU Mode";
                    vramBadge.textContent = "RAM Engine";
                    healthDot.className = "health-indicator warning";
                }
            }
        } catch (e) {
            gpuStatusText.textContent = "Server Offline";
            healthDot.className = "health-indicator warning";
        }
    }
    checkHealth();

    // --- Load Sample Benchmark Datasets ---
    async function loadSamples() {
        try {
            const res = await fetch("/api/samples");
            if (!res.ok) return;
            const data = await res.json();
            const samples = data.samples || [];

            if (samples.length === 0) {
                samplesGallery.innerHTML = `<div class="sample-skeleton">No samples found.</div>`;
                return;
            }

            samplesGallery.innerHTML = samples.map(s => `
                <div class="sample-item" data-id="${s.id}" data-query="${s.default_query}" data-vqa="${s.default_vqa}">
                    <img src="${s.url}" alt="${s.title}" class="sample-thumb" loading="lazy">
                    <div>
                        <div class="sample-title" title="${s.title}">${s.title}</div>
                        <div class="sample-sub">${s.filename}</div>
                    </div>
                </div>
            `).join("");

            document.querySelectorAll(".sample-item").forEach(item => {
                item.addEventListener("click", () => {
                    document.querySelectorAll(".sample-item").forEach(el => el.classList.remove("selected"));
                    item.classList.add("selected");

                    selectedSampleId = item.dataset.id;
                    selectedFile = null;
                    fileInput.value = "";
                    fileInfo.classList.add("hidden");
                    dropzonePrompt.classList.remove("hidden");

                    queryInput.value = item.dataset.query || "";
                    vqaInput.value = item.dataset.vqa || "";

                    // Reset transforms
                    resetZoom();

                    // Preview the sample
                    const sampleUrl = `/api/sample/${selectedSampleId}`;
                    imgOriginal.src = sampleUrl;
                    imgOriginal.classList.add("visible");
                    placeholderOriginal.classList.add("hidden");

                    imgOriginal.onload = () => {
                        imageResBadge.textContent = `${imgOriginal.naturalWidth} × ${imgOriginal.naturalHeight} px`;
                    };

                    showToast(`Loaded benchmark dataset: ${selectedSampleId}`, "success");
                    statusMsg.textContent = `Ready to analyze ${selectedSampleId}. Click Execute GeoNLI Pipeline.`;
                });
            });

        } catch (e) {
            samplesGallery.innerHTML = `<div class="sample-skeleton">Could not load benchmark samples.</div>`;
        }
    }
    loadSamples();

    // --- File Upload & Drag-and-Drop ---
    dropzone.addEventListener("dragover", (e) => {
        e.preventDefault();
        dropzone.classList.add("drag-over");
    });

    dropzone.addEventListener("dragleave", () => {
        dropzone.classList.remove("drag-over");
    });

    dropzone.addEventListener("drop", (e) => {
        e.preventDefault();
        dropzone.classList.remove("drag-over");
        if (e.dataTransfer.files && e.dataTransfer.files[0]) {
            handleSelectedFile(e.dataTransfer.files[0]);
        }
    });

    fileInput.addEventListener("change", (e) => {
        if (e.target.files && e.target.files[0]) {
            handleSelectedFile(e.target.files[0]);
        }
    });

    function handleSelectedFile(file) {
        if (!file.type.startsWith("image/") && !file.name.toLowerCase().endsWith(".tiff") && !file.name.toLowerCase().endsWith(".tif")) {
            showToast("Please upload a valid satellite image (PNG, JPG, TIFF, WEBP).", "error");
            return;
        }

        selectedFile = file;
        selectedSampleId = null;
        document.querySelectorAll(".sample-item").forEach(el => el.classList.remove("selected"));

        fileNameDisplay.textContent = `${file.name} (${(file.size / 1024 / 1024).toFixed(2)} MB)`;
        fileInfo.classList.remove("hidden");
        dropzonePrompt.classList.add("hidden");

        resetZoom();

        const previewUrl = URL.createObjectURL(file);
        imgOriginal.src = previewUrl;
        imgOriginal.classList.add("visible");
        placeholderOriginal.classList.add("hidden");

        imgOriginal.onload = () => {
            imageResBadge.textContent = `${imgOriginal.naturalWidth} × ${imgOriginal.naturalHeight} px`;
        };

        statusMsg.textContent = `Loaded image: ${file.name}. Ready for processing.`;
        showToast("Satellite image loaded successfully.", "success");
    }

    removeFileBtn.addEventListener("click", (e) => {
        e.stopPropagation();
        selectedFile = null;
        fileInput.value = "";
        fileInfo.classList.add("hidden");
        dropzonePrompt.classList.remove("hidden");
        imgOriginal.src = "";
        imgOriginal.classList.remove("visible");
        placeholderOriginal.classList.remove("hidden");
        imageResBadge.textContent = "0 × 0 px";
        statusMsg.textContent = "Image removed. Select a benchmark sample or upload a file.";
    });

    // --- Stepper UI ---
    function setStep(stepNumber) {
        stepNodes.forEach(node => {
            const step = parseInt(node.dataset.step);
            if (step < stepNumber) {
                node.className = "step-node completed";
            } else if (step === stepNumber) {
                node.className = "step-node active";
            } else {
                node.className = "step-node";
            }
        });
    }

    function startTimer() {
        startTime = Date.now();
        statusTimer.classList.remove("hidden");
        statusTimer.textContent = "0.0s";
        timerInterval = setInterval(() => {
            const elapsed = ((Date.now() - startTime) / 1000).toFixed(1);
            statusTimer.textContent = `${elapsed}s`;
        }, 100);
    }

    function stopTimer() {
        clearInterval(timerInterval);
        if (startTime) {
            const elapsed = ((Date.now() - startTime) / 1000).toFixed(2);
            valTime.textContent = `${elapsed}s`;
        }
    }

    // --- Tab Switcher ---
    tabBtns.forEach(btn => {
        btn.addEventListener("click", () => {
            tabBtns.forEach(b => b.classList.remove("active"));
            btn.classList.add("active");
            const view = btn.dataset.view;
            canvasStage.className = `canvas-stage view-${view}`;
        });
    });

    function switchView(viewName) {
        tabBtns.forEach(btn => {
            if (btn.dataset.view === viewName) {
                btn.click();
            }
        });
    }

    // --- Copy Actions ---
    copyCaptionBtn.addEventListener("click", () => {
        const text = captionBox.innerText;
        if (text && !captionBox.querySelector(".empty-state")) {
            navigator.clipboard.writeText(text);
            showToast("Scene description copied to clipboard!", "success");
        }
    });

    copyVqaBtn.addEventListener("click", () => {
        const text = vqaAnswerText.innerText;
        if (text && !vqaAnswerText.querySelector(".empty-state")) {
            navigator.clipboard.writeText(text);
            showToast("VQA answer copied to clipboard!", "success");
        }
    });

    // --- Form Submit & Pipeline Execution ---
    form.addEventListener("submit", async (e) => {
        e.preventDefault();

        if (!selectedFile && !selectedSampleId) {
            showToast("Please upload a satellite image or select a benchmark sample first.", "error");
            return;
        }

        btnSubmit.disabled = true;
        btnSpinner.classList.remove("hidden");
        btnText.textContent = "Executing Geospatial Pipeline...";

        setStep(1);
        statusMsg.textContent = "Generating high-resolution spatial tiles...";
        startTimer();

        // Clear previous outputs & highlights
        imgAnnotated.src = "";
        imgAnnotated.classList.remove("visible");
        placeholderAnnotated.classList.remove("hidden");
        interactiveOverlay.innerHTML = "";
        downloadAnnotatedBtn.classList.add("disabled");
        downloadAnnotatedBtn.removeAttribute("href");

        try {
            const formData = new FormData();

            if (selectedFile) {
                formData.append("image", selectedFile);
            } else if (selectedSampleId) {
                formData.append("sample_id", selectedSampleId);
            }

            formData.append("query", queryInput.value.trim());
            formData.append("vqa_question", vqaInput.value.trim());
            formData.append("tile_size", tileSizeSelect.value);
            formData.append("overlap", overlapSelect.value);
            formData.append("confidence", confSlider ? confSlider.value : "0.25");
            formData.append("run_yolo", toggleYolo.checked ? "true" : "false");
            formData.append("run_grounding", toggleGrounding.checked ? "true" : "false");
            formData.append("run_captioning", toggleCaptioning.checked ? "true" : "false");
            formData.append("run_vqa", toggleVqa.checked ? "true" : "false");

            // Progressive step indicators
            setTimeout(() => { if (btnSubmit.disabled) { setStep(2); statusMsg.textContent = "Running YOLOv11 Oriented Bounding Box detection on tiles..."; } }, 600);
            setTimeout(() => { if (btnSubmit.disabled) { setStep(3); statusMsg.textContent = "Performing Rotated Non-Maximum Suppression (Rotated NMS)..."; } }, 1400);
            setTimeout(() => { if (btnSubmit.disabled) { setStep(4); statusMsg.textContent = "Running Grounding DINO text grounding & Qwen2-VL reasoning..."; } }, 2200);

            const response = await fetch("/api/analyze", {
                method: "POST",
                body: formData,
            });

            const data = await response.json();

            if (!response.ok || data.status === "error") {
                throw new Error(data.error || "Pipeline execution failed on server.");
            }

            setStep(5);
            currentResults = data;
            renderResults(data);
            showToast("GeoNLI pipeline execution complete!", "success");
            statusMsg.textContent = "Analysis completed successfully. Inspect detections and multimodal answers below.";

        } catch (err) {
            console.error(err);
            showToast(`Error: ${err.message}`, "error");
            statusMsg.textContent = `Execution failed: ${err.message}`;
        } finally {
            stopTimer();
            btnSubmit.disabled = false;
            btnSpinner.classList.add("hidden");
            btnText.textContent = "Execute GeoNLI Pipeline";
        }
    });

    // --- Render Results ---
    function renderResults(data) {
        // Metrics
        valTiles.textContent = data.tiles || 0;
        valYolo.textContent = data.final_detection_count || 0;
        valGrounding.textContent = data.grounding_count || 0;

        // Annotated Image View
        if (data.annotated_image) {
            imgAnnotated.src = data.annotated_image + `?t=${Date.now()}`;
            imgAnnotated.classList.add("visible");
            placeholderAnnotated.classList.add("hidden");

            downloadAnnotatedBtn.href = data.annotated_image;
            downloadAnnotatedBtn.classList.remove("disabled");
        }

        // VLM Caption
        if (data.caption) {
            captionBox.innerHTML = `<p>${formatMarkdown(data.caption)}</p>`;
        } else {
            captionBox.innerHTML = `<p class="empty-state">Captioning was disabled or returned no output.</p>`;
        }

        // VQA
        if (data.vqa_question) {
            vqaQuestionText.textContent = data.vqa_question;
            vqaQuestionDisplay.classList.remove("hidden");
        } else {
            vqaQuestionDisplay.classList.add("hidden");
        }
        
        if (data.vqa_answer) {
            vqaAnswerText.innerHTML = `<p>${formatMarkdown(data.vqa_answer)}</p>`;
        } else {
            vqaAnswerText.innerHTML = `<p class="empty-state">No VQA question was submitted or answered.</p>`;
        }

        // Build Combined Detections List
        combinedDetections = [];
        const yoloDets = data.detections || [];
        const groundDets = data.grounded || [];

        yoloDets.forEach((d, idx) => {
            combinedDetections.push({
                id: `det-yolo-${idx}`,
                index: idx + 1,
                engine: "YOLO OBB",
                engineClass: "yolo",
                label: d.class_name,
                score: d.score,
                cx: d.cx,
                cy: d.cy,
                w: d.w,
                h: d.h,
                angle: d.angle_deg || 0,
                coords: `(${d.cx.toFixed(1)}, ${d.cy.toFixed(1)})`,
                dim: `${d.w.toFixed(1)} × ${d.h.toFixed(1)}`,
                angleStr: `${d.angle_deg ? d.angle_deg.toFixed(1) + '°' : '0°'}`,
            });
        });

        groundDets.forEach((g, idx) => {
            const w = g.w ? g.w : (g.x2 - g.x1);
            const h = g.h ? g.h : (g.y2 - g.y1);
            const cx = g.x1 + w / 2;
            const cy = g.y1 + h / 2;
            combinedDetections.push({
                id: `det-dino-${idx}`,
                index: yoloDets.length + idx + 1,
                engine: "Grounding DINO",
                engineClass: "dino",
                label: g.label,
                score: g.score,
                cx: cx,
                cy: cy,
                w: w,
                h: h,
                angle: 0,
                coords: `(${g.x1.toFixed(1)}, ${g.y1.toFixed(1)})`,
                dim: `${w.toFixed(1)} × ${h.toFixed(1)}`,
                angleStr: `0.0°`,
            });
        });

        // Setup Class Filter Chips
        setupClassFilters(combinedDetections);

        // Render Table & Overlay
        renderTable(combinedDetections);
        renderInteractiveOverlay(combinedDetections, data.image_width, data.image_height);
    }

    function formatMarkdown(text) {
        if (!text) return "";
        return text
            .replace(/\n\n/g, "<br><br>")
            .replace(/\*\*(.*?)\*\*/g, "<strong>$1</strong>")
            .replace(/\*(.*?)\*/g, "<em>$1</em>");
    }

    // --- Setup Class Filter Chips ---
    function setupClassFilters(items) {
        const counts = { ALL: items.length };
        items.forEach(it => {
            counts[it.label] = (counts[it.label] || 0) + 1;
        });

        activeFilterClass = "ALL";
        let chipsHtml = `<span class="class-chip active" data-class="ALL">All (${items.length})</span>`;

        Object.keys(counts).forEach(cls => {
            if (cls !== "ALL") {
                chipsHtml += `<span class="class-chip" data-class="${cls}">${cls} (${counts[cls]})</span>`;
            }
        });

        filterChipsContainer.innerHTML = chipsHtml;

        filterChipsContainer.querySelectorAll(".class-chip").forEach(chip => {
            chip.addEventListener("click", () => {
                filterChipsContainer.querySelectorAll(".class-chip").forEach(c => c.classList.remove("active"));
                chip.classList.add("active");
                activeFilterClass = chip.dataset.class;
                applyFilters();
            });
        });
    }

    function applyFilters() {
        const query = filterInput.value.toLowerCase();
        const filtered = combinedDetections.filter(item => {
            const matchesClass = (activeFilterClass === "ALL" || item.label === activeFilterClass);
            const matchesSearch = (!query || item.label.toLowerCase().includes(query) || item.engine.toLowerCase().includes(query));
            return matchesClass && matchesSearch;
        });

        populateTableRows(filtered);
    }

    filterInput.addEventListener("input", applyFilters);

    // --- Render Detections Table ---
    function renderTable(items) {
        if (items.length === 0) {
            tableBody.innerHTML = `<tr><td colspan="8" class="text-center empty-cell">No objects detected above confidence threshold.</td></tr>`;
            return;
        }
        populateTableRows(items);
    }

    function populateTableRows(items) {
        if (items.length === 0) {
            tableBody.innerHTML = `<tr><td colspan="8" class="text-center empty-cell">No detections match the current filter.</td></tr>`;
            return;
        }

        tableBody.innerHTML = items.map(item => {
            const pct = Math.round(item.score * 100);
            return `
                <tr id="row-${item.id}" data-id="${item.id}" data-cx="${item.cx}" data-cy="${item.cy}">
                    <td>${item.index}</td>
                    <td><span class="engine-tag ${item.engineClass}">${item.engine}</span></td>
                    <td><strong>${item.label}</strong></td>
                    <td>
                        <div class="score-bar-wrapper">
                            <span>${(item.score).toFixed(2)}</span>
                            <div class="score-bar">
                                <div class="score-fill" style="width: ${pct}%"></div>
                            </div>
                        </div>
                    </td>
                    <td>${item.coords}</td>
                    <td>${item.dim}</td>
                    <td>${item.angleStr}</td>
                    <td>
                        <button type="button" class="btn-inspect-row" data-id="${item.id}" title="Center & zoom into detection">Inspect</button>
                    </td>
                </tr>
            `;
        }).join("");

        // Add Row Hover & Click Inspect Event Listeners
        tableBody.querySelectorAll("tr").forEach(tr => {
            const detId = tr.dataset.id;
            
            tr.addEventListener("mouseenter", () => {
                highlightOverlayBox(detId);
            });

            tr.addEventListener("mouseleave", () => {
                removeOverlayHighlight(detId);
            });

            tr.addEventListener("click", () => {
                focusOnDetection(detId);
            });
        });

        tableBody.querySelectorAll(".btn-inspect-row").forEach(btn => {
            btn.addEventListener("click", (e) => {
                e.stopPropagation();
                focusOnDetection(btn.dataset.id);
            });
        });
    }

    // --- Render Interactive Overlay ---
    function renderInteractiveOverlay(items, imgW, imgH) {
        interactiveOverlay.innerHTML = "";
        if (!items || items.length === 0 || !imgW || !imgH) return;

        items.forEach(item => {
            const box = document.createElement("div");
            box.className = `det-highlight-box ${item.engineClass}`;
            box.id = `box-${item.id}`;

            // Coordinates as percentage of natural image dimensions
            const leftPct = ((item.cx - item.w / 2) / imgW) * 100;
            const topPct = ((item.cy - item.h / 2) / imgH) * 100;
            const widthPct = (item.w / imgW) * 100;
            const heightPct = (item.h / imgH) * 100;

            box.style.left = `${leftPct}%`;
            box.style.top = `${topPct}%`;
            box.style.width = `${widthPct}%`;
            box.style.height = `${heightPct}%`;
            if (item.angle) {
                box.style.transform = `rotate(${item.angle}deg)`;
            }

            box.title = `${item.label} (${(item.score * 100).toFixed(0)}%)`;

            box.addEventListener("mouseenter", () => {
                highlightTableRow(item.id);
            });

            box.addEventListener("mouseleave", () => {
                removeTableHighlight(item.id);
            });

            box.addEventListener("click", (e) => {
                e.stopPropagation();
                focusOnDetection(item.id);
            });

            interactiveOverlay.appendChild(box);
        });
    }

    function highlightOverlayBox(id) {
        const box = document.getElementById(`box-${id}`);
        if (box) {
            box.style.borderColor = "#fff";
            box.style.backgroundColor = "rgba(0, 229, 153, 0.4)";
            box.style.zIndex = "10";
        }
    }

    function removeOverlayHighlight(id) {
        const box = document.getElementById(`box-${id}`);
        if (box) {
            box.style.borderColor = "";
            box.style.backgroundColor = "";
            box.style.zIndex = "";
        }
    }

    function highlightTableRow(id) {
        const row = document.getElementById(`row-${id}`);
        if (row) {
            row.classList.add("highlighted");
            row.scrollIntoView({ behavior: "smooth", block: "nearest" });
        }
    }

    function removeTableHighlight(id) {
        const row = document.getElementById(`row-${id}`);
        if (row) {
            row.classList.remove("highlighted");
        }
    }

    // Zoom & Center to specific detection
    function focusOnDetection(id) {
        const item = combinedDetections.find(d => d.id === id);
        if (!item || !currentResults) return;

        // Switch to Annotated or Side-by-side view if needed
        const stage = canvasStage.className;
        if (stage.includes("view-original")) {
            switchView("annotated");
        }

        zoomLevel = 2.5;
        // Center pan relative to canvas center
        const imgW = currentResults.image_width || 1024;
        const imgH = currentResults.image_height || 1024;
        const normX = (item.cx / imgW) - 0.5;
        const normY = (item.cy / imgH) - 0.5;

        panX = -normX * 400 * zoomLevel;
        panY = -normY * 400 * zoomLevel;
        updateTransform();

        highlightOverlayBox(id);
        highlightTableRow(id);
        showToast(`Inspecting ${item.label} [${item.score.toFixed(2)}]`, "info");
    }

    // --- Export CSV ---
    exportCsvBtn.addEventListener("click", () => {
        if (!combinedDetections || combinedDetections.length === 0) {
            showToast("No detections to export yet.", "info");
            return;
        }

        const headers = ["Index", "Engine", "Class_Label", "Confidence", "Center_X", "Center_Y", "Width", "Height", "Angle_Deg"];
        const rows = combinedDetections.map(d => [
            d.index,
            `"${d.engine}"`,
            `"${d.label}"`,
            d.score,
            d.cx,
            d.cy,
            d.w,
            d.h,
            d.angle || 0,
        ]);

        const csvContent = [headers.join(","), ...rows.map(r => r.join(","))].join("\n");
        const blob = new Blob([csvContent], { type: "text/csv;charset=utf-8;" });
        const url = URL.createObjectURL(blob);
        const a = document.createElement("a");
        a.href = url;
        a.download = `geonli_detections_${Date.now()}.csv`;
        a.click();
        URL.revokeObjectURL(url);
        showToast("Detections exported to CSV successfully.", "success");
    });

    // --- Export JSON ---
    exportJsonBtn.addEventListener("click", () => {
        if (!currentResults) {
            showToast("No analysis results to export yet.", "info");
            return;
        }

        const jsonStr = JSON.stringify(currentResults, null, 2);
        const blob = new Blob([jsonStr], { type: "application/json" });
        const url = URL.createObjectURL(blob);
        const a = document.createElement("a");
        a.href = url;
        a.download = `geonli_analysis_${Date.now()}.json`;
        a.click();
        URL.revokeObjectURL(url);
        showToast("Full analysis JSON exported successfully.", "success");
    });

    // --- Reset View ---
    resetAllBtn.addEventListener("click", () => {
        selectedSampleId = null;
        selectedFile = null;
        currentResults = null;
        combinedDetections = [];
        fileInput.value = "";
        queryInput.value = "";
        vqaInput.value = "";
        filterInput.value = "";
        fileInfo.classList.add("hidden");
        dropzonePrompt.classList.remove("hidden");
        
        imgOriginal.src = "";
        imgOriginal.classList.remove("visible");
        placeholderOriginal.classList.remove("hidden");

        imgAnnotated.src = "";
        imgAnnotated.classList.remove("visible");
        placeholderAnnotated.classList.remove("hidden");
        interactiveOverlay.innerHTML = "";

        valTiles.textContent = "0";
        valYolo.textContent = "0";
        valGrounding.textContent = "0";
        valTime.textContent = "0.0s";

        captionBox.innerHTML = `<p class="empty-state">Detailed vision-language description will appear here after analysis.</p>`;
        vqaAnswerText.innerHTML = `<p class="empty-state">VQA reasoning output will appear here.</p>`;
        vqaQuestionDisplay.classList.add("hidden");

        tableBody.innerHTML = `<tr><td colspan="8" class="text-center empty-cell">No detections recorded yet. Run the pipeline to inspect bounding objects.</td></tr>`;
        filterChipsContainer.innerHTML = `<span class="class-chip active" data-class="ALL">All (0)</span>`;
        document.querySelectorAll(".sample-item").forEach(el => el.classList.remove("selected"));

        resetZoom();
        setStep(1);
        statusMsg.textContent = "Studio reset. Select a benchmark sample or upload an image to begin.";
        showToast("GeoNLI Studio state reset.", "info");
    });

    // --- Global Keyboard Shortcuts ---
    window.addEventListener("keydown", (e) => {
        // Ctrl+Enter or Cmd+Enter to execute
        if ((e.ctrlKey || e.metaKey) && e.key === "Enter") {
            e.preventDefault();
            form.requestSubmit();
        }
        // Esc to reset zoom & blur
        if (e.key === "Escape") {
            resetZoom();
            if (document.activeElement) document.activeElement.blur();
        }
        // Switch views with 1, 2, 3 when not typing in inputs
        const isInput = ["INPUT", "TEXTAREA", "SELECT"].includes(document.activeElement.tagName);
        if (!isInput) {
            if (e.key === "1") switchView("side-by-side");
            if (e.key === "2") switchView("annotated");
            if (e.key === "3") switchView("original");
            if (e.key === "+" || e.key === "=") setZoom(zoomLevel * 1.25);
            if (e.key === "-") setZoom(zoomLevel / 1.25);
            if (e.key === "0") resetZoom();
        }
    });
});
