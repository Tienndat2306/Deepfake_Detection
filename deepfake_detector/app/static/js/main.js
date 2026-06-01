const form = document.getElementById("analyze-form");
const input = document.getElementById("video-input");
const chooseButton = document.getElementById("choose-file");
const analyzeButton = document.getElementById("analyze-button");
const fileName = document.getElementById("file-name");
const fileSize = document.getElementById("file-size");
const preview = document.getElementById("video-preview");
const emptyPreview = document.getElementById("empty-preview");
const systemStatus = document.getElementById("system-status");
const systemDetail = document.getElementById("system-detail");
const frameModal = document.getElementById("frame-modal");
const frameModalImage = document.getElementById("frame-modal-image");
const frameModalCaption = document.getElementById("frame-modal-caption");
const frameModalClose = document.getElementById("frame-modal-close");

function formatPercent(value) {
    return `${(Number(value) * 100).toFixed(2)}%`;
}

function formatBytes(bytes) {
    if (!bytes) return "0 B";
    const units = ["B", "KB", "MB", "GB"];
    const index = Math.min(Math.floor(Math.log(bytes) / Math.log(1024)), units.length - 1);
    return `${(bytes / Math.pow(1024, index)).toFixed(index === 0 ? 0 : 2)} ${units[index]}`;
}

function setText(id, value) {
    const element = document.getElementById(id);
    if (element) element.textContent = value;
}

function setStatus(title) {
    if (systemStatus) systemStatus.textContent = title;
    if (systemDetail) systemDetail.textContent = ""; 
}

function showError(message) {
    const existing = document.querySelector(".toast");
    if (existing) existing.remove();
    const toast = document.createElement("div");
    toast.className = "toast";
    toast.textContent = message;
    form.insertAdjacentElement("afterend", toast);
}

function clearError() {
    const existing = document.querySelector(".toast");
    if (existing) existing.remove();
}

function setLoading(isLoading) {
    analyzeButton.disabled = isLoading || !input.files.length;
    analyzeButton.textContent = isLoading ? "Analyzing..." : "Analyze";
}

function openFrameModal(frame) {
    if (!frameModal || !frameModalImage || !frameModalCaption) return;
    frameModalImage.src = frame.thumbnail_url;
    frameModalImage.alt = `Frame ${frame.frame_number}`;
    frameModalCaption.textContent = `Frame ${frame.frame_number}`;
    frameModal.hidden = false;
    document.body.classList.add("modal-open");
}

function closeFrameModal() {
    if (!frameModal || !frameModalImage) return;
    frameModal.hidden = true;
    frameModalImage.src = "";
    document.body.classList.remove("modal-open");
}

function updateResult(result) {
    const verdict = result.verdict;
    const fakeProbability = Number(result.fake_probability);
    const realProbability = Number(result.real_probability);
    const confidence = Number(result.confidence);
    const meta = result.metadata || {};

    setText("result-subtitle", `Session ${result.session_id}`);
    setText("fake-score", formatPercent(fakeProbability));
    setText("real-score", formatPercent(realProbability));
    setText("confidence-score", formatPercent(confidence));

    const fakeMeter = document.getElementById("fake-meter");
    if (fakeMeter) fakeMeter.style.width = `${Math.round(fakeProbability * 100)}%`;

    const pill = document.getElementById("verdict-pill");
    if (pill) {
        pill.textContent = verdict;
        pill.className = `pill ${verdict.toLowerCase()}`;
    }

    setText("meta-frames", meta.total_frames ?? "--");
    setText("meta-resolution", meta.width && meta.height ? `${meta.width} x ${meta.height}` : "--");
    setText("meta-duration", meta.duration_seconds ? `${Number(meta.duration_seconds).toFixed(2)}s` : "--");

    const frames = document.getElementById("keyframes");
    if (!frames) return;
    frames.innerHTML = "";
    if (!result.keyframes || result.keyframes.length === 0) {
        frames.innerHTML = '<div class="empty-frames">No keyframes extracted!</div>';
        return;
    }

    result.keyframes.forEach((frame) => {
        const item = document.createElement("article");
        item.className = "keyframe";

        const button = document.createElement("button");
        button.className = "keyframe-preview";
        button.type = "button";
        button.setAttribute("aria-label", `Open frame ${frame.frame_number}`);

        const image = document.createElement("img");
        image.src = frame.thumbnail_url;
        image.alt = `Frame ${frame.frame_number}`;
        button.appendChild(image);
        button.addEventListener("click", () => openFrameModal(frame));

        const footer = document.createElement("footer");
        const frameNumber = document.createElement("span");
        frameNumber.textContent = `Frame ${frame.frame_number}`;
        const confidence = document.createElement("span");
        confidence.textContent = `${Number(frame.confidence || 0).toFixed(1)}%`;
        footer.append(frameNumber, confidence);

        item.append(button, footer);
        frames.appendChild(item);
    });
}

if (frameModalClose) {
    frameModalClose.addEventListener("click", closeFrameModal);
}

if (frameModal) {
    frameModal.addEventListener("click", (event) => {
        if (event.target === frameModal) closeFrameModal();
    });
}

document.addEventListener("keydown", (event) => {
    if (event.key === "Escape" && frameModal && !frameModal.hidden) {
        closeFrameModal();
    }
});

if (chooseButton && input) {
    chooseButton.addEventListener("click", () => input.click());
}

if (input) {
    input.addEventListener("change", () => {
        clearError();
        const file = input.files[0];
        if (!file) return;
        fileName.textContent = file.name;
        fileSize.textContent = formatBytes(file.size);
        analyzeButton.disabled = false;

        const url = URL.createObjectURL(file);
        preview.src = url;
        preview.style.display = "block";
        emptyPreview.style.display = "none";
    });
}

if (form) {
    form.addEventListener("submit", async (event) => {
        event.preventDefault();
        clearError();
        if (!input.files.length) return;

        const data = new FormData();
        data.append("video", input.files[0]);

        try {
            setLoading(true);
            setStatus("RUNNING"); 
            const response = await fetch("/api/analyze", {
                method: "POST",
                body: data,
            });
            const payload = await response.json();
            if (!response.ok || !payload.ok) {
                throw new Error(payload.error || "Failed to analyze video.");
            }
            updateResult(payload.result);
            setStatus("DONE"); 
        } catch (error) {
            showError(error.message);
            setStatus("ERROR"); 
        } finally {
            setLoading(false);
        }
    });
}
