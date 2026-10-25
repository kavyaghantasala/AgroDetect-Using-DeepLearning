/**
 * AgroDetect - Interactive Frontend Controller
 * Handles image selection, drag-and-drop, API communication,
 * results visualization, and real-time multilingual switching.
 */

document.addEventListener("DOMContentLoaded", () => {
    // Current application state
    let currentLanguage = "en";
    let selectedFile = null;
    let latestPredictionData = null;

    // DOM Elements
    const dropzone = document.getElementById("dropzone");
    const fileInput = document.getElementById("fileInput");
    const previewContainer = document.getElementById("previewContainer");
    const previewImage = document.getElementById("previewImage");
    const previewFilename = document.getElementById("previewFilename");
    const previewFilesize = document.getElementById("previewFilesize");
    const btnChangeImage = document.getElementById("btnChangeImage");
    const actionContainer = document.getElementById("actionContainer");
    const btnAnalyze = document.getElementById("btnAnalyze");
    const btnAnalyzeText = document.getElementById("btnAnalyzeText");
    const btnSpinner = document.getElementById("btnSpinner");
    const alertBox = document.getElementById("alertBox");
    const resultsContainer = document.getElementById("resultsContainer");
    const btnReset = document.getElementById("btnReset");
    const langSelect = document.getElementById("langSelect");

    // =========================================================================
    // Multilingual Localization
    // =========================================================================
    function getLocalizedClassInfo(className, lang) {
        if (window.DISEASE_NAME_TRANSLATIONS && window.DISEASE_NAME_TRANSLATIONS[className]) {
            const entry = window.DISEASE_NAME_TRANSLATIONS[className];
            const disp = entry[lang] || entry["en"] || className;
            let crop = "";
            let cond = disp;
            if (disp.includes(" - ")) {
                const parts = disp.split(" - ");
                crop = parts[0];
                cond = parts[1];
            }
            return { display_name: disp, crop: crop, condition: cond };
        }
        return { display_name: className, crop: "Plant", condition: className };
    }

    function applyLanguage(lang) {
        if (!UI_TRANSLATIONS[lang]) return;
        currentLanguage = lang;
        const dict = UI_TRANSLATIONS[lang];

        // Update elements with data-i18n attribute
        document.querySelectorAll("[data-i18n]").forEach(elem => {
            const key = elem.getAttribute("data-i18n");
            if (dict[key]) {
                elem.textContent = dict[key];
            }
        });

        // Update dropzone browse text if not custom
        if (dict.browseBtn) {
            const browseBtn = document.getElementById("btnBrowseText");
            if (browseBtn) browseBtn.textContent = dict.browseBtn;
        }

        // If prediction results are currently displayed, refresh localized disease info, crop, condition, and top predictions!
        if (latestPredictionData && latestPredictionData.predicted_class) {
            const localized = getLocalizedClassInfo(latestPredictionData.predicted_class, lang);
            document.getElementById("resCropName").textContent = localized.crop;
            document.getElementById("resConditionName").textContent = localized.condition;

            const topPredsContainer = document.getElementById("resTopPreds");
            topPredsContainer.innerHTML = "";
            if (latestPredictionData.top_predictions && latestPredictionData.top_predictions.length > 0) {
                latestPredictionData.top_predictions.forEach(item => {
                    const itemLoc = getLocalizedClassInfo(item.class_name, lang);
                    const row = document.createElement("div");
                    row.className = "top-pred-row";
                    row.innerHTML = `
                        <span class="top-pred-name">${escapeHtml(itemLoc.display_name)}</span>
                        <span class="top-pred-badge">${item.confidence.toFixed(1)}%</span>
                    `;
                    topPredsContainer.appendChild(row);
                });
            }

            refreshLocalizedDiseaseInfo(latestPredictionData.predicted_class, lang);
        }
    }

    if (langSelect) {
        langSelect.addEventListener("change", (e) => {
            applyLanguage(e.target.value);
        });
    }

    // =========================================================================
    // File Selection & Drag-and-Drop
    // =========================================================================
    dropzone.addEventListener("click", () => fileInput.click());

    dropzone.addEventListener("dragover", (e) => {
        e.preventDefault();
        dropzone.classList.add("dragover");
    });

    dropzone.addEventListener("dragleave", () => {
        dropzone.classList.remove("dragover");
    });

    dropzone.addEventListener("drop", (e) => {
        e.preventDefault();
        dropzone.classList.remove("dragover");
        if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
            handleFileSelection(e.dataTransfer.files[0]);
        }
    });

    fileInput.addEventListener("change", (e) => {
        if (e.target.files && e.target.files.length > 0) {
            handleFileSelection(e.target.files[0]);
        }
    });

    btnChangeImage.addEventListener("click", () => {
        fileInput.value = "";
        fileInput.click();
    });

    btnReset.addEventListener("click", () => {
        resetAnalysisState();
    });

    function showAlert(message, isWarning = false) {
        alertBox.textContent = message;
        alertBox.className = isWarning ? "alert alert-warning" : "alert alert-danger";
        alertBox.style.display = "block";
        alertBox.scrollIntoView({ behavior: "smooth", block: "center" });
    }

    function hideAlert() {
        alertBox.style.display = "none";
    }

    function handleFileSelection(file) {
        hideAlert();
        const dict = UI_TRANSLATIONS[currentLanguage] || UI_TRANSLATIONS.en;

        // Validation 1: Allowed MIME / extension
        const validExtensions = [".jpg", ".jpeg", ".png"];
        const fileName = file.name.toLowerCase();
        const isValidExt = validExtensions.some(ext => fileName.endsWith(ext));

        if (!isValidExt) {
            showAlert(dict.errorInvalidFormat || "Please upload JPG, JPEG, or PNG images only.");
            return;
        }

        // Validation 2: Maximum file size (10 MB)
        if (file.size > 10 * 1024 * 1024) {
            showAlert(dict.errorFileTooBig || "File size exceeds 10 MB limit.");
            return;
        }

        selectedFile = file;

        // Display Preview
        const reader = new FileReader();
        reader.onload = (e) => {
            previewImage.src = e.target.result;
            previewFilename.textContent = file.name;
            const sizeInMb = (file.size / (1024 * 1024)).toFixed(2);
            previewFilesize.textContent = `${sizeInMb} MB`;

            dropzone.style.display = "none";
            previewContainer.style.display = "block";
            actionContainer.style.display = "block";
            resultsContainer.style.display = "none";
        };
        reader.readAsDataURL(file);
    }

    function resetAnalysisState() {
        selectedFile = null;
        latestPredictionData = null;
        fileInput.value = "";
        dropzone.style.display = "block";
        previewContainer.style.display = "none";
        actionContainer.style.display = "none";
        resultsContainer.style.display = "none";
        hideAlert();
        dropzone.scrollIntoView({ behavior: "smooth", block: "center" });
    }

    // =========================================================================
    // Predict API Request
    // =========================================================================
    btnAnalyze.addEventListener("click", async () => {
        if (!selectedFile) {
            const dict = UI_TRANSLATIONS[currentLanguage] || UI_TRANSLATIONS.en;
            showAlert(dict.errorNoFile || "Please select an image file first.");
            return;
        }

        hideAlert();
        setLoadingState(true);

        const formData = new FormData();
        formData.append("file", selectedFile);

        try {
            const response = await fetch(`/predict?lang=${encodeURIComponent(currentLanguage)}`, {
                method: "POST",
                body: formData
            });

            const data = await response.json();

            if (!response.ok) {
                // If model is missing (503 Service Unavailable)
                if (response.status === 503) {
                    showAlert(
                        data.detail + (data.hint ? ` (${data.hint})` : ""),
                        true
                    );
                } else {
                    showAlert(data.detail || "Prediction request failed.");
                }
                setLoadingState(false);
                return;
            }

            latestPredictionData = data;
            renderPredictionResults(data);
            setLoadingState(false);

        } catch (err) {
            console.error("Prediction error:", err);
            const dict = UI_TRANSLATIONS[currentLanguage] || UI_TRANSLATIONS.en;
            showAlert(dict.errorPredictionFailed || "Network error. Failed to reach the prediction service.");
            setLoadingState(false);
        }
    });

    function setLoadingState(isLoading) {
        const dict = UI_TRANSLATIONS[currentLanguage] || UI_TRANSLATIONS.en;
        btnAnalyze.disabled = isLoading;
        if (isLoading) {
            btnSpinner.style.display = "inline-block";
            btnAnalyzeText.textContent = dict.btnAnalyzing || "Analyzing with ResNet50...";
        } else {
            btnSpinner.style.display = "none";
            btnAnalyzeText.textContent = dict.btnAnalyze || "Analyze Leaf Condition";
        }
    }

    // =========================================================================
    // Render Results
    // =========================================================================
    function renderPredictionResults(data) {
        const dict = UI_TRANSLATIONS[currentLanguage] || UI_TRANSLATIONS.en;

        // 1. Status Badge
        const statusBadge = document.getElementById("resStatusBadge");
        if (data.is_healthy) {
            statusBadge.className = "status-badge healthy";
            statusBadge.textContent = dict.statusHealthy || "Healthy Foliage";
        } else {
            statusBadge.className = "status-badge diseased";
            statusBadge.textContent = dict.statusDiseased || "Disease Detected";
        }

        // 2. Crop & Condition (localized in current language)
        const primaryLoc = getLocalizedClassInfo(data.predicted_class, currentLanguage);
        document.getElementById("resCropName").textContent = primaryLoc.crop || data.crop;
        document.getElementById("resConditionName").textContent = primaryLoc.condition || data.condition;

        // 3. Confidence meter
        const confPercent = Math.min(100, Math.max(0, data.confidence));
        document.getElementById("resConfidenceVal").textContent = `${confPercent.toFixed(1)}%`;
        const progressBar = document.getElementById("resConfidenceBar");
        progressBar.style.width = "0%";
        setTimeout(() => {
            progressBar.style.width = `${confPercent}%`;
        }, 100);

        // 4. Top-3 Predictions (localized in current language)
        const topPredsContainer = document.getElementById("resTopPreds");
        topPredsContainer.innerHTML = "";
        if (data.top_predictions && data.top_predictions.length > 0) {
            data.top_predictions.forEach(item => {
                const itemLoc = getLocalizedClassInfo(item.class_name, currentLanguage);
                const row = document.createElement("div");
                row.className = "top-pred-row";
                row.innerHTML = `
                    <span class="top-pred-name">${escapeHtml(itemLoc.display_name || item.display_name)}</span>
                    <span class="top-pred-badge">${item.confidence.toFixed(1)}%</span>
                `;
                topPredsContainer.appendChild(row);
            });
        }

        // 5. Populate Symptoms & Prevention Advice
        renderAdviceContent(data.disease_info);

        resultsContainer.style.display = "block";
        resultsContainer.scrollIntoView({ behavior: "smooth", block: "start" });
    }

    function renderAdviceContent(info) {
        const symptomsList = document.getElementById("resSymptomsList");
        const preventList = document.getElementById("resPreventList");
        const manageList = document.getElementById("resManageList");

        symptomsList.innerHTML = "";
        preventList.innerHTML = "";
        manageList.innerHTML = "";

        if (info) {
            // Symptoms
            if (info.symptoms && info.symptoms.length > 0) {
                info.symptoms.forEach(s => {
                    const li = document.createElement("li");
                    li.textContent = s;
                    symptomsList.appendChild(li);
                });
            } else {
                symptomsList.innerHTML = "<li>No specific visual lesions noted.</li>";
            }

            // Preventive Measures
            if (info.preventive_measures && info.preventive_measures.length > 0) {
                info.preventive_measures.forEach(p => {
                    const li = document.createElement("li");
                    li.textContent = p;
                    preventList.appendChild(li);
                });
            } else {
                preventList.innerHTML = "<li>Maintain regular agricultural sanitation and inspection.</li>";
            }

            // General Management
            if (info.general_management && info.general_management.length > 0) {
                info.general_management.forEach(m => {
                    const li = document.createElement("li");
                    li.textContent = m;
                    manageList.appendChild(li);
                });
            }
        }
    }

    async function refreshLocalizedDiseaseInfo(className, lang) {
        try {
            const res = await fetch(`/disease/${encodeURIComponent(className)}?lang=${encodeURIComponent(lang)}`);
            if (res.ok) {
                const info = await res.json();
                renderAdviceContent(info);
            }
        } catch (e) {
            console.warn("Could not refresh localized disease info:", e);
        }
    }

    function escapeHtml(text) {
        const div = document.createElement("div");
        div.textContent = text;
        return div.innerHTML;
    }
});
