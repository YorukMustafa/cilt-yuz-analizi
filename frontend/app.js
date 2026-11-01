// Akıllı API Yönlendirme: Spring Boot (8080) varsa oraya gider, yoksa doğrudan Python (8000)'e bağlanır
let API_BASE_URL = window.location.port === "8080" 
    ? "http://localhost:8080/api/v1" 
    : "http://localhost:8000/api";

let selectedFile = null;

// DOM Elemanları
const dropZone = document.getElementById("dropZone");
const dropZoneContent = document.getElementById("dropZoneContent");
const fileInput = document.getElementById("fileInput");
const imagePreview = document.getElementById("imagePreview");
const analyzeBtn = document.getElementById("analyzeBtn");
const resetBtn = document.getElementById("resetBtn");
const btnSpinner = document.getElementById("btnSpinner");
const apiStatusBadge = document.getElementById("apiStatusBadge");

// Sonuç Alanı DOM Elemanları
const emptyState = document.getElementById("emptyState");
const resultsContent = document.getElementById("resultsContent");
const overallScore = document.getElementById("overallScore");
const statusEvaluation = document.getElementById("statusEvaluation");
const spotCount = document.getElementById("spotCount");
const spotSeverity = document.getElementById("spotSeverity");
const spotBar = document.getElementById("spotBar");
const wrinkleScore = document.getElementById("wrinkleScore");
const smoothnessScore = document.getElementById("smoothnessScore");
const wrinkleBar = document.getElementById("wrinkleBar");
const haralickContrast = document.getElementById("haralickContrast");
const haralickHomogeneity = document.getElementById("haralickHomogeneity");
const haralickEnergy = document.getElementById("haralickEnergy");
const annotatedImage = document.getElementById("annotatedImage");

// 1. Akıllı Sağlık & Bağlantı Kontrolü
async function checkApiHealth() {
    // Önce Spring Boot'u dene (Port 8080)
    try {
        const springRes = await fetch("http://localhost:8080/api/v1/health");
        if (springRes.ok) {
            API_BASE_URL = "http://localhost:8080/api/v1";
            apiStatusBadge.textContent = "● Spring Boot + DB Bağlı (Port 8080)";
            apiStatusBadge.classList.add("connected");
            return;
        }
    } catch (_) {}

    // Spring Boot çalışmıyorsa doğrudan Python servisini kontrol et (Port 8000)
    try {
        const pythonRes = await fetch("http://localhost:8000/health");
        if (pythonRes.ok) {
            API_BASE_URL = "http://localhost:8000/api";
            apiStatusBadge.textContent = "● Python AI Motoru Bağlı (Port 8000)";
            apiStatusBadge.classList.add("connected");
            return;
        }
    } catch (_) {}

    apiStatusBadge.textContent = "○ Sunucu Bekleniyor (Spring Boot / Python)...";
    apiStatusBadge.classList.remove("connected");
}

setInterval(checkApiHealth, 5000);
checkApiHealth();

// 2. Dosya Seçimi ve Sürükle-Bırak Olayları
dropZone.addEventListener("click", () => fileInput.click());

fileInput.addEventListener("change", (e) => {
    if (e.target.files && e.target.files[0]) {
        handleFileSelect(e.target.files[0]);
    }
});

dropZone.addEventListener("dragover", (e) => {
    e.preventDefault();
    dropZone.classList.add("dragover");
});

dropZone.addEventListener("dragleave", () => {
    dropZone.classList.remove("dragover");
});

dropZone.addEventListener("drop", (e) => {
    e.preventDefault();
    dropZone.classList.remove("dragover");
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
        handleFileSelect(e.dataTransfer.files[0]);
    }
});

function handleFileSelect(file) {
    if (!file.type.startsWith("image/")) {
        alert("Lütfen geçerli bir resim dosyası seçiniz.");
        return;
    }
    selectedFile = file;

    const reader = new FileReader();
    reader.onload = (e) => {
        imagePreview.src = e.target.result;
        imagePreview.style.display = "block";
        dropZoneContent.style.display = "none";
        analyzeBtn.disabled = false;
        resetBtn.style.display = "inline-flex";
    };
    reader.readAsDataURL(file);
}

// 3. Formu Sıfırlama
resetBtn.addEventListener("click", () => {
    selectedFile = null;
    fileInput.value = "";
    imagePreview.src = "";
    imagePreview.style.display = "none";
    dropZoneContent.style.display = "block";
    analyzeBtn.disabled = true;
    resetBtn.style.display = "none";
    emptyState.style.display = "block";
    resultsContent.style.display = "none";
});

// 4. Analiz İsteği Gönderme
analyzeBtn.addEventListener("click", async () => {
    if (!selectedFile) return;

    analyzeBtn.disabled = true;
    btnSpinner.style.display = "inline-block";
    const btnText = analyzeBtn.querySelector(".btn-text");
    btnText.textContent = "Analiz Ediliyor...";

    const formData = new FormData();
    formData.append("file", selectedFile);

    try {
        const response = await fetch(`${API_BASE_URL}/analyze`, {
            method: "POST",
            body: formData,
        });

        if (!response.ok) {
            const errData = await response.json();
            throw new Error(errData.error || errData.detail || "Sunucu hatası oluştu.");
        }

        const data = await response.json();
        renderResults(data);

    } catch (err) {
        alert(`Analiz başarısız: ${err.message}\nLütfen servislerin çalıştığından emin olunuz.`);
    } finally {
        analyzeBtn.disabled = false;
        btnSpinner.style.display = "none";
        btnText.textContent = "Yeniden Analiz Et";
    }
});

// 5. Sonuçları Ekrana Yazdırma
function renderResults(data) {
    emptyState.style.display = "none";
    resultsContent.style.display = "block";

    // Genel Puan ve Yorum
    overallScore.textContent = data.overall_health_score;
    statusEvaluation.textContent = data.status_evaluation;

    // Leke Metrikleri
    const blemish = data.blemish_analysis;
    spotCount.textContent = blemish.detected_spots_count;
    spotSeverity.textContent = `${blemish.blemish_severity_score} / 100`;
    spotBar.style.width = `${Math.min(100, blemish.blemish_severity_score)}%`;

    // Kırışıklık Metrikleri
    const texture = data.texture_and_wrinkle_analysis;
    wrinkleScore.textContent = `${texture.wrinkle_score}`;
    smoothnessScore.textContent = `${texture.smoothness_score}%`;
    wrinkleBar.style.width = `${Math.min(100, texture.wrinkle_score)}%`;

    // Haralick Doku Değerleri (Akademik Rapor Metrikleri)
    haralickContrast.textContent = texture.haralick_contrast;
    haralickHomogeneity.textContent = texture.haralick_homogeneity;
    haralickEnergy.textContent = texture.haralick_energy;

    // İşaretlenmiş Fotoğraf (Base64)
    annotatedImage.src = data.annotated_image_base64;
}
