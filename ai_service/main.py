"""
Yüz ve Cilt Analizi - Python FastAPI Mikroservisi
-------------------------------------------------
Spring Boot veya Web Arayüzü tarafından gönderilen fotoğrafları karşılar,
görüntü işleme & yapay zeka analizini çalıştırır ve JSON rapor üretir.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from analyzer import SkinAnalyzer
import uvicorn

app = FastAPI(
    title="Face & Skin AI Analysis Service",
    description="Yapay Zeka ve Görüntü İşleme Tabanlı Cilt Analiz API",
    version="1.0.0"
)

# Web arayüzünün doğrudan veya Spring Boot üzerinden erişebilmesi için CORS izinleri
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Analiz motoru nesnesini başlat
analyzer = SkinAnalyzer()


@app.get("/health")
def health_check():
    return {"status": "UP", "engine": "OpenCV + GLCM + AI"}


@app.post("/api/analyze")
async def analyze_skin(file: UploadFile = File(...)):
    """
    Kullanıcının yüklediği yüz görselini alır, leke ve doku analizi yapar.
    """
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Lütfen geçerli bir resim dosyası (.jpg, .png) yükleyin.")

    try:
        image_bytes = await file.read()
        results = analyzer.analyze(image_bytes)
        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analiz sırasında hata oluştu: {str(e)}")


# Web arayüzünü (Frontend) doğrudan http://localhost:8000 adresinde sun
frontend_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "frontend"))
if os.path.exists(frontend_dir):
    app.mount("/", StaticFiles(directory=frontend_dir, html=True), name="frontend")



if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
