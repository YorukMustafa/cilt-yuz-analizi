# Yapay Zeka ve Görüntü İşleme Tabanlı Web Destekli Yüz ve Cilt Analizi Sistemi

Bu proje, Tekirdağ Namık Kemal Üniversitesi Bilgisayar Mühendisliği Bölümü **Proje I ve II** dersleri kapsamında geliştirilmektedir.

---

## 🎯 Projenin Amacı ve Özellikleri
Kullanıcıların web tarayıcısı üzerinden yükledikleri yüz fotoğrafları üzerinde hibrit görüntü işleme algoritmaları çalıştırarak:
1. **Leke ve Hiperpigmentasyon Tespiti:** HSV ve YCrCb renk uzaylarında adaptif renk eşikleme (Color Thresholding) ve morfolojik filtreleme.
2. **Kırışıklık ve Doku Analizi:** Gri Seviye Eş-Oluşum Matrisi (GLCM) ile Haralick doku metriklerinin (Kontrast, Homojenlik, Enerji, Korelasyon) ve kenar yoğunluğunun hesaplanması.
3. **Java Spring Boot Backend:** Kurumsal düzeyde dosya doğrulama, RESTful mikroservis haberleşmesi (`WebClient`) ve analiz geçmişinin H2 veritabanında saklanması.
4. **Genel Cilt Sağlık İndeksi:** Sayısal skorlar (0-100) ve fotoğrafta tespit edilen sorunlu alanların görsel olarak işaretlenmesi (Annotated Heatmap).

---

## 🏗️ Sistem Mimarisi

```
face-skin-analysis/
│
├── ai_service/             # 1. Python FastAPI Yapay Zeka & Görüntü İşleme Motoru (Port: 8000)
│   ├── analyzer.py         # Çekirdek algoritma (OpenCV + GLCM + Leke/Kırışıklık)
│   ├── main.py             # REST API sunucusu (FastAPI + Uvicorn)
│   ├── test_local.py       # Bağımsız test betiği
│   └── requirements.txt    # Gerekli Python kütüphaneleri
│
├── backend/                # 2. Java Spring Boot Kurumsal Katmanı (Port: 8080)
│   ├── pom.xml             # Maven bağımlılıkları (Spring Web, WebFlux, Data JPA, H2)
│   └── src/main/java/com/dermai/skinanalysis/
│       ├── SkinAnalysisApplication.java
│       ├── controller/     # SkinAnalysisController (REST Endpoints)
│       ├── service/        # SkinAnalysisService (Python Entegrasyonu & DB Kayıt)
│       ├── repository/     # AnalysisRecordRepository (JPA)
│       ├── model/          # AnalysisRecord (Veritabanı Varlığı)
│       └── config/         # WebClient & CORS Konfigürasyonu
│
├── frontend/               # 3. Bağımsız Web Arayüzü (HTML5 + CSS3 + Modern JS)
│   ├── index.html
│   ├── style.css
│   └── app.js
│
└── zotero_kaynaklar.ris    # Akademik Rapor Kaynakçası (Zotero APA 7 Import Dosyası)
```

---

## 🚀 Sistemi Çalıştırma Adımları

### 1. Adım: Python AI Servisini Başlatma
1. Bir PowerShell veya CMD terminali açın:
   ```powershell
   cd ai_service
   pip install -r requirements.txt
   uvicorn main:app --reload --port 8000
   ```
   *Yapay zeka analiz motoru `http://localhost:8000` adresinde hazır hale gelecektir.*

### 2. Adım: Java Spring Boot Backend'i Başlatma
1. Yeni bir terminal açın:
   ```powershell
   cd backend
   mvn spring-boot:run
   ```
   *(veya IntelliJ IDEA / Eclipse / VS Code ile `backend` klasörünü açıp `SkinAnalysisApplication.java` dosyasını çalıştırın).*
   *Spring Boot sunucusu `http://localhost:8080` adresinde çalışacaktır.*

### 3. Adım: Web Sitesini Açma
- Tarayıcınızda doğrudan `http://localhost:8080` adresine gidin.
- Yüz fotoğrafınızı sürükleyip bırakarak **"Analizi Başlat"** butonuna basın.
- Sistem analiz yapacak, lekeleri ve kırışıklıkları hesaplayacak, görseli işaretleyecek ve sonucu otomatik olarak H2 veritabanına kaydedecektir!
- Veritabanındaki geçmiş kayıtları incelemek için: `http://localhost:8080/h2-console`
