"""
Yapay Zeka ve Görüntü İşleme Tabanlı Yüz ve Cilt Analiz Motoru
---------------------------------------------------------------
Algoritmalar:
1. Cilt Segmentasyonu (HSV + YCrCb Hibrit Maskeleme)
2. Leke / Hiperpigmentasyon Tespiti (Color Thresholding + Morfoloji)
3. Kırışıklık ve Doku Analizi (GLCM Haralick Öznitelikleri: Kontrast, Homojenlik, Enerji)
4. Görselleştirme (Leke ve Kırışıklık Haritası Çıkarımı)
"""

import cv2
import numpy as np
import base64
from skimage.feature import graycomatrix, graycoprops


class SkinAnalyzer:
    def __init__(self):
        # Yüz tespiti için Haar Cascade sınıflandırıcısı
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )

    def decode_image(self, image_bytes: bytes) -> np.ndarray:
        """Bayt dizisinden OpenCV formatında (BGR) görsel okur."""
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("Görsel çözümlenemedi. Geçerli bir resim formatı yükleyiniz.")
        return img

    def extract_skin_mask(self, img_bgr: np.ndarray) -> np.ndarray:
        """
        Işık değişimlerine dayanıklı HSV ve YCrCb renk uzaylarını
        birlikte kullanarak cilt piksellerini izole eder.
        """
        # HSV uzayında cilt renk eşikleri
        hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
        lower_hsv = np.array([0, 30, 60], dtype=np.uint8)
        upper_hsv = np.array([25, 180, 255], dtype=np.uint8)
        mask_hsv = cv2.inRange(hsv, lower_hsv, upper_hsv)

        # YCrCb uzayında cilt renk eşikleri (ten rengi ayrımında çok başarılıdır)
        ycrcb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2YCrCb)
        lower_ycrcb = np.array([0, 133, 77], dtype=np.uint8)
        upper_ycrcb = np.array([255, 173, 127], dtype=np.uint8)
        mask_ycrcb = cv2.inRange(ycrcb, lower_ycrcb, upper_ycrcb)

        # İki maskenin kesişimi
        skin_mask = cv2.bitwise_and(mask_hsv, mask_ycrcb)

        # Morfolojik temizleme (küçük gürültüleri yok et)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_OPEN, kernel, iterations=2)
        skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_DILATE, kernel, iterations=1)
        return skin_mask

    def detect_blemishes(self, img_bgr: np.ndarray, skin_mask: np.ndarray):
        """
        Renk eşikleme ve adaptif kontrast ile ciltteki lekeleri
        (kızarıklık, hiperpigmentasyon, çil/güneş lekesi) tespit eder.
        """
        # LAB renk uzayına geçip L (parlaklık) kanalına CLAHE uygula
        lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        cl = clahe.apply(l)
        enhanced_lab = cv2.merge((cl, a, b))
        enhanced_bgr = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)

        # Griye çevirip cilt bölgesini al
        gray = cv2.cvtColor(enhanced_bgr, cv2.COLOR_BGR2GRAY)
        skin_gray = cv2.bitwise_and(gray, gray, mask=skin_mask)

        # Adaptif eşikleme ile leke ve gölge kenarlarını çıkar
        blurred = cv2.GaussianBlur(skin_gray, (7, 7), 0)
        thresh = cv2.adaptiveThreshold(
            blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 15, 3
        )

        # Yalnızca cilt maskesi içindeki lekeleri tut
        thresh = cv2.bitwise_and(thresh, thresh, mask=skin_mask)

        # Çok küçük (gürültü) ve çok büyük (saç, kaş) alanları filtrele
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        valid_blemishes = []
        blemish_mask = np.zeros_like(thresh)

        skin_pixel_count = cv2.countNonZero(skin_mask) + 1e-5

        for cnt in contours:
            area = cv2.contourArea(cnt)
            # 8 piksel ile 400 piksel arasındaki lekeler cilt lekesidir
            if 8 < area < 400:
                valid_blemishes.append(cnt)
                cv2.drawContours(blemish_mask, [cnt], -1, 255, -1)

        total_blemish_area = sum([cv2.contourArea(c) for c in valid_blemishes])
        blemish_ratio = float((total_blemish_area / skin_pixel_count) * 100)

        # 0 - 100 arası leke skoru (0 = tamamen lekesiz, 100 = çok yoğun lekeli)
        blemish_score = min(100.0, round(blemish_ratio * 15, 2))

        return {
            "blemish_count": len(valid_blemishes),
            "blemish_area_ratio": round(blemish_ratio, 3),
            "blemish_score": blemish_score,
            "contours": valid_blemishes,
            "mask": blemish_mask
        }

    def analyze_wrinkles_glcm(self, img_bgr: np.ndarray, skin_mask: np.ndarray):
        """
        GLCM (Gri Seviye Eş-Oluşum Matrisi) ile Haralick Doku Analizi yapar.
        Kırışıklıklar doku kontrastını yükseltirken, homojenliği ve enerjiyi düşürür.
        """
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        
        # Cilt piksellerini al
        skin_pixels = gray[skin_mask > 0]
        if len(skin_pixels) < 1000:
            # Maske çok küçükse tüm gri görüntüyü kullan
            roi_gray = gray
        else:
            # Görüntü boyutunu optimize et
            roi_gray = cv2.resize(gray, (256, 256))

        # 16 seviyeli kuantizasyon (hesaplama hızını ve gürültü toleransını artırır)
        quantized = (roi_gray // 16).astype(np.uint8)

        # GLCM matrisini 4 yönde (0, 45, 90, 135 derece) hesapla
        glcm = graycomatrix(
            quantized,
            distances=[1, 2],
            angles=[0, np.pi/4, np.pi/2, 3*np.pi/4],
            levels=16,
            symmetric=True,
            normed=True
        )

        # Haralick doku metriklerini çıkar
        contrast = float(graycoprops(glcm, 'contrast').mean())
        homogeneity = float(graycoprops(glcm, 'homogeneity').mean())
        energy = float(graycoprops(glcm, 'energy').mean())
        correlation = float(graycoprops(glcm, 'correlation').mean())

        # Kırışıklık kenarlarını Canny filtresiyle doğrula
        edges = cv2.Canny(roi_gray, 50, 150)
        edge_density = float(np.sum(edges > 0) / (edges.shape[0] * edges.shape[1]))

        # Normalize kırışıklık puanı (0: genç pürüzsüz doku, 100: yoğun kırışıklık/çizgi)
        # Kontrast ve kenar yoğunluğu arttıkça kırışıklık skoru artar
        wrinkle_score = min(100.0, max(0.0, round((contrast * 12.0) + (edge_density * 400.0), 2)))
        texture_smoothness = min(100.0, max(0.0, round(homogeneity * 100, 2)))

        return {
            "glcm_contrast": round(contrast, 3),
            "glcm_homogeneity": round(homogeneity, 3),
            "glcm_energy": round(energy, 3),
            "glcm_correlation": round(correlation, 3),
            "wrinkle_score": wrinkle_score,
            "texture_smoothness_score": texture_smoothness
        }

    def generate_annotated_image(self, img_bgr: np.ndarray, blemish_contours, skin_mask: np.ndarray) -> str:
        """
        Tespit edilen lekeleri ve analiz bölgelerini orijinal resim üzerine
        işaretleyerek Base64 JPEG formatında döndürür (Web arayüzünde göstermek için).
        """
        output = img_bgr.copy()

        # Lekeleri kırmızı daire ve konturlarla çiz
        for cnt in blemish_contours:
            (x, y), radius = cv2.minEnclosingCircle(cnt)
            center = (int(x), int(y))
            radius = max(3, int(radius))
            # Kırmızı çember
            cv2.circle(output, center, radius + 2, (0, 0, 255), 1)

        # Cilt alanına hafif saydam yeşil bir katman ekle (analiz edilen bölgeyi gösterir)
        overlay = output.copy()
        overlay[skin_mask > 0] = cv2.addWeighted(
            output[skin_mask > 0], 0.85, np.full_like(output[skin_mask > 0], (0, 220, 100)), 0.15, 0
        )
        output = cv2.addWeighted(overlay, 0.4, output, 0.6, 0)

        # JPEG encode ve base64 dönüşümü
        _, buffer = cv2.imencode('.jpg', output, [cv2.IMWRITE_JPEG_QUALITY, 85])
        base64_str = base64.b64encode(buffer).decode('utf-8')
        return f"data:image/jpeg;base64,{base64_str}"

    def analyze(self, image_bytes: bytes) -> dict:
        """Tüm analiz adımlarını birleştiren ana metot."""
        img_bgr = self.decode_image(image_bytes)

        # 1. Cilt Maskesini Çıkar
        skin_mask = self.extract_skin_mask(img_bgr)

        # 2. Leke Analizi (Color Thresholding)
        blemish_res = self.detect_blemishes(img_bgr, skin_mask)

        # 3. Kırışıklık ve Doku Analizi (GLCM)
        texture_res = self.analyze_wrinkles_glcm(img_bgr, skin_mask)

        # 4. Genel Cilt Sağlık Puanı (100 üzerinden; yüksek puan daha sağlıklı cilt)
        health_score = max(0.0, min(100.0, round(
            100.0 - (blemish_res["blemish_score"] * 0.5 + texture_res["wrinkle_score"] * 0.5), 1
        )))

        # 5. Cilt Durum Yorumu
        if health_score >= 80:
            status_text = "Çok İyi: Cilt dokusu pürüzsüz ve leke yoğunluğu düşük seviyede."
        elif health_score >= 60:
            status_text = "İyi: Belirli bölgelerde hafif lekeler ve ince çizgiler mevcut."
        else:
            status_text = "Bakım Gerektirir: Doku pürüzlülüğü veya leke yoğunluğu yüksek seviyede."

        # 6. İşaretli Görsel Çıktısı
        annotated_b64 = self.generate_annotated_image(img_bgr, blemish_res["contours"], skin_mask)

        return {
            "status": "success",
            "overall_health_score": health_score,
            "status_evaluation": status_text,
            "blemish_analysis": {
                "detected_spots_count": blemish_res["blemish_count"],
                "blemish_density_percentage": blemish_res["blemish_area_ratio"],
                "blemish_severity_score": blemish_res["blemish_score"]
            },
            "texture_and_wrinkle_analysis": {
                "wrinkle_score": texture_res["wrinkle_score"],
                "smoothness_score": texture_res["texture_smoothness_score"],
                "haralick_contrast": texture_res["glcm_contrast"],
                "haralick_homogeneity": texture_res["glcm_homogeneity"],
                "haralick_energy": texture_res["glcm_energy"]
            },
            "annotated_image_base64": annotated_b64
        }
