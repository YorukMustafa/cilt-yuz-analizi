"""
Yapay Zeka ve Görüntü İşleme Tabanlı Yüz ve Cilt Analiz Motoru
---------------------------------------------------------------
Gelişmiş Algoritmalar:
1. İç Yüz Segmentasyonu (Kulak, kıyafet, yaka ve arka planı filtreleyen eliptik ROI)
2. Koyu Özellik Maskeleme (Göz, kaş, sakal, bıyık ve saç diplerini leke analizinden çıkarma)
3. Dermatolojik Egzema & Kızarıklık Tespiti (Erythema İndeksi: R - G fark analizi)
4. Melanin & Hiperpigmentasyon Tespiti (Lokal kontrast parlaklık analizi)
5. Alın & Yanak GLCM Haralick Doku / Kırışıklık Analizi
"""

import cv2
import numpy as np
import base64

try:
    from skimage.feature import graycomatrix, graycoprops
    SKIMAGE_AVAILABLE = True
except ImportError:
    SKIMAGE_AVAILABLE = False


def _numpy_glcm_features(quantized, levels=16):
    """NumPy tabanlı hızlı Haralick Doku (GLCM) hesaplayıcı (skimage yedeği)."""
    h, w = quantized.shape
    pairs = [
        (quantized[:, :-1], quantized[:, 1:]),                # 0 derece (yatay)
        (quantized[:-1, :-1], quantized[1:, 1:]),            # 45 derece (çapraz)
        (quantized[:-1, :], quantized[1:, :]),                # 90 derece (dikey)
        (quantized[:-1, 1:], quantized[1:, :-1])             # 135 derece (ters çapraz)
    ]
    
    contrasts, homogeneities, energies, correlations = [], [], [], []
    
    for p1, p2 in pairs:
        f1 = p1.ravel()
        f2 = p2.ravel()
        mat = np.zeros((levels, levels), dtype=np.float64)
        np.add.at(mat, (f1, f2), 1)
        mat = mat + mat.T
        total = mat.sum()
        if total > 0:
            mat /= total
            
        i, j = np.indices((levels, levels))
        diff = i - j
        contrast = np.sum((diff ** 2) * mat)
        homogeneity = np.sum(mat / (1.0 + np.abs(diff)))
        energy = np.sum(mat ** 2)
        
        mi = np.sum(i * mat)
        mj = np.sum(j * mat)
        si = np.sqrt(np.sum(((i - mi) ** 2) * mat))
        sj = np.sqrt(np.sum(((j - mj) ** 2) * mat))
        correlation = np.sum((i - mi) * (j - mj) * mat) / (si * sj + 1e-8)
        
        contrasts.append(contrast)
        homogeneities.append(homogeneity)
        energies.append(energy)
        correlations.append(correlation)
        
    return {
        "contrast": float(np.mean(contrasts)),
        "homogeneity": float(np.mean(homogeneities)),
        "energy": float(np.mean(energies)),
        "correlation": float(np.mean(correlations))
    }


class SkinAnalyzer:
    def __init__(self):
        self.face_cascade = None

    def decode_image(self, image_bytes: bytes) -> np.ndarray:
        """Bayt dizisinden OpenCV formatında (BGR) görsel okur."""
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("Görsel çözümlenemedi. Geçerli bir resim formatı yükleyiniz.")
        
        # Çok büyük fotoğrafları analiz performansı ve tutarlılık için normalize et
        max_dim = 1200
        h, w = img.shape[:2]
        if max(h, w) > max_dim:
            scale = max_dim / max(h, w)
            img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
            
        return img

    def extract_face_and_skin_mask(self, img_bgr: np.ndarray):
        """
        Kulakları, kazak/yakayı, arka planı, saçları ve sakalı dışarıda bırakarak
        YALNIZCA iç yüz bölgesini (alın, yanaklar, burun, çene) izole eder.
        """
        h, w = img_bgr.shape[:2]
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

        # 1. Ten rengi tespiti (YCrCb + HSV)
        ycrcb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2YCrCb)
        skin_ycrcb = cv2.inRange(ycrcb, np.array([55, 133, 80]), np.array([245, 175, 128]))

        hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
        skin_hsv = cv2.inRange(hsv, np.array([0, 25, 45]), np.array([35, 210, 255]))

        raw_skin = cv2.bitwise_and(skin_ycrcb, skin_hsv)

        # 2. Morfolojik temizleme
        k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        skin_closed = cv2.morphologyEx(raw_skin, cv2.MORPH_CLOSE, k_close)

        # 3. Ana baş konturunu bul
        contours, _ = cv2.findContours(skin_closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)

        if contours and cv2.contourArea(contours[0]) > 10000:
            x, y, cw, ch = cv2.boundingRect(contours[0])
        else:
            x, y, cw, ch = int(w * 0.15), int(h * 0.15), int(w * 0.7), int(h * 0.7)

        # 4. KESİN İÇ YÜZ OVALİ (Kulakları ve kazak/omuz yakasını dışarıda bırakır)
        inner_face_mask = np.zeros((h, w), dtype=np.uint8)
        center = (int(x + cw * 0.5), int(y + ch * 0.43))
        axes = (int(cw * 0.31), int(ch * 0.36))
        cv2.ellipse(inner_face_mask, center, axes, 0, 0, 360, 255, -1)

        # 5. Gözler, kaşlar, sakal, bıyık ve saç diplerini koyuluk eşiğiyle eler
        skin_pixels = gray[inner_face_mask > 0]
        mean_bright = np.median(skin_pixels) if len(skin_pixels) > 0 else 130
        dark_thresh = max(60, min(95, int(mean_bright * 0.65)))

        dark_features = ((gray < dark_thresh) & (inner_face_mask > 0)).astype(np.uint8) * 255
        dark_dilated = cv2.dilate(dark_features, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11)))

        # Saf iç yüz derisi = Ten rengi AND İç yüz ovali MINUS Göz/Sakal/Kaş
        pure_skin = cv2.bitwise_and(skin_closed, inner_face_mask)
        pure_skin = cv2.bitwise_and(pure_skin, cv2.bitwise_not(dark_dilated))

        return inner_face_mask, pure_skin

    def detect_blemishes(self, img_bgr: np.ndarray, pure_skin_mask: np.ndarray):
        """
        Dermatolojik Erythema (Kızarıklık/Egzema) ve Melanin (Koyu Leke)
        ayrımını yaparak gerçek lezyonları tespit eder.
        """
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        b, g, r = cv2.split(img_bgr.astype(np.float32))

        # 1. Egzema ve Enflamasyon (Kızarıklık / Erythema İndeksi: R - G)
        erythema = r - g
        erythema_smooth = cv2.GaussianBlur(erythema, (25, 25), 0)
        erythema_diff = erythema - erythema_smooth
        eczema_mask = (erythema_diff > 3.4) & (pure_skin_mask > 0)

        # 2. Koyu Leke ve Hiperpigmentasyon (Lokal Parlaklık Farkı)
        gray_smooth = cv2.GaussianBlur(gray.astype(np.float32), (25, 25), 0)
        gray_diff = gray_smooth - gray.astype(np.float32)
        pigment_mask = (gray_diff > 11.0) & (pure_skin_mask > 0)

        # İki tür lezyonu birleştir
        combined_mask = ((eczema_mask | pigment_mask).astype(np.uint8)) * 255
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))

        contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        valid_blemishes = []
        eczema_count = 0
        pigment_count = 0

        for cnt in contours:
            area = cv2.contourArea(cnt)
            # 8 ile 650 piksel arasındaki lezyon ve egzema plakları
            if 8 < area < 650:
                mask_cnt = np.zeros_like(gray, dtype=np.uint8)
                cv2.drawContours(mask_cnt, [cnt], -1, 255, -1)
                
                is_eczema = bool(np.mean(erythema_diff[mask_cnt > 0]) > 2.8)
                if is_eczema:
                    eczema_count += 1
                else:
                    pigment_count += 1

                valid_blemishes.append({
                    "contour": cnt,
                    "is_eczema": is_eczema,
                    "area": float(area)
                })

        pure_pixels = cv2.countNonZero(pure_skin_mask) + 1e-5
        total_area = sum([b["area"] for b in valid_blemishes])
        blemish_ratio = float((total_area / pure_pixels) * 100)

        total_count = len(valid_blemishes)
        blemish_score = min(100.0, round(total_count * 3.2, 1))

        return {
            "blemish_count": total_count,
            "eczema_count": eczema_count,
            "pigment_count": pigment_count,
            "blemish_area_ratio": round(blemish_ratio, 3),
            "blemish_score": blemish_score,
            "items": valid_blemishes
        }

    def analyze_wrinkles_glcm(self, img_bgr: np.ndarray, pure_skin_mask: np.ndarray):
        """
        GLCM ile yalnızca alın ve üst yanak bölgesinde doku ve kırışıklık analizi yapar.
        """
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        
        # Saf cilt bölgesinin sınırlarını bul
        skin_indices = np.where(pure_skin_mask > 0)
        if len(skin_indices[0]) < 200:
            roi_gray = cv2.resize(gray, (128, 128))
        else:
            ymin, ymax = np.min(skin_indices[0]), np.max(skin_indices[0])
            xmin, xmax = np.min(skin_indices[1]), np.max(skin_indices[1])
            # Alın ve üst yüz bölgesi (kırışıklıkların en net görüldüğü alan)
            forehead_h = int(ymin + (ymax - ymin) * 0.45)
            roi_crop = gray[ymin:forehead_h, xmin:xmax]
            if roi_crop.size > 100:
                roi_gray = cv2.resize(roi_crop, (128, 128))
            else:
                roi_gray = cv2.resize(gray, (128, 128))

        quantized = (roi_gray // 16).astype(np.uint8)

        if SKIMAGE_AVAILABLE:
            glcm = graycomatrix(
                quantized,
                distances=[1],
                angles=[0, np.pi/4, np.pi/2, 3*np.pi/4],
                levels=16,
                symmetric=True,
                normed=True
            )
            contrast = float(graycoprops(glcm, 'contrast').mean())
            homogeneity = float(graycoprops(glcm, 'homogeneity').mean())
            energy = float(graycoprops(glcm, 'energy').mean())
        else:
            glcm_res = _numpy_glcm_features(quantized, levels=16)
            contrast = glcm_res["contrast"]
            homogeneity = glcm_res["homogeneity"]
            energy = glcm_res["energy"]

        # Canny kenar yoğunluğu
        edges = cv2.Canny(roi_gray, 40, 110)
        edge_density = float(np.sum(edges > 0) / (edges.shape[0] * edges.shape[1]))

        # Normalleştirilmiş kırışıklık puanı
        wrinkle_score = min(100.0, max(0.0, round((contrast * 20.0) + (edge_density * 120.0), 1)))
        smoothness_score = min(100.0, max(0.0, round(homogeneity * 100.0, 1)))

        return {
            "glcm_contrast": round(contrast, 3),
            "glcm_homogeneity": round(homogeneity, 3),
            "glcm_energy": round(energy, 3),
            "wrinkle_score": wrinkle_score,
            "texture_smoothness_score": smoothness_score
        }

    def generate_annotated_image(self, img_bgr: np.ndarray, blemish_items, pure_skin_mask: np.ndarray) -> str:
        """
        Egzema/kızarıklık lezyonlarını parlak kırmızı halka ile,
        koyu pigment lekelerini amber/turuncu halka ile işaretler.
        """
        output = img_bgr.copy()

        # 1. Saf cilt bölgesini hafif yeşil katmanla göster
        overlay = output.copy()
        overlay[pure_skin_mask > 0] = cv2.addWeighted(
            output[pure_skin_mask > 0], 0.88, np.full_like(output[pure_skin_mask > 0], (0, 220, 100)), 0.12, 0
        )
        output = cv2.addWeighted(overlay, 0.5, output, 0.5, 0)

        # 2. Lezyonları renkle işaretle
        for item in blemish_items:
            cnt = item["contour"]
            (x, y), radius = cv2.minEnclosingCircle(cnt)
            center = (int(x), int(y))
            r = max(5, int(radius) + 3)

            if item["is_eczema"]:
                # Egzema / Kızarıklık: Parlak Kırmızı
                cv2.circle(output, center, r, (0, 0, 255), 2)
                cv2.circle(output, center, 2, (0, 0, 255), -1)
            else:
                # Koyu Leke / Melanin: Amber / Turuncu
                cv2.circle(output, center, r, (0, 140, 255), 2)
                cv2.circle(output, center, 2, (0, 140, 255), -1)

        # JPEG encode ve base64 dönüşümü
        _, buffer = cv2.imencode('.jpg', output, [cv2.IMWRITE_JPEG_QUALITY, 85])
        base64_str = base64.b64encode(buffer).decode('utf-8')
        return f"data:image/jpeg;base64,{base64_str}"

    def analyze(self, image_bytes: bytes) -> dict:
        """Tüm analiz adımlarını birleştiren ana metot."""
        img_bgr = self.decode_image(image_bytes)

        # 1. İç Yüz ve Saf Cilt Maskesini Çıkar (Kulak, kazak yakası, saç, sakal elenir)
        face_mask, pure_skin_mask = self.extract_face_and_skin_mask(img_bgr)

        # 2. Leke & Egzema Analizi (Erythema + Pigmentasyon)
        blemish_res = self.detect_blemishes(img_bgr, pure_skin_mask)

        # 3. Kırışıklık ve Doku Analizi (Alın/Yanak GLCM)
        texture_res = self.analyze_wrinkles_glcm(img_bgr, pure_skin_mask)

        # 4. Genel Cilt Sağlık Puanı (Dermatolojik ölçeklendirme)
        health_score = max(0.0, min(100.0, round(
            100.0 - (blemish_res["blemish_score"] * 0.45 + texture_res["wrinkle_score"] * 0.35), 1
        )))

        # 5. Cilt Durum Yorumu
        if health_score >= 82:
            status_text = "Çok İyi: Cilt dokusu pürüzsüz ve lezyon yoğunluğu düşük."
        elif health_score >= 65:
            status_text = "İyi / Normal: Cilt genel olarak sağlıklı; hafif bölgesel kızarıklık/egzema veya lekeler mevcut."
        else:
            status_text = "Bakım Önerilir: Bölgesel kızarıklık, egzema veya leke yoğunluğu belirgin seviyede."

        # 6. İşaretli Görsel Çıktısı
        annotated_b64 = self.generate_annotated_image(img_bgr, blemish_res["items"], pure_skin_mask)

        return {
            "status": "success",
            "overall_health_score": health_score,
            "status_evaluation": status_text,
            "blemish_analysis": {
                "detected_spots_count": blemish_res["blemish_count"],
                "eczema_redness_count": blemish_res["eczema_count"],
                "pigment_spots_count": blemish_res["pigment_count"],
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
