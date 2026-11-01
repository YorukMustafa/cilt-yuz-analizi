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

try:
    from skimage.feature import graycomatrix, graycoprops
    SKIMAGE_AVAILABLE = True
except ImportError:
    SKIMAGE_AVAILABLE = False


def _numpy_glcm_features(quantized, levels=16):
    """NumPy tabanlı hızlı Haralick Doku (GLCM) hesaplayıcı (skimage yedeği)."""
    h, w = quantized.shape
    # Basit ve hızlı 4 yönlü komşuluk (sağ, alt-sağ, alt, alt-sol)
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
        # Simetrik yap
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
        Arka planı (çarşaf, duvar, perde), saçları, sakalı ve gözleri eleyerek
        YALNIZCA gerçek yüz cilt bölgesini (alın, yanaklar, burun çevresi) izole eder.
        """
        h, w = img_bgr.shape[:2]
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

        # 1. YCrCb ve HSV renk uzaylarında ten rengi segmentasyonu
        ycrcb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2YCrCb)
        skin_ycrcb = cv2.inRange(ycrcb, np.array([55, 133, 80]), np.array([245, 175, 128]))

        hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
        skin_hsv = cv2.inRange(hsv, np.array([0, 25, 45]), np.array([30, 200, 255]))

        raw_skin = cv2.bitwise_and(skin_ycrcb, skin_hsv)

        # 2. Morfolojik temizleme
        k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        skin_closed = cv2.morphologyEx(raw_skin, cv2.MORPH_CLOSE, k_close)

        # 3. Yalnızca merkezi ana yüz konturunu seç (Arka plan nesnelerini ve çarşafları yok et)
        contours, _ = cv2.findContours(skin_closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        face_mask = np.zeros((h, w), dtype=np.uint8)

        if contours:
            # Konturları alana göre sırala
            contours = sorted(contours, key=cv2.contourArea, reverse=True)
            # En büyük bileşeni ana yüz adayı olarak al
            for cnt in contours:
                area = cv2.contourArea(cnt)
                if area > 10000:
                    x, y, cw, ch = cv2.boundingRect(cnt)
                    # Yüz genellikle görüntünün orta dikey ekseninde yer alır
                    cx = x + cw / 2
                    if 0.15 * w < cx < 0.85 * w:
                        cv2.drawContours(face_mask, [cnt], -1, 255, -1)
                        break

        # Eğer kontur bulunamadıysa (örneğin çok yakın çekim), tüm ten maskesini kullan
        if cv2.countNonZero(face_mask) < 5000:
            face_mask = skin_closed

        # 4. Yüz içindeki koyu bölgeleri (Gözler, kaşlar, saç, sakal, bıyık, burun delikleri) TESPİT ET VE ÇIKAR
        # Cilt ortalama parlaklığına göre dinamik eşikleme
        skin_pixels = gray[face_mask > 0]
        mean_skin_brightness = np.median(skin_pixels) if len(skin_pixels) > 0 else 140
        dark_threshold = max(60, min(105, int(mean_skin_brightness * 0.65)))

        dark_features = ((gray < dark_threshold) & (face_mask > 0)).astype(np.uint8) * 255
        # Sakal ve saç diplerindeki kıl köklerini de kapsamak için genişlet (dilation)
        k_features = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        dark_features_dilated = cv2.dilate(dark_features, k_features)

        # Saf cilt alanı = Ana yüz maskesi EKSİ koyu bölgeler (sakal/saç/gözler)
        pure_skin_mask = cv2.bitwise_and(face_mask, cv2.bitwise_not(dark_features_dilated))

        # Kenarlardaki gürültüleri gidermek için yumuşat
        pure_skin_mask = cv2.morphologyEx(pure_skin_mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))

        return face_mask, pure_skin_mask

    def detect_blemishes(self, img_bgr: np.ndarray, pure_skin_mask: np.ndarray):
        """
        Adaptif ve yerel kontrast farkı (Difference from Local Skin) kullanarak
        sadece gerçek leke, sivilce ve hiperpigmentasyon noktalarını tespit eder.
        """
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        
        # 1. Yerel pürüzsüz cilt arka planı çıkar (Geniş Gauss filtresi)
        local_smooth = cv2.GaussianBlur(gray, (31, 31), 0)
        
        # Piksellerin çevrelerine göre ne kadar koyu olduğunu hesapla
        diff = local_smooth.astype(np.int16) - gray.astype(np.int16)

        # 2. Kırmızı sivilce/enflamasyon tespiti (LAB uzayında A kanalı sapması)
        lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
        a_channel = lab[:, :, 1]
        a_smooth = cv2.GaussianBlur(a_channel, (31, 31), 0)
        red_diff = a_channel.astype(np.int16) - a_smooth.astype(np.int16)

        # YALNIZCA saf cilt maskesi içinde ve belirgin kontrast sapması olan yerleri al
        # diff > 20 (koyu lekeler) veya red_diff > 12 (kızarık/sivilce lekeleri)
        blemish_candidates = ((diff > 20) | (red_diff > 14)) & (pure_skin_mask > 0)
        blemish_mask = blemish_candidates.astype(np.uint8) * 255

        # Küçük tek piksellik gürültüleri filtrele
        k_clean = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        blemish_mask = cv2.morphologyEx(blemish_mask, cv2.MORPH_OPEN, k_clean)

        # 3. Kontur geometrisi ve alan analizi
        contours, _ = cv2.findContours(blemish_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        valid_blemishes = []

        for cnt in contours:
            area = cv2.contourArea(cnt)
            # Bir leke en az 12 piksel, en fazla 400 piksel olmalı (daha büyükler gölge veya saç kalıntısıdır)
            if 12 < area < 450:
                # Dairesellik kontrolü (uzun çizgiler ve gölgeleri ele)
                perimeter = cv2.arcLength(cnt, True)
                if perimeter > 0:
                    circularity = 4 * np.pi * (area / (perimeter * perimeter))
                    if circularity > 0.2:  # Lekeler yuvarlaktır
                        valid_blemishes.append(cnt)

        pure_skin_pixel_count = cv2.countNonZero(pure_skin_mask) + 1e-5
        total_blemish_area = sum([cv2.contourArea(c) for c in valid_blemishes])
        blemish_ratio = float((total_blemish_area / pure_skin_pixel_count) * 100)

        # Gerçekçi leke şiddet puanı (0-100)
        # Normal genç bir ciltte 5-20 leke normaldir
        blemish_count = len(valid_blemishes)
        blemish_score = min(100.0, round(blemish_count * 2.2, 1))

        return {
            "blemish_count": blemish_count,
            "blemish_area_ratio": round(blemish_ratio, 3),
            "blemish_score": blemish_score,
            "contours": valid_blemishes
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
        wrinkle_score = min(100.0, max(0.0, round((contrast * 22.0) + (edge_density * 150.0), 1)))
        smoothness_score = min(100.0, max(0.0, round(homogeneity * 100.0, 1)))

        return {
            "glcm_contrast": round(contrast, 3),
            "glcm_homogeneity": round(homogeneity, 3),
            "glcm_energy": round(energy, 3),
            "wrinkle_score": wrinkle_score,
            "texture_smoothness_score": smoothness_score
        }

    def generate_annotated_image(self, img_bgr: np.ndarray, blemish_contours, pure_skin_mask: np.ndarray) -> str:
        """
        Yalnızca gerçek lekelere kırmızı çember çizer ve analiz edilen saf cilt
        alanını hafif saydam yeşil renkle vurgular.
        """
        output = img_bgr.copy()

        # 1. Saf cilt bölgesini hafif yeşil katmanla göster
        overlay = output.copy()
        overlay[pure_skin_mask > 0] = cv2.addWeighted(
            output[pure_skin_mask > 0], 0.88, np.full_like(output[pure_skin_mask > 0], (0, 220, 100)), 0.12, 0
        )
        output = cv2.addWeighted(overlay, 0.5, output, 0.5, 0)

        # 2. Tespit edilen gerçek lekelere ince kırmızı halkalar çiz
        for cnt in blemish_contours:
            (x, y), radius = cv2.minEnclosingCircle(cnt)
            center = (int(x), int(y))
            r = max(4, int(radius) + 2)
            cv2.circle(output, center, r, (0, 0, 240), 2)
            cv2.circle(output, center, 1, (0, 0, 255), -1)

        # JPEG encode ve base64 dönüşümü
        _, buffer = cv2.imencode('.jpg', output, [cv2.IMWRITE_JPEG_QUALITY, 85])
        base64_str = base64.b64encode(buffer).decode('utf-8')
        return f"data:image/jpeg;base64,{base64_str}"

    def analyze(self, image_bytes: bytes) -> dict:
        """Tüm analiz adımlarını birleştiren ana metot."""
        img_bgr = self.decode_image(image_bytes)

        # 1. Yüz ve Saf Cilt Maskesini Çıkar (Arka plan ve sakal/saç/göz elenir)
        face_mask, pure_skin_mask = self.extract_face_and_skin_mask(img_bgr)

        # 2. Leke Analizi (Hassas yerel kontrast)
        blemish_res = self.detect_blemishes(img_bgr, pure_skin_mask)

        # 3. Kırışıklık ve Doku Analizi (Alın/Yanak GLCM)
        texture_res = self.analyze_wrinkles_glcm(img_bgr, pure_skin_mask)

        # 4. Genel Cilt Sağlık Puanı (Dermatolojik ölçeklendirme)
        health_score = max(0.0, min(100.0, round(
            100.0 - (blemish_res["blemish_score"] * 0.45 + texture_res["wrinkle_score"] * 0.35), 1
        )))

        # 5. Cilt Durum Yorumu
        if health_score >= 82:
            status_text = "Çok İyi: Cilt dokusu pürüzsüz ve leke yoğunluğu düşük seviyede."
        elif health_score >= 65:
            status_text = "İyi / Normal: Cilt dokusu sağlıklı, hafif bölgesel lekeler mevcut."
        else:
            status_text = "Bakım Önerilir: Leke yoğunluğu veya doku pürüzlülüğü belirgin seviyede."

        # 6. İşaretli Görsel Çıktısı
        annotated_b64 = self.generate_annotated_image(img_bgr, blemish_res["contours"], pure_skin_mask)

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

