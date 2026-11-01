"""
Hızlı Test Betiği:
Sentetik bir test görseli oluşturup analyzer.py motorunu doğrular.
Çalıştırmak için: python test_local.py
"""

import numpy as np
import cv2
from analyzer import SkinAnalyzer


def run_test():
    print("Test görseli oluşturuluyor...")
    # Sentetik ten rengi arka plan (180, 200, 240 BGR)
    img = np.full((400, 400, 3), (170, 195, 235), dtype=np.uint8)

    # Birkaç test lekesi ekle (koyu kahverengi noktalar)
    cv2.circle(img, (150, 150), 6, (80, 110, 140), -1)
    cv2.circle(img, (220, 180), 8, (70, 100, 130), -1)
    cv2.circle(img, (180, 260), 5, (90, 120, 150), -1)

    # Birkaç ince çizgi ekle (kırışıklık simülasyonu)
    cv2.line(img, (120, 100), (280, 105), (140, 170, 210), 1)
    cv2.line(img, (130, 120), (270, 122), (140, 170, 210), 1)

    # JPEG baytlarına kodla
    _, buffer = cv2.imencode('.jpg', img)
    image_bytes = buffer.tobytes()

    print("SkinAnalyzer motoru çalıştırılıyor...")
    analyzer = SkinAnalyzer()
    res = analyzer.analyze(image_bytes)

    print("\n--- ANALİZ BAŞARIYLA TAMAMLANDI ---")
    print(f"Genel Sağlık Puanı: {res['overall_health_score']} / 100")
    print(f"Durum Değerlendirmesi: {res['status_evaluation']}")
    print(f"Tespit Edilen Leke Sayısı: {res['blemish_analysis']['detected_spots_count']}")
    print(f"Leke Şiddet Puanı: {res['blemish_analysis']['blemish_severity_score']}")
    print(f"Kırışıklık Puanı: {res['texture_and_wrinkle_analysis']['wrinkle_score']}")
    print(f"Doku Pürüzsüzlük Skoru: {res['texture_and_wrinkle_analysis']['smoothness_score']}")
    print(f"GLCM Haralick Kontrast: {res['texture_and_wrinkle_analysis']['haralick_contrast']}")
    print(f"Görsel Çıktısı (Base64 uzunluğu): {len(res['annotated_image_base64'])} karakter.")
    print("\nMotor sorunsuz çalışıyor!")


if __name__ == "__main__":
    run_test()
