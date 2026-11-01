package com.dermai.skinanalysis.service;

import com.dermai.skinanalysis.dto.PythonAnalysisResponse;
import com.dermai.skinanalysis.model.AnalysisRecord;
import com.dermai.skinanalysis.repository.AnalysisRecordRepository;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.core.io.ByteArrayResource;
import org.springframework.http.HttpEntity;
import org.springframework.http.MediaType;
import org.springframework.http.client.MultipartBodyBuilder;
import org.springframework.stereotype.Service;
import org.springframework.util.MultiValueMap;
import org.springframework.web.multipart.MultipartFile;
import org.springframework.web.reactive.function.BodyInserters;
import org.springframework.web.reactive.function.client.WebClient;

import java.io.IOException;
import java.util.List;

@Service
@RequiredArgsConstructor
@Slf4j
public class SkinAnalysisService {

    private final WebClient webClient;
    private final AnalysisRecordRepository recordRepository;

    /**
     * Kullanıcıdan gelen fotoğrafı doğrular, Python mikroservisine iletir,
     * analiz sonuçlarını alır ve veritabanına kaydeder.
     */
    public PythonAnalysisResponse processAndSaveAnalysis(MultipartFile file) throws IOException {
        if (file.isEmpty()) {
            throw new IllegalArgumentException("Yüklenen dosya boş olamaz.");
        }

        String contentType = file.getContentType();
        if (contentType == null || !contentType.startsWith("image/")) {
            throw new IllegalArgumentException("Yalnızca görsel dosyaları (.jpg, .png) analiz edilebilir.");
        }

        log.info("Fotoğraf alındı: '{}', Boyut: {} bytes. Python servisine yönlendiriliyor...",
                file.getOriginalFilename(), file.getSize());

        // Multipart form-data gövdesi hazırla
        MultipartBodyBuilder bodyBuilder = new MultipartBodyBuilder();
        ByteArrayResource fileResource = new ByteArrayResource(file.getBytes()) {
            @Override
            public String getFilename() {
                return file.getOriginalFilename();
            }
        };
        bodyBuilder.part("file", fileResource).contentType(MediaType.parseMediaType(contentType));

        MultiValueMap<String, HttpEntity<?>> multipartBody = bodyBuilder.build();

        // Python FastAPI Mikroservisine İstek Gönder
        PythonAnalysisResponse aiResponse = webClient.post()
                .uri("/api/analyze")
                .contentType(MediaType.MULTIPART_FORM_DATA)
                .body(BodyInserters.fromMultipartData(multipartBody))
                .retrieve()
                .bodyToMono(PythonAnalysisResponse.class)
                .block(); // Asenkron cevabı bekle

        if (aiResponse == null) {
            throw new RuntimeException("Python yapay zeka servisinden yanıt alınamadı.");
        }

        // Analiz sonuçlarını H2 veritabanına kaydet
        AnalysisRecord record = AnalysisRecord.builder()
                .originalFileName(file.getOriginalFilename())
                .overallScore(aiResponse.getOverallHealthScore())
                .statusEvaluation(aiResponse.getStatusEvaluation())
                .detectedSpotsCount(aiResponse.getBlemishAnalysis() != null ? aiResponse.getBlemishAnalysis().getDetectedSpotsCount() : 0)
                .blemishSeverityScore(aiResponse.getBlemishAnalysis() != null ? aiResponse.getBlemishAnalysis().getBlemishSeverityScore() : 0.0)
                .wrinkleScore(aiResponse.getTextureAndWrinkleAnalysis() != null ? aiResponse.getTextureAndWrinkleAnalysis().getWrinkleScore() : 0.0)
                .smoothnessScore(aiResponse.getTextureAndWrinkleAnalysis() != null ? aiResponse.getTextureAndWrinkleAnalysis().getSmoothnessScore() : 0.0)
                .build();

        recordRepository.save(record);
        log.info("Analiz başarıyla tamamlandı ve veritabanına kaydedildi. Kayıt ID: {}", record.getId());

        return aiResponse;
    }

    /**
     * Geçmiş tüm analiz kayıtlarını veritabanından getirir.
     */
    public List<AnalysisRecord> getAllAnalysisHistory() {
        return recordRepository.findAllByOrderByCreatedAtDesc();
    }
}
