package com.dermai.skinanalysis.controller;

import com.dermai.skinanalysis.dto.PythonAnalysisResponse;
import com.dermai.skinanalysis.model.AnalysisRecord;
import com.dermai.skinanalysis.service.SkinAnalysisService;
import lombok.RequiredArgsConstructor;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;
import org.springframework.web.multipart.MultipartFile;

import java.io.IOException;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

@RestController
@RequestMapping("/api/v1")
@RequiredArgsConstructor
public class SkinAnalysisController {

    private final SkinAnalysisService analysisService;

    /**
     * Sağlık Kontrol Uç Noktası
     */
    @GetMapping("/health")
    public ResponseEntity<Map<String, Object>> healthCheck() {
        Map<String, Object> status = new HashMap<>();
        status.put("service", "Java Spring Boot Skin Analysis Backend");
        status.put("status", "UP");
        status.put("database", "H2 In-Memory DB Active");
        return ResponseEntity.ok(status);
    }

    /**
     * Fotoğraf Yükleme ve Analiz Uç Noktası
     */
    @PostMapping(value = "/analyze", consumes = "multipart/form-data")
    public ResponseEntity<?> analyzeFaceSkin(@RequestParam("file") MultipartFile file) {
        try {
            PythonAnalysisResponse response = analysisService.processAndSaveAnalysis(file);
            return ResponseEntity.ok(response);
        } catch (IllegalArgumentException e) {
            return ResponseEntity.badRequest().body(Map.of("error", e.getMessage()));
        } catch (IOException e) {
            return ResponseEntity.internalServerError().body(Map.of("error", "Dosya işleme hatası: " + e.getMessage()));
        } catch (Exception e) {
            return ResponseEntity.internalServerError().body(Map.of(
                    "error", "Yapay zeka analiz servisine bağlanılamadı. Lütfen Python FastAPI servisinin (Port 8000) çalıştığından emin olunuz.",
                    "details", e.getMessage()
            ));
        }
    }

    /**
     * Geçmiş Analiz Kayıtlarını Listeleme Uç Noktası
     */
    @GetMapping("/history")
    public ResponseEntity<List<AnalysisRecord>> getAnalysisHistory() {
        return ResponseEntity.ok(analysisService.getAllAnalysisHistory());
    }
}
