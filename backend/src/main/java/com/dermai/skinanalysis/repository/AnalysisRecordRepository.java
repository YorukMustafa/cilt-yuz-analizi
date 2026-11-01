package com.dermai.skinanalysis.repository;

import com.dermai.skinanalysis.model.AnalysisRecord;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.stereotype.Repository;

import java.util.List;

@Repository
public interface AnalysisRecordRepository extends JpaRepository<AnalysisRecord, Long> {

    // En son yapılan analizleri tarihe göre yeniden eskiye doğru sıralar
    List<AnalysisRecord> findAllByOrderByCreatedAtDesc();
}
