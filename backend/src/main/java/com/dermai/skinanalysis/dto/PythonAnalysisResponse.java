package com.dermai.skinanalysis.dto;

import com.fasterxml.jackson.annotation.JsonProperty;
import lombok.Data;

@Data
public class PythonAnalysisResponse {

    private String status;

    @JsonProperty("overall_health_score")
    private Double overallHealthScore;

    @JsonProperty("status_evaluation")
    private String statusEvaluation;

    @JsonProperty("blemish_analysis")
    private BlemishDetails blemishAnalysis;

    @JsonProperty("texture_and_wrinkle_analysis")
    private TextureDetails textureAndWrinkleAnalysis;

    @JsonProperty("annotated_image_base64")
    private String annotatedImageBase64;

    @Data
    public static class BlemishDetails {
        @JsonProperty("detected_spots_count")
        private Integer detectedSpotsCount;

        @JsonProperty("blemish_density_percentage")
        private Double blemishDensityPercentage;

        @JsonProperty("blemish_severity_score")
        private Double blemishSeverityScore;
    }

    @Data
    public static class TextureDetails {
        @JsonProperty("wrinkle_score")
        private Double wrinkleScore;

        @JsonProperty("smoothness_score")
        private Double smoothnessScore;

        @JsonProperty("haralick_contrast")
        private Double haralickContrast;

        @JsonProperty("haralick_homogeneity")
        private Double haralickHomogeneity;

        @JsonProperty("haralick_energy")
        private Double haralickEnergy;
    }
}
