package com.dermai.skinanalysis.config;

import org.springframework.beans.factory.annotation.Value;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.web.reactive.function.client.ExchangeStrategies;
import org.springframework.web.reactive.function.client.WebClient;
import org.springframework.web.servlet.config.annotation.CorsRegistry;
import org.springframework.web.servlet.config.annotation.WebMvcConfigurer;

@Configuration
public class AppConfig implements WebMvcConfigurer {

    @Value("${ai.service.url:http://localhost:8000}")
    private String aiServiceUrl;

    // Python servisine yüksek boyutlu görsel gönderip alabilmek için bellek limitini genişletilmiş WebClient
    @Bean
    public WebClient webClient() {
        final int size = 16 * 1024 * 1024; // 16 MB bellek tamponu
        final ExchangeStrategies strategies = ExchangeStrategies.builder()
                .codecs(codecs -> codecs.defaultCodecs().maxInMemorySize(size))
                .build();

        return WebClient.builder()
                .baseUrl(aiServiceUrl)
                .exchangeStrategies(strategies)
                .build();
    }

    // Frontend için Cross-Origin (CORS) İzinleri
    @Override
    public void addCorsMappings(CorsRegistry registry) {
        registry.addMapping("/**")
                .allowedOrigins("*")
                .allowedMethods("GET", "POST", "PUT", "DELETE", "OPTIONS")
                .allowedHeaders("*");
    }
}
