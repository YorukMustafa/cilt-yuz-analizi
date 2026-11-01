package com.dermai.skinanalysis;

import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;

@SpringBootApplication
public class SkinAnalysisApplication {

    public static void main(String[] args) {
        SpringApplication.run(SkinAnalysisApplication.class, args);
        System.out.println("==========================================================");
        System.out.println(" DermAI Spring Boot Sunucusu Basariyla Calisiyor!         ");
        System.out.println(" Port: http://localhost:8080                              ");
        System.out.println(" H2 Veritabani Konsolu: http://localhost:8080/h2-console  ");
        System.out.println("==========================================================");
    }
}
