@echo off
title DermAI - Cilt ve Yuz Analizi Sunucusu
echo ========================================================
echo  DermAI Yapay Zeka ve Goruntu Isleme Sunucusu Baslatiliyor...
echo ========================================================
cd /d "C:\Users\namel\.gemini\antigravity\scratch\face-skin-analysis\ai_service"
"C:\Users\namel\.gemini\antigravity\scratch\py311\python.exe" -m uvicorn main:app --host 127.0.0.1 --port 8000 --reload
pause
