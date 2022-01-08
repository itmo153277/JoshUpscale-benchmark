@echo off
title NVIDIA-SMI

path %~dp0\bin;%PATH%

:PrintStats
cls
nvidia-smi | tee -a gpu-stats.log
timeout 1 /nobreak > nul
goto PrintStats
