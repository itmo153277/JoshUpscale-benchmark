@echo off

verify extentions >nul 2>&1
setlocal EnableExtensions
if errorlevel 1 goto Unsupported
wmic Alias /? >nul 2>&1 
if errorlevel 1 goto Unsupported

cd /d "%~dp0"
REM path %WINDIR%;%WINDIR%\system32;%WINDIR%\system32\wbem
path %CD%\bin;%PATH%

set ScriptName=%~nx0
set ScriptTitle=JoshUpscale Benchmark

if not "%~1" == "/logging" goto SetupLogging
set logTimestamp=%~2
shift
shift

title %ScriptTitle%

mkdir profiles\%logTimestamp%

:: Benchmark start

if not "%~1" == "" (
  call :Benchmark "%~f1"
  goto Success
)

for /F "eol=; delims=" %%a in (benchmark-list.conf) do (
  call :Benchmark %%a
)

:: Benchmark end

goto Success

:SetupLogging
call :GenLogTimestamp
call :Echo JoshUpscale model benchmark
Call :Tool 7z.exe "%ProgramFiles%\7-Zip"
call :Echo Logging to: %~dp0logs\%logTimestamp%.log
call "%~f0" /logging %logTimestamp% %* 2>&1 | tee "%~dp0logs\%logTimestamp%.log"
7z a -mx9 "%logTimestamp%.zip" "logs\%logTimestamp%.log" "profiles\%logTimestamp%" >nul
call :Echo Archive path: %logTimestamp%.zip
endlocal
pause >nul
goto :EOF

:Benchmark
title %ScriptTitle% - %*
set CUDA_ROOT=%CD%
set CUDA_HOME=%CD%
set CUDA_VISIBLE_DEVICES=0
set TF_CPP_MIN_LOG_LEVEL=0
set TF_CPP_VMODULE=
call :Execute benchmark --profile-path profiles\%logTimestamp% %*
title %ScriptTitle%
goto :EOF

:GenLogTimestamp
setlocal

for /f "skip=1 tokens=1-6" %%a in ('wmic Path Win32_LocalTime Get Day^,Month^,Year^,Hour^,Minute^,Second /Format:table') do (
  if not "%%~f" == "" (
    set currentYear=%%f
    set currentMonth=00%%d
    set currentDay=00%%a
    set currentHour=00%%b
    set currentMinute=00%%c
    set currentSecond=00%%e
  )
)
set currentMonth=%currentMonth:~-2%
set currentDay=%currentDay:~-2%
set currentHour=%currentHour:~-2%
set currentMinute=%currentMinute:~-2%
set currentSecond=%currentSecond:~-2%
(
  endlocal
  set logTimestamp=%currentYear%-%currentMonth%-%currentDay%_%currentHour%_%currentMinute%_%currentSecond%
)
goto :EOF

:Echo
echo %DATE% %TIME% [%ScriptName%] %*
goto :EOF

:Execute
call :Echo Execute: %*
copy nul nul 1>nul 2>&1
%*
if errorlevel 1 goto Error
goto :EOF

:Tool
setlocal
set oldPath=%PATH%
set newToolPath=%~2
:ToolCheck
if not exist "%newToolPath%\*" (
  for /F "delims=" %%a in ("%newToolPath%") do set newToolPath=%%~dpa
)
if not "%newToolPath%" == "" path %oldPath%;%newToolPath%
for /F "delims=" %%a in ("%~1") do set foundPath=%%~f$PATH:a
if "%foundPath%" == "" goto :ToolPrompt
call :Echo Found %~1: %foundPath%
(
  endlocal
  call :ToolSetPath "%PATH%"
)
goto :EOF
:ToolPrompt
set /p newToolPath="Enter path to %~1: "
set newToolPath=%newToolPath:"=%
goto ToolCheck
:ToolSetPath
set PATH=%~1
goto :EOF

:Success
call :Echo Benchmark finished successfully
endlocal
exit

:Error
call :Echo Error %ErrorLevel%
(
  title %ScriptTitle%
  endlocal
  exit %ErrorLevel%
)

:Unsupported
echo Unsupported COMMAND.COM
pause
