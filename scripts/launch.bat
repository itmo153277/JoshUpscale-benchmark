@echo off

verify extentions >nul 2>&1
setlocal EnableExtensions
if errorlevel 1 goto Unsupported
wmic Alias /? >nul 2>&1 
if errorlevel 1 goto Unsupported

path %WINDIR%;%WINDIR%\system32;%WINDIR%\system32\wbem
path %~dp0\bin;%PATH%

set ScriptName=%~nx0
set ScriptTitle=JoshUpscale Benchmark

if not "%~1" == "/logging" goto SetupLogging
set logTimestamp=%~2
shift /1
shift /1

title %ScriptTitle%

:Main

set globalProfileDir=profiles\%logTimestamp%
set globalCacheDir=%TEMP%\JoshUpscale\%logTimestamp%
if not exist "%globalCacheDir%" mkdir "%globalCacheDir%"
if not exist "%~dp0%globalProfileDir%" mkdir "%~dp0%globalProfileDir%"

call :Echo Cache dir: %globalCacheDir%
call :Echo Profile dir: %globalProfileDir%

if not "%~1" == "" goto FromCmdline

for /F "eol=; delims=" %%a in (benchmark-list.conf) do (
  call :Benchmark %%a
)

goto Success

:FromCmdline

if "%~1" == "" goto Success

call :Benchmark %1

shift
goto FromCmdline

:SetupLogging
call :GenLogTimestamp
call :Echo JoshUpscale model benchmark
Call :Tool 7z.exe "%ProgramFiles%\7-Zip"
call :Echo Logging to: %~dp0logs\%logTimestamp%.log
call "%~f0" /logging %logTimestamp% %* 2>&1 | tee "%~dp0logs\%logTimestamp%.log"
cd /d "%~dp0"
7z a -mx9 "%logTimestamp%.zip" "logs\%logTimestamp%.log" "profiles\%logTimestamp%" >nul
call :Echo Archive path: %logTimestamp%.zip
endlocal
echo Press any key to exit
pause >nul
goto :EOF

:Benchmark
title %ScriptTitle% - %*
set CUDA_ROOT=%~dp0
set CUDA_HOME=%~dp0
set CUDA_VISIBLE_DEVICES=0
set TF_CPP_MIN_LOG_LEVEL=0
set TF_CPP_VMODULE=
set OMP_NUM_THREADS=%NUMBER_OF_PROCESSORS%
set configFile=%~f1
set configName=%~n1
set profileDir=%globalProfileDir%\%configName%
set cacheDir=%globalCacheDir%\%configName%
pushd %~dp0
if not exist "%profileDir%" mkdir "%profileDir%"
if not exist "%cacheDir%" mkdir "%cacheDir%"
call :Execute benchmark --profile-path "%profileDir%" --cache-path "%cacheDir%" "%configFile%"
popd
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
if not %ERRORLEVEL% == 0 goto Error
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
set ExitCode=0
goto CleanUp

:Error
call :Echo Error %ErrorLevel%
set ExitCode=%ErrorLevel%
goto CleanUp

:CleanUp
if not "%globalCacheDir%" == "" rmdir /s /q "%globalCacheDir%"
(
  title %ScriptTitle%
  endlocal
  exit %ExitCode%
)

:Unsupported
echo Unsupported COMMAND.COM
pause
