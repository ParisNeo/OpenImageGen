@ECHO OFF
REM Build OpenImageGen documentation on Windows.
pushd %~dp0

if "%SPHINXBUILD%" == "" set SPHINXBUILD=python -m sphinx
set SOURCEDIR=.
set BUILDDIR=_build
if "%SPHINXOPTS%" == "" set SPHINXOPTS=-W --keep-going

if "%1" == "" goto help

%SPHINXBUILD% -M %1 %SOURCEDIR% %BUILDDIR% %SPHINXOPTS%
if errorlevel 1 exit /b 1
goto end

:help
%SPHINXBUILD% -M help %SOURCEDIR% %BUILDDIR% %SPHINXOPTS%

:end
popd