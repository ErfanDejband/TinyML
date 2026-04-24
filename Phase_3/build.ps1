# build.ps1 — Build script that works around CMake's UTF-8 BOM issue on Windows
# Usage: .\build.ps1        (build only)
#        .\build.ps1 -Run   (build + run)

param([switch]$Run)

$ErrorActionPreference = "Stop"
$buildDir = "$PSScriptRoot\build"
$rspFile  = "$buildDir\CMakeFiles\main.rsp"

# Step 1: Run cmake build (compiles all .cc -> .obj, linking will fail due to BOM)
Write-Host "=== Building... ===" -ForegroundColor Cyan
cmake --build $buildDir 2>&1 | ForEach-Object { $_ }

# Step 2: If the .rsp file has a UTF-8 BOM, strip it and re-link
if (Test-Path $rspFile) {
    $bytes = [System.IO.File]::ReadAllBytes($rspFile)
    if ($bytes.Length -ge 3 -and $bytes[0] -eq 0xEF -and $bytes[1] -eq 0xBB -and $bytes[2] -eq 0xBF) {
        Write-Host "=== Stripping BOM from response file and re-linking... ===" -ForegroundColor Yellow
        $content = [System.IO.File]::ReadAllText($rspFile, [System.Text.Encoding]::UTF8)
        [System.IO.File]::WriteAllText($rspFile, $content, (New-Object System.Text.UTF8Encoding $false))
        
        Push-Location $buildDir
        & C:\msys64\ucrt64\bin\c++.exe "@CMakeFiles\main.rsp" -o main.exe
        if ($LASTEXITCODE -ne 0) { Pop-Location; throw "Linking failed!" }
        Pop-Location
        Write-Host "=== Link successful! ===" -ForegroundColor Green
    }
}

# Step 3: Optionally run
if ($Run) {
    Write-Host "=== Running main.exe ===" -ForegroundColor Cyan
    & "$buildDir\main.exe"
}
