# update_git.ps1 — commit and push all changes to GitHub
# Usage:  powershell -ExecutionPolicy Bypass -File update_git.ps1 "commit message"
param([string]$Message = "update")

Set-Location $PSScriptRoot

Write-Host "== git status =="
git status --short

if (-not $Message) { $Message = "update" }

git add -A
git commit -m $Message
git push

Write-Host ""
Write-Host "== done =="
git log --oneline -1
