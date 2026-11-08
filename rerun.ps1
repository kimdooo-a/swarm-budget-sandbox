# Rerun one episode with a fixed seed: ./rerun.ps1 --seed 3 --condition pressure --n 8
Set-Location $PSScriptRoot
python rerun.py @args
