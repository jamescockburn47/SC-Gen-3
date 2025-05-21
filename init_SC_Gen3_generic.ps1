# PowerShell script to create Strategic Counsel Gen 3 folder structure

$root = "SC_Gen3"

$folders = @(
    "$root",
    "$root\tests",
    "$root\memory",
    "$root\memory\digests",
    "$root\summaries",
    "$root\exports",
    "$root\logs",
    "$root\static"
)

foreach ($folder in $folders) {
    if (-not (Test-Path $folder)) {
        New-Item -ItemType Directory -Path $folder -Force | Out-Null
        Write-Host "Created: $folder"
    } else {
        Write-Host "Already exists: $folder"
    }
}

Write-Host "`n✅ SC_Gen3 folder structure is now ready in: $((Get-Location).Path)\SC_Gen3"
