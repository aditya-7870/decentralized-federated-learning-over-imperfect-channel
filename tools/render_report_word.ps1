param([Parameter(Mandatory=$true)][string]$InputDocx,[Parameter(Mandatory=$true)][string]$OutputPdf)
$ErrorActionPreference = 'Stop'
$resolvedDocx = (Resolve-Path -LiteralPath $InputDocx).Path
$resolvedPdf = [System.IO.Path]::GetFullPath($OutputPdf)
$wordApp = New-Object -ComObject Word.Application
$wordApp.Visible = $false
$wordApp.DisplayAlerts = 0
$wordApp.AutomationSecurity = 3
$document = $null
try {
    $document = $wordApp.Documents.Open($resolvedDocx, $false, $true)
    $document.ExportAsFixedFormat($resolvedPdf, 17)
    Write-Output "Rendered $resolvedPdf"
} finally {
    if ($null -ne $document) { $document.Close(0) }
    $wordApp.Quit()
}
