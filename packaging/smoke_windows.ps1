# End-to-end check of the unzipped Windows beta on a real Windows machine (CI or a PC).
#   powershell -ExecutionPolicy Bypass -File smoke_windows.ps1 -App <unzipped folder>
# Speaks a short lecture with Windows' own voice, imports it, and requires a
# transcript with the right words, notes, a Markdown export and an archive.
param([Parameter(Mandatory)] [string] $App, [int] $Port = 8477)
$ErrorActionPreference = "Stop"
$App = (Resolve-Path $App).Path
$work = Join-Path ([IO.Path]::GetTempPath()) ("lectern-smoke-" + [guid]::NewGuid().ToString("N").Substring(0, 8))
New-Item -ItemType Directory -Force $work | Out-Null
$py = Join-Path $App "python\python.exe"
$bin = Join-Path $App "bin"

function Step($m) { Write-Host "==> $m" }
function Fail($m) { Write-Host "FAIL: $m"; if (Test-Path "$work\server.err") { Get-Content "$work\server.err" -Tail 60 }; exit 1 }

Step "bundled tools start"
& "$bin\ffmpeg.exe" -hide_banner -version | Select-Object -First 1
if ($LASTEXITCODE -ne 0) { Fail "ffmpeg.exe did not run" }
& "$bin\whisper-cli.exe" --help *> "$work\whisper-help.txt"
if ($LASTEXITCODE -ne 0) { Get-Content "$work\whisper-help.txt" -Tail 20; Fail "whisper-cli.exe did not run (missing DLL?)" }

Step "bundled Python imports every dependency"
& $py -c "import lectern, flask, waitress, numpy, psutil, sounddevice; print('python ok', lectern.__file__); print(len(sounddevice.query_devices()), 'audio devices')"
if ($LASTEXITCODE -ne 0) { Fail "bundled Python could not import the app" }

Step "speech model (tiny.en)"
$model = Join-Path $work "ggml-tiny.en.bin"
Invoke-WebRequest "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-tiny.en.bin" -OutFile $model

Step "a spoken test lecture"
Add-Type -AssemblyName System.Speech
$tts = New-Object System.Speech.Synthesis.SpeechSynthesizer
$tts.SetOutputToWaveFile("$work\lecture.wav")
$tts.Speak("Today we are talking about mitochondria. Mitochondria convert energy from glucose into a molecule called A T P. " +
           "The inner membrane folds into ridges. This process is called cellular respiration, and it happens in three stages.")
$tts.Dispose()

Step "start Lectern"
$env:LECTERN_ROOT = "$work\Lectures"; $env:LECTERN_SUPPORT = "$work\support"
$env:LECTERN_BIN_DIR = $bin; $env:LECTERN_MODEL = $model; $env:LECTERN_NOTES = "extractive"
$server = Start-Process $py -ArgumentList "-m", "lectern" -WorkingDirectory $App -PassThru -NoNewWindow `
  -RedirectStandardOutput "$work\server.out" -RedirectStandardError "$work\server.err"
$base = "http://127.0.0.1:$Port"
$up = $false
foreach ($i in 1..60) { try { Invoke-RestMethod "$base/api/status" | Out-Null; $up = $true; break } catch { Start-Sleep 1 } }
if (-not $up) { Fail "server did not answer on $base" }
$status = Invoke-RestMethod "$base/api/status"
if ($status.needs_model) { Fail "server does not see the speech model" }

try {
  Step "import and process"
  Invoke-RestMethod -Method Post "$base/api/notice" | Out-Null
  $r = Invoke-RestMethod -Method Post "$base/api/import?title=Smoke%20test&filename=lecture.wav" `
         -InFile "$work\lecture.wav" -ContentType "application/octet-stream"
  $id = $r.lecture_id
  $state = ""
  foreach ($i in 1..300) {
    $state = (Invoke-RestMethod "$base/api/lecture/$id").state
    if ($state -in "complete", "failed") { break }
    Start-Sleep 2
  }
  $lec = Invoke-RestMethod "$base/api/lecture/$id"
  if ($state -ne "complete") { Fail "lecture ended as '$state': $($lec.failure)" }

  Step "transcript, notes, export, archive"
  $t = Invoke-RestMethod "$base/api/lecture/$id/transcript"
  $text = ($t.words | ForEach-Object { $_.w }) -join " "
  Write-Host "transcript: $text"
  foreach ($w in "mitochondria", "glucose", "respiration") {
    if ($text -notmatch $w) { Fail "transcript is missing '$w'" }
  }
  $md = Invoke-WebRequest "$base/api/lecture/$id/download/markdown" -UseBasicParsing
  if ($md.Content -notmatch "Smoke test") { Fail "Markdown export looks wrong" }
  $archives = Get-ChildItem -Recurse "$work\Lectures" -Include *.m4a
  if (-not $archives) { Fail "no verified audio archive was written" }
  Write-Host "PASS: Windows beta records the full pipeline (state=$state, $($t.words.Count) words, archive $($archives[0].Name))"
} finally {
  Stop-Process -Id $server.Id -Force -ErrorAction SilentlyContinue
}
