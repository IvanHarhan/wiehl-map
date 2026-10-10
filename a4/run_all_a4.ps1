# The whole A4 corridor, unattended: wait for the tile downloads, build + export every chunk, merge into dexstand,
# stream the new tiles in Unity (batch mode, the editor must be closed) and build the PC EXE.
# Progress in a4\STATUS.txt. Safe to re-run: finished steps are skipped.
$chunks = @(0, 1, 2, 3)  # chunk 3 (the smallest, next to Wiehl) is checked by hand first
$w = "A:\blender_work"
$un = "C:\Program Files\Unity\Hub\Editor\6000.2.10f1\Editor\Unity.exe"
$status = "$w\a4\STATUS.txt"
function Say($m) { "$(Get-Date -Format 'dd.MM HH:mm') $m" | Tee-Object -FilePath $status -Append }

Say "start, chunks $($chunks -join ',')"
# 1. downloads: every chunk tile of every product on disk (fetchers leave .part files while busy)
$all = Get-Content "$w\a4\tiles_all.json" | ConvertFrom-Json
for ($i = 0; $i -lt 720; $i++) {
    $missing = 0
    foreach ($p in 'dgm1', 'dom1', 'dop10', 'lod2') {
        $have = @{}
        Get-ChildItem "$w\a4\tiles\$p" -File -ErrorAction SilentlyContinue | Where-Object { $_.Extension -ne '.part' } |
            ForEach-Object { if ($_.Name -match '_32_(\d{3})_(\d{4})_') { $have["$($Matches[1])_$($Matches[2])"] = 1 } }
        foreach ($t in $all) { if (-not $have.ContainsKey("$($t[0])_$($t[1])")) { $missing++ } }
    }
    if ($missing -eq 0) { break }
    if ($i % 10 -eq 0) { Say "waiting for downloads: $missing files missing" }
    Start-Sleep 60
}
if ($missing -ne 0) { Say "DOWNLOADS INCOMPLETE ($missing missing) - restart a4\fetch_tiles.py"; exit 1 }
Say "downloads complete"

# 2. chunks
foreach ($k in $chunks) {
    $r = & powershell -NoProfile -ExecutionPolicy Bypass -File "$w\a4\run_chunk.ps1" $k
    $r | ForEach-Object { Say $_ }
    if (-not ($r -match "CHUNK $k DONE")) { Say "STOPPED at chunk $k"; exit 1 }
}

# 3. into dexstand
if (Get-Process Unity -ErrorAction SilentlyContinue) { Say "Unity is open - close it, then re-run this script"; exit 1 }
python "$w\a4\merge_unity.py" *> "$w\a4\merge_log.txt"
if (-not (Select-String -Path "$w\a4\merge_log.txt" -Pattern 'MERGE_DONE' -Quiet)) { Say "MERGE FAILED"; exit 1 }
Say (Select-String -Path "$w\a4\merge_log.txt" -Pattern 'MERGE_DONE').Line
& $un -batchmode -quit -projectPath C:\dexstand -executeMethod DexstandSceneBuilder.StreamNewTiles -logFile "$w\a4\unity_stream_log.txt" | Out-Null
$s = Select-String -Path "$w\a4\unity_stream_log.txt" -Pattern 'StreamNewTiles: \d+' | Select-Object -Last 1
if (-not $s) { Say "STREAMNEW FAILED (a4\unity_stream_log.txt)"; exit 1 }
Say $s.Line.Trim()
& $un -batchmode -quit -projectPath C:\dexstand -executeMethod DexstandBuild.BuildWindows -logFile "$w\exe_log.txt" | Out-Null
$b = Select-String -Path "$w\exe_log.txt" -Pattern '\[Build\] (EXE OK|FAILED).*' | Select-Object -Last 1
Say ("EXE: " + $(if ($b) { $b.Line.Trim() } else { "no result line, see exe_log.txt" }))
Say "ALL DONE"
