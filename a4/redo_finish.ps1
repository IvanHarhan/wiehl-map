# The Unity half of redo_full.ps1 alone: stream the tiles (again, from where a crashed run stopped: it saves its
# progress), the A4 check, the EXE. Progress in a4\REDO.txt.
$w = "A:\blender_work"
$un = "C:\Program Files\Unity\Hub\Editor\6000.2.10f1\Editor\Unity.exe"
$log = "$w\a4\REDO.txt"
function Say($m) { "$(Get-Date -Format 'dd.MM HH:mm') $m" | Tee-Object -FilePath $log -Append }
while (Get-Process Unity -ErrorAction SilentlyContinue) { Start-Sleep 30 }
$ok = $false
for ($try = 1; $try -le 4 -and -not $ok; $try++) {
    & $un -batchmode -quit -projectPath C:\dexstand -executeMethod DexstandSceneBuilder.StreamNewTiles -logFile "$w\a4\unity_stream_log.txt" | Out-Null
    $s = Select-String -Path "$w\a4\unity_stream_log.txt" -Pattern 'StreamNewTiles: \d+ new' | Select-Object -Last 1
    if ($s) { Say $s.Line.Trim(); $ok = $true } else { Say "stream run $try stopped early, going on from its saved progress" }
}
if (-not $ok) { Say "STREAMNEW FAILED"; exit 1 }
& $un -batchmode -quit -projectPath C:\dexstand -executeMethod DexstandA4Check.Run -logFile "$w\a4\unity_check_log.txt" | Out-Null
Say ("check: " + (Get-Content "$w\a4\a4_check.txt" -TotalCount 1))
& $un -batchmode -quit -projectPath C:\dexstand -executeMethod DexstandBuild.BuildWindows -logFile "$w\exe_log.txt" | Out-Null
$b = Select-String -Path "$w\exe_log.txt" -Pattern '\[Build\] (EXE OK|FAILED).*' | Select-Object -Last 1
Say ("EXE: " + $(if ($b) { $b.Line.Trim() } else { "no result line" }))
Say "ALL DONE"
