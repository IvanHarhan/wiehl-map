# Everything again with the Autobahn's solid outer guard rails and the region's A4 up to the Wiehl junction:
# region build + its 13 A4 tiles, all 4 corridor chunks rebuilt, merged, whole-map minimap, tiles re-streamed,
# the drive check (a4_check.txt), EXE. Progress in a4\REDO.txt. Unity must be closed.
$w = "A:\blender_work"
$un = "C:\Program Files\Unity\Hub\Editor\6000.2.10f1\Editor\Unity.exe"
$log = "$w\a4\REDO.txt"
function Say($m) { "$(Get-Date -Format 'dd.MM HH:mm') $m" | Tee-Object -FilePath $log -Append }
function Has($f, $m) { (Test-Path $f) -and [bool](Select-String -Path $f -Pattern $m -Quiet) }
$regionTiles = "393_5647,393_5648,394_5647,394_5648,395_5647,395_5648,396_5647,397_5647,398_5647,399_5647,400_5647,400_5648,401_5648".Split(",")

Say "start"
& powershell -NoProfile -ExecutionPolicy Bypass -File "$w\a4\rebuild_region_a4.ps1"
if (-not (Has "$w\a4\region_rebuild.txt" 'EXPORT_DONE')) { Say "REGION FAILED"; exit 1 }
Say "region ok"
foreach ($k in 0, 1, 2, 3) {
    Remove-Item "$w\a4\chunk_$k\build_log.txt", "$w\a4\chunk_$k\export_log_*.txt" -ErrorAction SilentlyContinue
    $r = & powershell -NoProfile -ExecutionPolicy Bypass -File "$w\a4\run_chunk.ps1" $k
    if (-not ($r -match "CHUNK $k DONE")) { $r | ForEach-Object { Say $_ }; Say "STOPPED at chunk $k"; exit 1 }
    Say "chunk $k ok"
}
if (Get-Process Unity -ErrorAction SilentlyContinue) { Say "Unity is open - close it and re-run"; exit 1 }
python "$w\a4\merge_unity.py" *> "$w\a4\merge_log.txt"
if (-not (Has "$w\a4\merge_log.txt" 'MERGE_DONE')) { Say "MERGE FAILED"; exit 1 }
foreach ($t in $regionTiles) { Copy-Item "$w\unity_export\region\Tile_$t.fbx" "C:\dexstand\Assets\Maps\Import\" -Force }
$names = $regionTiles + (Get-ChildItem "$w\unity_export\a4_*\Tile_*.fbx" | ForEach-Object { $_.BaseName.Substring(5) })
$names | ForEach-Object { "Tile_$_" } | Set-Content "C:\dexstand\restream.txt"
python "$w\a4\make_minimap_all.py" "C:\dexstand\Assets\Resources\Minimap" *> "$w\a4\minimap_log.txt"
if (-not (Has "$w\a4\minimap_log.txt" 'MINIMAP_DONE')) { Say "MINIMAP FAILED"; exit 1 }
# 11250 px wide: the importer must keep it (it capped the old one at 2048)
$meta = "C:\dexstand\Assets\Resources\Minimap\minimap.jpg.meta"
(Get-Content $meta -Raw) -replace 'maxTextureSize: \d+', 'maxTextureSize: 16384' | Set-Content $meta -Encoding ascii
Say "merged, minimap written"
& $un -batchmode -quit -projectPath C:\dexstand -executeMethod DexstandSceneBuilder.StreamNewTiles -logFile "$w\a4\unity_stream_log.txt" | Out-Null
$s = Select-String -Path "$w\a4\unity_stream_log.txt" -Pattern 'StreamNewTiles: \d+ new' | Select-Object -Last 1
if (-not $s) { Say "STREAMNEW FAILED"; exit 1 }
Say $s.Line.Trim()
& $un -batchmode -quit -projectPath C:\dexstand -executeMethod DexstandA4Check.Run -logFile "$w\a4\unity_check_log.txt" | Out-Null
Say ("check: " + (Get-Content "$w\a4\a4_check.txt" -TotalCount 1))
& $un -batchmode -quit -projectPath C:\dexstand -executeMethod DexstandBuild.BuildWindows -logFile "$w\exe_log.txt" | Out-Null
$b = Select-String -Path "$w\exe_log.txt" -Pattern '\[Build\] (EXE OK|FAILED).*' | Select-Object -Last 1
Say ("EXE: " + $(if ($b) { $b.Line.Trim() } else { "no result line" }))
Say "ALL DONE"
