# Whole map again (shop names on every sign band, Spielhallen/betting shops, Kaufland pylon + logo, solid A4 rails):
# region build + all its tiles, all 4 corridor chunks, merge, minimap, every tile re-streamed, A4 drive check, EXE.
# The Unity steps wait until the editor is closed. Progress in a4\REDO.txt.
$w = "A:\blender_work"
$bl = "C:\Program Files (x86)\Steam\steamapps\common\Blender\blender.exe"
$un = "C:\Program Files\Unity\Hub\Editor\6000.2.10f1\Editor\Unity.exe"
$karta = "C:\Users\harha\Documents\" + (-join ([char[]](0x43A, 0x430, 0x440, 0x442, 0x430))) + ".blend"
$log = "$w\a4\REDO.txt"
function Say($m) { "$(Get-Date -Format 'dd.MM HH:mm') $m" | Tee-Object -FilePath $log -Append }
function Has($f, $m) { (Test-Path $f) -and [bool](Select-String -Path $f -Pattern $m -Quiet) }

Say "full redo start"
# Köln tiles on disk first
$kt = Get-Content "$w\a4\tiles_koeln.json" | ConvertFrom-Json
while ($true) {
    $miss = 0
    foreach ($pr in 'dgm1', 'dom1', 'dop10', 'lod2') {
        $have = @{}
        Get-ChildItem "$w\a4\tiles\$pr" -File -ErrorAction SilentlyContinue | Where-Object { $_.Extension -ne '.part' } |
            ForEach-Object { if ($_.Name -match '_32_(\d{3})_(\d{4})_') { $have["$($Matches[1])_$($Matches[2])"] = 1 } }
        foreach ($t in $kt) { if (-not $have.ContainsKey("$($t[0])_$($t[1])")) { $miss++ } }
    }
    if ($miss -eq 0) { break }
    Start-Sleep 60
}
Say "koeln tiles on disk"
# chunk 0 (grown westwards) and chunk 4 (the city) are prepared from scratch
foreach ($k in 0, 4) {
    foreach ($f in 'prep_log', 'veg_log', 'osm_log', 'textures_log') { Remove-Item "$w\a4\chunk_$k\$f.txt" -ErrorAction SilentlyContinue }
}
Copy-Item "$w\wiehl_territory.blend" "$w\wiehl_territory_prev.blend" -Force
& $bl -b $karta --python "$w\wiehl_data\build_blend.py" -- "$w\wiehl_territory.blend" "$w\region_data\prepared" *> "$w\build_log.txt"
if (-not (Has "$w\build_log.txt" '^BUILT')) { Say "REGION BUILD FAILED"; exit 1 }
Say ("region built; " + ((Select-String -Path "$w\build_log.txt" -Pattern '^SHOP_SIGNS').Line))
Copy-Item "$w\textures\nrw_*.png" "$w\unity_export\region\Textures\" -Force
Remove-Item Env:ONLY_TILES -ErrorAction SilentlyContinue
foreach ($p in 0, 1, 2) {
    & $bl -b "$w\wiehl_territory.blend" --python "$w\wiehl_data\export_region.py" -- "$w\unity_export\region" "$w\region_data\prepared" all "part=$p/3" *> "$w\export_log_$p.txt"
    if (-not (Has "$w\export_log_$p.txt" 'EXPORT_DONE')) { Say "REGION EXPORT $p FAILED"; exit 1 }
}
Say "region exported"
foreach ($k in 0, 1, 2, 3, 4) {
    Remove-Item "$w\a4\chunk_$k\build_log.txt", "$w\a4\chunk_$k\export_log_*.txt" -ErrorAction SilentlyContinue
    $r = & powershell -NoProfile -ExecutionPolicy Bypass -File "$w\a4\run_chunk.ps1" $k
    if (-not ($r -match "CHUNK $k DONE")) { $r | ForEach-Object { Say $_ }; Say "STOPPED at chunk $k"; exit 1 }
    Say ("chunk $k ok; " + ((Select-String -Path "$w\a4\chunk_$k\build_log.txt" -Pattern '^SHOP_SIGNS').Line))
}
while (Get-Process Unity -ErrorAction SilentlyContinue) { Start-Sleep 30 }
Say "Unity closed, merging"
python "$w\a4\merge_unity.py" *> "$w\a4\merge_log.txt"
if (-not (Has "$w\a4\merge_log.txt" 'MERGE_DONE')) { Say "MERGE FAILED"; exit 1 }
Copy-Item "$w\unity_export\region\Tile_*.fbx" "C:\dexstand\Assets\Maps\Import\" -Force
Get-ChildItem "$w\unity_export\region\Tile_*.fbx", "$w\unity_export\a4_*\Tile_*.fbx" | ForEach-Object { $_.BaseName } | Set-Content "C:\dexstand\restream.txt"
python "$w\a4\make_minimap_all.py" "C:\dexstand\Assets\Resources\Minimap" *> "$w\a4\minimap_log.txt"
if (-not (Has "$w\a4\minimap_log.txt" 'MINIMAP_DONE')) { Say "MINIMAP FAILED"; exit 1 }
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
