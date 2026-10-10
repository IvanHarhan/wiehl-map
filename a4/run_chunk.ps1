# One A4 corridor chunk from downloaded tiles to Unity FBX tiles: prep, vegetation, OSM, Blender build, textures,
# 3 export parts. Usage: run_chunk.ps1 K   (logs in a4\chunk_K\*.txt; stops at the first failure; done steps skipped)
param([int]$k)
$bl = "C:\Program Files (x86)\Steam\steamapps\common\Blender\blender.exe"
$w = "A:\blender_work"
$cd = "$w\a4\chunk_$k"
$out = "$w\unity_export\a4_$k"
# the user's Documents map file, spelled in char codes: Windows PowerShell reads this BOM-less file as ANSI
$karta = "C:\Users\harha\Documents\" + (-join ([char[]](0x43A, 0x430, 0x440, 0x442, 0x430))) + ".blend"

# a missing log must read as "not done": Select-String on a missing path throws here and skipped whole if-statements
function Has($log, $mark) {
    if (-not (Test-Path $log)) { return $false }
    return [bool](Select-String -Path $log -Pattern $mark -Quiet)
}
function Step($name, $log, $mark) {
    if (-not (Has $log $mark)) { "CHUNK $k $name FAILED (see $log)"; exit 1 }
    "CHUNK $k $name ok $(Get-Date -Format HH:mm)"
}

if (-not (Has "$cd\prep_log.txt" 'PREP_DONE')) { python "$w\a4\prep_chunk.py" $cd *> "$cd\prep_log.txt" }
Step "prep" "$cd\prep_log.txt" 'PREP_DONE'
if (-not (Has "$cd\veg_log.txt" 'VEG_DONE')) { python "$w\a4\prep_veg_chunk.py" $cd *> "$cd\veg_log.txt" }
Step "veg" "$cd\veg_log.txt" 'VEG_DONE'
if (-not (Has "$cd\osm_log.txt" 'OSM_DONE')) { python "$w\a4\fetch_osm_chunk.py" $cd *> "$cd\osm_log.txt" }
Step "osm" "$cd\osm_log.txt" 'OSM_DONE'
Copy-Item "$cd\prepared\osm_*.json" $cd -Force
if (-not (Has "$cd\build_log.txt" '^BUILT')) {
    & $bl -b $karta --python "$w\wiehl_data\build_blend.py" -- "$cd\chunk.blend" "$cd\prepared" *> "$cd\build_log.txt"
}
Step "build" "$cd\build_log.txt" '^BUILT'
New-Item -ItemType Directory -Force $out | Out-Null
if (-not (Has "$cd\textures_log.txt" 'TEXTURES_DONE') -or -not (Test-Path "$out\cells.json")) {
    python "$w\a4\unity_textures_chunk.py" $out $cd *> "$cd\textures_log.txt"
}
Step "textures" "$cd\textures_log.txt" 'TEXTURES_DONE'
Copy-Item "$w\textures\nrw_*.png" "$out\Textures\" -Force
foreach ($p in 0, 1, 2) {
    if (-not (Has "$cd\export_log_$p.txt" 'EXPORT_DONE')) {
        & $bl -b "$cd\chunk.blend" --python "$w\wiehl_data\export_region.py" -- $out "$cd\prepared" all "part=$p/3" *> "$cd\export_log_$p.txt"
    }
    Step "export $p" "$cd\export_log_$p.txt" 'EXPORT_DONE'
}
"CHUNK $k DONE"
