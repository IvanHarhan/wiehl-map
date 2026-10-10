# The main region rebuilt with Autobahn support and the outline grown along the A4 (extend_region_mask.py);
# only the tiles the A4 runs through are exported again (into unity_export\region, over the old ones).
$bl = "C:\Program Files (x86)\Steam\steamapps\common\Blender\blender.exe"
$w = "A:\blender_work"
$karta = "C:\Users\harha\Documents\" + (-join ([char[]](0x43A, 0x430, 0x440, 0x442, 0x430))) + ".blend"
$log = "$w\a4\region_rebuild.txt"
function Has($f, $m) { (Test-Path $f) -and [bool](Select-String -Path $f -Pattern $m -Quiet) }
"start $(Get-Date -Format HH:mm)" | Set-Content $log
Copy-Item "$w\wiehl_territory.blend" "$w\wiehl_territory_prev.blend" -Force
& $bl -b $karta --python "$w\wiehl_data\build_blend.py" -- "$w\wiehl_territory.blend" "$w\region_data\prepared" *> "$w\build_log.txt"
if (-not (Has "$w\build_log.txt" '^BUILT')) { "BUILD FAILED" | Add-Content $log; exit 1 }
"build ok $(Get-Date -Format HH:mm)" | Add-Content $log
Copy-Item "$w\textures\nrw_*.png" "$w\unity_export\region\Textures\" -Force
$env:ONLY_TILES = "393_5647,393_5648,394_5647,394_5648,395_5647,395_5648,396_5647,397_5647,398_5647,399_5647,400_5647,400_5648,401_5648"
& $bl -b "$w\wiehl_territory.blend" --python "$w\wiehl_data\export_region.py" -- "$w\unity_export\region" "$w\region_data\prepared" all *> "$w\a4\export_region_a4.txt"
if (-not (Has "$w\a4\export_region_a4.txt" 'EXPORT_DONE')) { "EXPORT FAILED" | Add-Content $log; exit 1 }
"EXPORT_DONE $(Get-Date -Format HH:mm)" | Add-Content $log
