# full region pipeline: Blender build -> 3 export parts -> copy into Unity -> scene + gameplay (stops at the first failure)
$bl = "C:\Program Files (x86)\Steam\steamapps\common\Blender\blender.exe"
$un = "C:\Program Files\Unity\Hub\Editor\6000.2.10f1\Editor\Unity.exe"
$w = "A:\blender_work"
# the user's Documents map file, spelled in char codes: Windows PowerShell reads this BOM-less file as ANSI and would mangle it
$karta = "C:\Users\harha\Documents\" + (-join ([char[]](0x43A, 0x430, 0x440, 0x442, 0x430))) + ".blend"
Copy-Item "$w\wiehl_territory.blend" "$w\wiehl_territory_prev.blend" -Force
& $bl -b $karta --python "$w\wiehl_data\build_blend.py" -- "$w\wiehl_territory.blend" "$w\region_data\prepared" *> "$w\build_log.txt"
if (-not (Select-String -Path "$w\build_log.txt" -Pattern '^BUILT' -Quiet)) { "BUILD FAILED"; exit 1 }
"BUILD ok"
# the build draws sign pictures into A:\blender_work\textures; the export and Unity look for them beside the map
Copy-Item "$w\textures\nrw_*.png" "$w\unity_export\region\Textures\" -Force
Copy-Item "$w\textures\nrw_*.png" "C:\dexstand\Assets\Maps\Import\Textures\" -Force
foreach ($k in 0, 1, 2) {
    & $bl -b "$w\wiehl_territory.blend" --python "$w\wiehl_data\export_region.py" -- "$w\unity_export\region" "$w\region_data\prepared" all "part=$k/3" *> "$w\export_log_$k.txt"
    if (-not (Select-String -Path "$w\export_log_$k.txt" -Pattern 'EXPORT_DONE' -Quiet)) { "EXPORT $k FAILED"; exit 1 }
    "EXPORT $k ok"
}
Copy-Item "$w\unity_export\region\Tile_*.fbx", "$w\unity_export\region\region.json" "C:\dexstand\Assets\Maps\Import\" -Force
New-Item -ItemType Directory -Force "C:\dexstand\Assets\Resources\Map" | Out-Null
Copy-Item "$w\unity_export\region\parked.json", "$w\unity_export\region\trees.bytes", "$w\unity_export\region\tree_templates.json", "$w\unity_export\region\parked_drive_*.json" "C:\dexstand\Assets\Resources\Map\" -Force
python "$w\region_data\make_minimap.py" "$w\region_data\prepared" "C:\dexstand\Assets\Resources\Minimap" "$w\unity_export\region\region.json"
# touched, the orthophotos go through the importer again (it sets their phone size)
Get-ChildItem "C:\dexstand\Assets\Maps\Import\Textures\ortho_*.jpg" | ForEach-Object { $_.LastWriteTime = Get-Date }
& $un -batchmode -quit -projectPath C:\dexstand -executeMethod DexstandSceneBuilder.BuildScene -logFile "$w\unity_scene_log.txt" | Out-Null
"SCENE exit $LASTEXITCODE"
if ($LASTEXITCODE -ne 0) { exit 1 }
& $un -batchmode -quit -projectPath C:\dexstand -executeMethod DexstandGameplayBuilder.BuildGameplay -logFile "$w\unity_gameplay_log.txt" | Out-Null
"GAMEPLAY exit $LASTEXITCODE"
if ($args -contains "apk") {
    $env:GRADLE_USER_HOME = "A:\gradle"
    & $un -batchmode -quit -projectPath C:\dexstand -executeMethod DexstandBuild.BuildAndroid -logFile "$w\apk_log.txt" | Out-Null
    Select-String -Path "$w\apk_log.txt" -Pattern '\[Build\] (APK OK|FAILED)' | ForEach-Object { $_.Line }
}
