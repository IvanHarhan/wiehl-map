# The rest of run_all.ps1 after a Blender build that is already running: waits for it, then export, Unity scene
# and gameplay, then the PC build (no APK). Used when run_all had to be stopped mid-build.
$bl = "C:\Program Files (x86)\Steam\steamapps\common\Blender\blender.exe"
$un = "C:\Program Files\Unity\Hub\Editor\6000.2.10f1\Editor\Unity.exe"
$w = "A:\blender_work"
while (Get-Process blender -ErrorAction SilentlyContinue) { Start-Sleep 10 }
# the build's log went with the stopped run_all: the map file being newer than the build's start says it finished
if ((Get-Item "$w\wiehl_territory.blend").LastWriteTime -lt [datetime]"2026-09-25 14:04:56") { "BUILD FAILED"; exit 1 }
"BUILD ok"
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
Get-ChildItem "C:\dexstand\Assets\Maps\Import\Textures\ortho_*.jpg" | ForEach-Object { $_.LastWriteTime = Get-Date }
& $un -batchmode -quit -projectPath C:\dexstand -executeMethod DexstandSceneBuilder.BuildScene -logFile "$w\unity_scene_log.txt" | Out-Null
"SCENE exit $LASTEXITCODE"
if ($LASTEXITCODE -ne 0) { exit 1 }
& $un -batchmode -quit -projectPath C:\dexstand -executeMethod DexstandGameplayBuilder.BuildGameplay -logFile "$w\unity_gameplay_log.txt" | Out-Null
"GAMEPLAY exit $LASTEXITCODE"
& powershell -NoProfile -ExecutionPolicy Bypass -File "$w\build_exe.ps1"
