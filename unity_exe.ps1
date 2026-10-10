# Unity only (the map export is unchanged): scene, gameplay, then the PC build
$un = "C:\Program Files\Unity\Hub\Editor\6000.2.10f1\Editor\Unity.exe"
$w = "A:\blender_work"
& $un -batchmode -quit -projectPath C:\dexstand -executeMethod DexstandSceneBuilder.BuildScene -logFile "$w\unity_scene_log.txt" | Out-Null
"SCENE exit $LASTEXITCODE"
Select-String -Path "$w\unity_scene_log.txt" -Pattern 'error CS\d+|Shader error in' | Select-Object -First 8 | ForEach-Object { $_.Line }
if ($LASTEXITCODE -ne 0) { exit 1 }
& $un -batchmode -quit -projectPath C:\dexstand -executeMethod DexstandGameplayBuilder.BuildGameplay -logFile "$w\unity_gameplay_log.txt" | Out-Null
"GAMEPLAY exit $LASTEXITCODE"
& powershell -NoProfile -ExecutionPolicy Bypass -File "$w\build_exe.ps1"
