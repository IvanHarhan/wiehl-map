# PC build after an APK: frees the Android build caches on C: first (they regrow on the next APK)
$un = "C:\Program Files\Unity\Hub\Editor\6000.2.10f1\Editor\Unity.exe"
foreach ($p in "C:\dexstand\Library\Bee\Android", "C:\dexstand\Library\Bee\artifacts\Android") {
    if (Test-Path $p) { Remove-Item $p -Recurse -Force -ErrorAction SilentlyContinue }
}
& $un -batchmode -quit -projectPath C:\dexstand -executeMethod DexstandBuild.BuildWindows -logFile "A:\blender_work\exe_log.txt" | Out-Null
Select-String -Path "A:\blender_work\exe_log.txt" -Pattern '\[Build\] (EXE OK|FAILED)' | ForEach-Object { $_.Line }
# a zip to send a friend: the whole folder except Burst's debug symbols (not needed to play)
if (Test-Path "A:\DexstandBuild\Dexstand.exe") {
    Remove-Item "A:\DexstandBuild\Dexstand_BurstDebugInformation_DoNotShip" -Recurse -Force -ErrorAction SilentlyContinue
    Remove-Item "A:\Dexstand_PC.zip" -Force -ErrorAction SilentlyContinue
    Add-Type -AssemblyName System.IO.Compression.FileSystem
    [IO.Compression.ZipFile]::CreateFromDirectory("A:\DexstandBuild", "A:\Dexstand_PC.zip", [IO.Compression.CompressionLevel]::Optimal, $true)
    "ZIP {0:N2} GB" -f ((Get-Item "A:\Dexstand_PC.zip").Length / 1GB)
}
