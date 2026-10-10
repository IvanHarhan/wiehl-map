$t = Get-Content A:\blender_work\a4\chunk_3\territory_tiles.json | ConvertFrom-Json
while ($true) {
    $tot = 0
    foreach ($p in 'dgm1', 'dom1', 'dop10', 'lod2') { foreach ($x in $t) {
        if (Get-ChildItem "A:\blender_work\a4\tiles\$p" -Filter "*_32_$($x[0])_$($x[1])_*" -ErrorAction SilentlyContinue | Where-Object { $_.Extension -ne '.part' }) { $tot++ } } }
    if ($tot -ge 40) { break }
    Start-Sleep 20
}
& powershell -NoProfile -ExecutionPolicy Bypass -File A:\blender_work\a4\run_chunk.ps1 3 *> A:\blender_work\a4\chunk_3\run_out.txt
