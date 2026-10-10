"""Live link. Blender runs every .py that appears in live/inbox (one per half second)
and writes its output to live/outbox/<name>.log. Start: blender.exe file.blend --python blender_live.py,
or Scripting > Open > Run Script in an open Blender."""
import bpy, contextlib, glob, io, math, os, traceback
from mathutils import Vector

ROOT = r"A:\blender_work\live"
INBOX, OUTBOX = os.path.join(ROOT, "inbox"), os.path.join(ROOT, "outbox")
os.makedirs(INBOX, exist_ok=True)
os.makedirs(OUTBOX, exist_ok=True)


def view3d():
    """(window, area, region, space) of the largest 3D viewport."""
    best = None
    for win in bpy.context.window_manager.windows:
        for area in win.screen.areas:
            if area.type == "VIEW_3D" and (best is None or area.width * area.height > best[1].width * best[1].height):
                region = next(r for r in area.regions if r.type == "WINDOW")
                best = (win, area, region, area.spaces.active)
    return best


def aim(target, azimuth=212, elevation=30, distance=420, lens=35):
    """Orbit the viewport around `target`, looking from compass `azimuth` (0 = north, 90 = east)."""
    win, area, region, space = view3d()
    az, el = math.radians(azimuth), math.radians(elevation)
    eye_dir = Vector((math.sin(az) * math.cos(el), math.cos(az) * math.cos(el), math.sin(el)))
    r3d = space.region_3d
    r3d.view_perspective = "PERSP"
    r3d.view_location = Vector(target)
    r3d.view_rotation = eye_dir.to_track_quat("Z", "Y")
    r3d.view_distance = distance
    space.lens = lens
    area.tag_redraw()


def shot(path):
    """Screenshot of the 3D viewport exactly as the user sees it."""
    win, area, region, space = view3d()
    with bpy.context.temp_override(window=win, area=area, region=region):
        bpy.ops.wm.redraw_timer(type="DRAW_WIN_SWAP", iterations=1)
        bpy.ops.screen.screenshot_area(filepath=path)


def _tick():
    # one command per tick, so the UI redraws between commands
    files = sorted(glob.glob(os.path.join(INBOX, "*.py")))
    if not files:
        return 0.5
    f = files[0]
    name = os.path.splitext(os.path.basename(f))[0]
    out = io.StringIO()
    try:
        with open(f, encoding="utf-8") as fh:
            src = fh.read()
        os.remove(f)
        with contextlib.redirect_stdout(out):
            exec(compile(src, f, "exec"), {"__name__": "__live__", "bpy": bpy, "Vector": Vector, "math": math,
                                           "view3d": view3d, "aim": aim, "shot": shot})
        status = "OK"
    except Exception:
        out.write(traceback.format_exc())
        status = "ERROR"
    with open(os.path.join(OUTBOX, name + ".log"), "w", encoding="utf-8") as fh:
        fh.write(status + "\n" + out.getvalue())
    return 0.5


# running the script again replaces the old timer instead of adding a second one
_old = bpy.app.driver_namespace.get("live_link_tick")
if _old is not None and bpy.app.timers.is_registered(_old):
    bpy.app.timers.unregister(_old)
bpy.app.driver_namespace["live_link_tick"] = _tick
bpy.app.timers.register(_tick, first_interval=0.5, persistent=True)
print("Live link on:", INBOX)
