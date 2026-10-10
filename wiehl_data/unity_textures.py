"""Textures for the Unity export (system python + PIL): orthophoto tiles, and final-colour versions of textures that
Blender adjusts in the shader (Unity's FBX import only takes the image). Output: A:\\blender_work\\unity_export\\tex"""
import os
import numpy as np
from PIL import Image

Image.MAX_IMAGE_PIXELS = None
TEX = r"A:\blender_work\textures"
OUT = r"A:\blender_work\unity_export\tex"
os.makedirs(OUT, exist_ok=True)
TILES_X, TILES_Y, OVERLAP = 3, 2, 16


def srgb_pow(img, gamma):
    a = np.asarray(img, np.float32) / 255
    return Image.fromarray((np.clip(a ** gamma, 0, 1) * 255).astype(np.uint8))


# orthophoto tiles with a small overlap; the Blender ground material darkens it with gamma 1.2, and Unity has no
# tonemapping, so pale concrete is also scaled down or it burns out to white under the sun
ortho = srgb_pow(Image.open(r"A:\blender_work\wiehl_data\prepared\ortho.jpg").convert("RGB"), 1.2)
ortho = Image.fromarray((np.asarray(ortho, np.float32) * 0.85).astype(np.uint8))
W, H = ortho.size
for j in range(TILES_Y):
    for i in range(TILES_X):
        x0, x1 = W * i // TILES_X - OVERLAP, W * (i + 1) // TILES_X + OVERLAP
        # image row 0 is north = top of UV (v = 1): tile j counts from the south
        y0, y1 = H * (TILES_Y - 1 - j) // TILES_Y - OVERLAP, H * (TILES_Y - j) // TILES_Y + OVERLAP
        ortho.crop((max(0, x0), max(0, y0), min(W, x1), min(H, y1))).save(os.path.join(OUT, f"ground_{i}{j}.jpg"), quality=93)


def desaturate(name, sat, out):
    img = Image.open(os.path.join(TEX, name)).convert("RGB")
    a = np.asarray(img, np.float32)
    grey = a @ np.array([0.2126, 0.7152, 0.0722])
    Image.fromarray(np.clip(grey[..., None] + (a - grey[..., None]) * sat, 0, 255).astype(np.uint8)).save(os.path.join(OUT, out), quality=93)


desaturate("concrete_pavers_02_Diffuse.jpg", 0.15, "pavers_grey.jpg")
desaturate("brushed_concrete_Diffuse.jpg", 0.15, "curb_grey.jpg")


def tint_rgba(name, target, out):
    """scale an RGBA card so its opaque pixels average the target colour (sRGB 0-1)"""
    a = np.asarray(Image.open(os.path.join(TEX, name)).convert("RGBA"), np.float32) / 255
    solid = a[..., 3] > 0.5
    k = np.array(target) / np.maximum(a[solid][:, :3].mean(0), 1e-3)
    a[..., :3] = np.clip(a[..., :3] * k, 0, 1)
    Image.fromarray((a * 255).astype(np.uint8)).save(os.path.join(OUT, out))


tint_rgba("nrw_leaves.png", (0.42, 0.56, 0.22), "foliage_leaves.png")
tint_rgba("nrw_leaves.png", (0.46, 0.6, 0.24), "foliage_bush.png")
tint_rgba("nrw_needles.png", (0.26, 0.4, 0.2), "foliage_needles.png")

hedge = np.asarray(Image.open(os.path.join(TEX, "leafy_grass_Diffuse.jpg")).convert("RGB"), np.float32)
Image.fromarray(np.clip(hedge * np.array([0.55, 0.8, 0.45]), 0, 255).astype(np.uint8)).save(os.path.join(OUT, "hedge.jpg"), quality=93)

# chain-link fence: dark green wires on transparent, 20 cm mesh, 256 px per metre
S = 256
yy, xx = np.mgrid[0:S, 0:S] / S * 5
wire = (np.minimum(xx % 1, 1 - xx % 1) < 0.06) | (np.minimum(yy % 1, 1 - yy % 1) < 0.06)
fence = np.zeros((S, S, 4), np.uint8)
fence[wire] = (20, 45, 25, 255)
Image.fromarray(fence).save(os.path.join(OUT, "fence.png"))
print("TEXTURES", sorted(os.listdir(OUT)))
