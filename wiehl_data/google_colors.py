"""Run from build_blend.py (exec) right after NRW_Buildings mesh `me` is built.
Per building: ray-cast every wall onto the Google 3D Tiles mesh, read the photo texture there,
take the median facade colour -> face attributes `gcol` (linear colour) and `brick` (0/1)."""
FALLBACK = np.array([0.80, 0.76, 0.68])


def srgb_to_linear(c):
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


pcount = len(me.polygons)
p_bid = np.empty(pcount, np.int32); me.attributes["bid"].data.foreach_get("value", p_bid)
p_mat = np.empty(pcount, np.int32); me.polygons.foreach_get("material_index", p_mat)
p_ctr = np.empty(pcount * 3, np.float32); me.polygons.foreach_get("center", p_ctr); p_ctr = p_ctr.reshape(-1, 3)
p_nrm = np.empty(pcount * 3, np.float32); me.polygons.foreach_get("normal", p_nrm); p_nrm = p_nrm.reshape(-1, 3)
p_area = np.empty(pcount, np.float32); me.polygons.foreach_get("area", p_area)

samples = [[] for _ in range(n_b)]
if google:
    dg = bpy.context.evaluated_depsgraph_get()
    gev = google.evaluated_get(dg)
    gm = gev.to_mesh()
    gm.calc_loop_triangles()
    nt = len(gm.loop_triangles)
    t_loops = np.empty(nt * 3, np.int32); gm.loop_triangles.foreach_get("loops", t_loops); t_loops = t_loops.reshape(-1, 3)
    t_verts = np.empty(nt * 3, np.int32); gm.loop_triangles.foreach_get("vertices", t_verts); t_verts = t_verts.reshape(-1, 3)
    t_mat = np.empty(nt, np.int32); gm.loop_triangles.foreach_get("material_index", t_mat)
    g_co = np.empty(len(gm.vertices) * 3, np.float32); gm.vertices.foreach_get("co", g_co); g_co = g_co.reshape(-1, 3)
    M = np.array(gev.matrix_world)
    g_co = g_co @ M[:3, :3].T + M[:3, 3]
    g_uv = np.empty(len(gm.loops) * 2, np.float32); gm.uv_layers.active.data.foreach_get("uv", g_uv); g_uv = g_uv.reshape(-1, 2)
    tree = BVHTree.FromPolygons(g_co.tolist(), t_verts.tolist(), all_triangles=True)

    mat_img = {}
    for i, m in enumerate(gm.materials):
        img = next((n.image for n in (m.node_tree.nodes if m and m.node_tree else []) if n.type == "TEX_IMAGE" and n.image), None)
        mat_img[i] = img
    pix_cache = {}

    def pixels(img):
        if img.name not in pix_cache:
            w, h = img.size
            a = np.empty(w * h * 4, np.float32); img.pixels.foreach_get(a)
            pix_cache[img.name] = a.reshape(h, w, 4)[:, :, :3]
        return pix_cache[img.name]

    walls = np.nonzero((p_mat == 1) & (p_area > 3.0) & (np.abs(p_nrm[:, 2]) < 0.3))[0]
    for pi in walls:
        c, n = Vector(p_ctr[pi]), Vector(p_nrm[pi])
        for dz in (-1.0, 0.0, 1.0):
            o = c + Vector((0, 0, dz))
            loc, hn, ti, dist = tree.ray_cast(o + n * 2.5, -n, 5.0)
            if ti is None or hn.dot(n) < 0.3:
                continue
            img = mat_img.get(t_mat[ti])
            if img is None:
                continue
            a, b_, cc = (Vector(g_co[v]) for v in t_verts[ti])
            v0, v1, v2 = b_ - a, cc - a, loc - a
            d00, d01, d11, d20, d21 = v0.dot(v0), v0.dot(v1), v1.dot(v1), v2.dot(v0), v2.dot(v1)
            den = d00 * d11 - d01 * d01
            if abs(den) < 1e-12:
                continue
            w1 = (d11 * d20 - d01 * d21) / den; w2 = (d00 * d21 - d01 * d20) / den
            uv = g_uv[t_loops[ti]].T @ np.array([1 - w1 - w2, w1, w2])
            px = pixels(img)
            h, w = px.shape[:2]
            col_s = px[int(np.clip(uv[1], 0, 0.9999) * h), int(np.clip(uv[0], 0, 0.9999) * w)]
            r, g, bb = col_s
            if g > r * 1.08 and g > bb * 1.08:  # foliage in front of the wall
                continue
            samples[p_bid[pi]].append(col_s)
    gev.to_mesh_clear()

bcol = np.tile(FALLBACK, (n_b, 1))
brick = np.zeros(n_b, np.float32)
hit_b = 0
med = {i: np.median(np.array(s), axis=0) for i, s in enumerate(samples) if len(s) >= 3}
# Google tiles have a purple cast: gray-world balance so the average facade is warm off-white
wb = np.array([0.80, 0.77, 0.71]) / np.maximum(np.mean(list(med.values()), axis=0), 1e-3) if med else np.ones(3)
for i, c in med.items():
    hit_b += 1
    c = np.clip(c * wb, 0, 1)
    grey = c.mean()
    c = np.clip(grey + (c - grey) * 1.6, 0, 1)
    # Google photos carry baked shadow: lift brightness, keep hue
    c = np.clip(c * (0.62 / max(c.max(), 1e-3)) ** 0.5 * 1.1, 0, 1) if c.max() < 0.62 else c
    brick[i] = 1.0 if (c[0] > c[1] * 1.25 and c[0] > c[2] * 1.35 and c.max() < 0.6) else 0.0
    bcol[i] = srgb_to_linear(c)

a = me.attributes.new("gcol", "FLOAT_COLOR", "FACE")
a.data.foreach_set("color", np.c_[bcol[p_bid], np.ones(pcount)].astype(np.float32).ravel())
a = me.attributes.new("brick", "FLOAT", "FACE")
a.data.foreach_set("value", brick[p_bid])
print("GCOLORS buildings_with_colour", hit_b, "of", n_b, "brick", int(brick.sum()))
