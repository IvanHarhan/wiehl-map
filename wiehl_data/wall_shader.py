"""exec'd by build_blend.py: the wall material. Plaster/brick tinted per building, punched windows, doors, shop fronts,
glass store fronts, halls, schools, a historic facade. Kinds (face attribute "kind"): 0 dwelling, 1 commercial/office,
2 blank (garage/shed), 3 church, 4 glass front, 5 hall, 6 school, 7 plain store wall, 8 historic.
Uses build_blend globals: Nodes, band."""


def wall_material():
    m = bpy.data.materials.new("NRW_Wall")
    m.use_nodes = True
    nn = Nodes(m)
    uv = nn.uv("WallUV")
    sep = nn.new("ShaderNodeSeparateXYZ")
    nn.link(uv, sep.inputs[0])
    u, v = sep.outputs[0], sep.outputs[1]

    def packed(name):
        s = nn.new("ShaderNodeSeparateXYZ")
        nn.link(nn.new("ShaderNodeAttribute", attribute_name=name).outputs["Vector"], s.inputs[0])
        return s.outputs[0], s.outputs[1], s.outputs[2]

    # per-face data comes packed into four vectors: EEVEE binds only ~16 vertex attributes per material
    wlen, wbot, weave = packed("w_geo")
    door_u, door_t, height = packed("w_door")
    kind, shop, band_on = packed("w_kind")
    white, is_brick, dcol = packed("w_look")

    def mul(*xs):
        out = xs[0]
        for x in xs[1:]:
            out = nn.math("MULTIPLY", out, x)
        return out

    def any_of(*xs):
        out = xs[0]
        for x in xs[1:]:
            out = nn.math("ADD", out, x)
        return nn.math("MINIMUM", out, 1.0)

    def one_minus(x):
        return nn.math("SUBTRACT", 1.0, x)

    def fract_div(x, d):
        return nn.math("FRACT", nn.math("DIVIDE", x, d))

    def lt(x, y):
        return nn.math("LESS_THAN", x, y)

    def gt(x, y):
        return nn.math("GREATER_THAN", x, y)

    def palette(index, colours):
        ramp = nn.new("ShaderNodeValToRGB")
        ramp.color_ramp.interpolation = "CONSTANT"
        els = ramp.color_ramp.elements
        els[0].color = (*colours[0], 1)
        els[1].position, els[1].color = 1 / len(colours), (*colours[1], 1)
        for i in range(2, len(colours)):
            els.new(i / len(colours)).color = (*colours[i], 1)
        nn.link(nn.math("DIVIDE", nn.math("ADD", index, 0.5), len(colours)), ramp.inputs[0])
        return ramp.outputs["Color"]

    blank, church, glassk, hall, school, store, historic = (band(nn, kind, k - 0.5, k + 0.5) for k in (2, 3, 4, 5, 6, 7, 8))

    # plaster tinted with the building colour, brick where detected; halls, store walls and "white" faces get light render
    gcol = nn.new("ShaderNodeAttribute", attribute_name="gcol").outputs["Color"]
    pbw = nn.new("ShaderNodeRGBToBW")
    nn.link(nn.tex("painted_plaster_wall", "Diffuse", uv), pbw.inputs[0])
    pbright = nn.new("ShaderNodeBrightContrast")
    pbright.inputs["Bright"].default_value = 0.2
    pbright.inputs["Contrast"].default_value = -0.1
    nn.link(pbw.outputs[0], pbright.inputs["Color"])

    def tinted(colour):
        mx = nn.new("ShaderNodeMix", data_type="RGBA", blend_type="MULTIPLY")
        mx.inputs[0].default_value = 1.0
        nn.link(pbright.outputs[0], mx.inputs[6])
        if isinstance(colour, tuple):
            mx.inputs[7].default_value = colour
        else:
            nn.link(colour, mx.inputs[7])
        return mx.outputs[2]

    wall_col = nn.mix(is_brick, tinted(gcol), nn.tex("red_brick_03", "Diffuse", uv))
    wall_rough = nn.mix(is_brick, nn.tex("painted_plaster_wall", "Rough", uv, False), nn.tex("red_brick_03", "Rough", uv, False))
    wall_nor = nn.mix(is_brick, nn.tex("painted_plaster_wall", "nor_gl", uv, False), nn.tex("red_brick_03", "nor_gl", uv, False))
    wall_col = nn.mix(any_of(hall, store, white), wall_col, tinted((0.88, 0.88, 0.86, 1)))
    # store walls: big facade panels
    joints = mul(store, nn.math("MAXIMUM", lt(fract_div(v, 1.25), 0.012), lt(fract_div(u, 3.0), 0.004)))
    wall_col = nn.mix(joints, wall_col, (0.45, 0.45, 0.44, 1))

    # window grid: whole number of bays per facade, 3 m floors, only full floors below the eave
    res_like = nn.math("MAXIMUM", lt(kind, 0.5), historic)
    spacing = nn.math("SUBTRACT", 4.5, nn.math("MULTIPLY", res_like, 1.1))
    nwin = nn.math("MAXIMUM", nn.math("FLOOR", nn.math("DIVIDE", wlen, spacing)), 1.0)
    fu = fract_div(u, nn.math("DIVIDE", wlen, nwin))
    fv = fract_div(v, 3.0)
    fl = nn.math("MULTIPLY", nn.math("FLOOR", nn.math("DIVIDE", v, 3.0)), 3.0)
    ok_floor = mul(gt(fl, nn.math("SUBTRACT", wbot, 0.3)), lt(nn.math("ADD", fl, 3.0), nn.math("ADD", weave, 0.3)), gt(height, 4.0))
    ok_church = mul(gt(v, nn.math("ADD", wbot, 2.0)), lt(v, nn.math("MINIMUM", nn.math("SUBTRACT", weave, 1.0), nn.math("ADD", wbot, 9.0))))
    enabled = mul(one_minus(blank), gt(wlen, 1.8))

    def windows(fu_lo, fu_hi, fv_lo, fv_hi, cu_lo, cu_hi):
        res = mul(ok_floor, band(nn, fu, fu_lo, fu_hi), band(nn, fv, fv_lo, fv_hi))
        ch = mul(ok_church, band(nn, fu, cu_lo, cu_hi))
        return mul(nn.math("ADD", mul(res, one_minus(church)), mul(ch, church)), enabled)

    rel = nn.math("SUBTRACT", v, wbot)
    ground_floor = lt(rel, 3.0)
    dx = nn.math("ABSOLUTE", nn.math("SUBTRACT", u, door_u))
    edge_ok = band(nn, u, 0.25, nn.math("SUBTRACT", wlen, 0.25))

    # historic facade: stone plinth, cornices at every floor and the eave, quoins at the corners
    STONE = (0.8, 0.78, 0.72, 1)
    above_plinth = gt(rel, 0.9)
    cornice = mul(historic, nn.math("MAXIMUM", gt(fv, 0.955), gt(v, nn.math("SUBTRACT", weave, 0.4))), above_plinth)
    quoin = mul(historic, nn.math("MAXIMUM", lt(u, 0.7), gt(u, nn.math("SUBTRACT", wlen, 0.7))), lt(fract_div(v, 0.7), 0.5), above_plinth)
    wall_col = nn.mix(mul(lt(rel, 0.5), one_minus(glassk), one_minus(historic)), wall_col, (0.2, 0.2, 0.2, 1))
    wall_col = nn.mix(mul(historic, lt(rel, 0.9)), wall_col, (0.42, 0.4, 0.37, 1))
    wall_col = nn.mix(cornice, wall_col, STONE)
    wall_col = nn.mix(quoin, wall_col, STONE)

    # a little grime: fading up from the ground plus faint vertical streaks
    streak_uv = nn.new("ShaderNodeCombineXYZ")
    nn.link(nn.math("MULTIPLY", u, 3.0), streak_uv.inputs[0])
    nn.link(nn.math("MULTIPLY", v, 0.25), streak_uv.inputs[1])
    streak = nn.new("ShaderNodeTexNoise")
    streak.inputs["Detail"].default_value = 3.0
    nn.link(streak_uv.outputs[0], streak.inputs["Vector"])
    dirt = nn.math("ADD", nn.math("MULTIPLY", nn.math("MAXIMUM", one_minus(nn.math("DIVIDE", rel, 1.6)), 0.0), 0.12),
                   nn.math("MULTIPLY", gt(streak.outputs["Fac"], 0.62), 0.05))
    wall_col = nn.mix(mul(dirt, one_minus(glassk)), wall_col, (0.1, 0.09, 0.08, 1))

    # punched windows: not in the door bay, not on a shop's ground floor, not on glass/hall/school/store walls
    door_bay = mul(band(nn, door_t, 0.5, 1.5), ground_floor, lt(dx, nn.math("MULTIPLY", nn.math("DIVIDE", wlen, nwin), 0.5)))
    plain = mul(one_minus(door_bay), one_minus(mul(shop, ground_floor)), one_minus(any_of(glassk, hall, school, store)))
    frame = mul(windows(0.26, 0.74, 0.26, 0.84, 0.38, 0.62), plain)
    glass = mul(windows(0.29, 0.71, 0.29, 0.81, 0.41, 0.59), plain)
    h_sur = mul(windows(0.2, 0.8, 0.2, 0.9, 0.38, 0.62), plain, historic)
    bars = mul(glass, historic, nn.math("MAXIMUM", lt(nn.math("ABSOLUTE", nn.math("SUBTRACT", fu, 0.5)), 0.012),
                                        lt(nn.math("ABSOLUTE", nn.math("SUBTRACT", fv, 0.62)), 0.012)))
    # school: ribbon windows with mullions every 1.35 m
    s_frame = mul(school, ok_floor, band(nn, fv, 0.26, 0.86), band(nn, u, 0.4, nn.math("SUBTRACT", wlen, 0.4)))
    s_glass = mul(s_frame, band(nn, fv, 0.3, 0.82), gt(fract_div(u, 1.35), 0.05))
    # hall: light walls with a high ribbon of windows under the eave
    below_eave = nn.math("SUBTRACT", weave, v)
    h_frame = mul(hall, gt(nn.math("SUBTRACT", weave, wbot), 5.0), band(nn, below_eave, 1.2, 2.4),
                  band(nn, u, 0.5, nn.math("SUBTRACT", wlen, 0.5)))
    h_glass = mul(h_frame, band(nn, below_eave, 1.27, 2.33), gt(fract_div(u, 2.0), 0.04))
    # shop front: glass bays on the ground floor, sign band above
    sbay = nn.math("DIVIDE", wlen, nn.math("MAXIMUM", nn.math("FLOOR", nn.math("DIVIDE", wlen, 3.4)), 1.0))
    sh_frame = mul(shop, band(nn, rel, 0.3, 2.95), edge_ok)
    sh_glass = mul(sh_frame, band(nn, fract_div(u, sbay), 0.06, 0.94), band(nn, rel, 0.4, 2.85))
    sign = mul(shop, band(nn, rel, 3.05, 3.6), edge_ok)
    # glass front: mullions every 1.5 m and per storey; top band red where "band" is set, dark fascia otherwise
    g_mull = mul(glassk, nn.math("MAXIMUM", lt(fract_div(u, 1.5), 0.04), lt(fract_div(v, 3.0), 0.03)))
    top = mul(glassk, gt(v, nn.math("SUBTRACT", weave, 1.1)))
    red = mul(top, band_on)
    fascia = mul(top, one_minus(band_on))
    g_base = mul(glassk, lt(rel, 0.35))
    g_glass = mul(glassk, one_minus(g_mull), one_minus(top), one_minus(g_base))
    # doors: front door (1), garage door (2), wide glass entrance (3)
    house_t, gar_t, wide_t = band(nn, door_t, 0.5, 1.5), band(nn, door_t, 1.5, 2.5), gt(door_t, 2.5)
    house = mul(house_t, lt(dx, 0.55), lt(rel, 2.25))
    house_fr = mul(house_t, lt(dx, 0.63), lt(rel, 2.33))
    house_gl = mul(house, lt(dx, 0.12), band(nn, rel, 0.9, 1.9))
    garage = mul(gar_t, lt(dx, 1.25), lt(rel, 2.1))
    garage_fr = mul(gar_t, lt(dx, 1.33), lt(rel, 2.18))
    wide = mul(wide_t, lt(dx, 1.5), lt(rel, 2.6))
    wide_fr = mul(wide_t, lt(dx, 1.62), lt(rel, 2.72))
    wide_gl = mul(wide, gt(dx, 0.03))

    GLASS = (0.1, 0.13, 0.15, 1)
    glass_all = any_of(glass, s_glass, h_glass, sh_glass, house_gl, g_glass, wide_gl)
    col = nn.mix(h_sur, wall_col, STONE)
    col = nn.mix(frame, col, (0.86, 0.86, 0.84, 1))
    col = nn.mix(s_frame, col, (0.55, 0.56, 0.58, 1))
    col = nn.mix(h_frame, col, (0.35, 0.36, 0.37, 1))
    col = nn.mix(sh_frame, col, (0.12, 0.12, 0.13, 1))
    col = nn.mix(sign, col, palette(nn.attr("scol"), [(0.55, 0.03, 0.03), (0.04, 0.1, 0.35), (0.06, 0.06, 0.06),
                                             (0.04, 0.25, 0.1), (0.65, 0.4, 0.04), (0.85, 0.85, 0.85)]))
    col = nn.mix(house_fr, col, (0.8, 0.8, 0.78, 1))
    col = nn.mix(house, col, palette(dcol, [(0.8, 0.8, 0.78), (0.1, 0.1, 0.11), (0.28, 0.16, 0.08),
                                              (0.08, 0.18, 0.12), (0.4, 0.07, 0.05)]))
    col = nn.mix(garage_fr, col, (0.7, 0.7, 0.7, 1))
    col = nn.mix(garage, col, nn.tex("painted_metal_shutter", "Diffuse", uv))
    col = nn.mix(wide_fr, col, (0.25, 0.25, 0.26, 1))
    col = nn.mix(g_mull, col, (0.18, 0.18, 0.19, 1))
    col = nn.mix(red, col, (0.55, 0.01, 0.02, 1))
    col = nn.mix(fascia, col, (0.22, 0.22, 0.23, 1))
    col = nn.mix(g_base, col, (0.1, 0.1, 0.1, 1))
    col = nn.mix(glass_all, col, GLASS)
    col = nn.mix(bars, col, (0.9, 0.9, 0.88, 1))
    nn.link(col, nn.bsdf.inputs["Base Color"])

    solid = any_of(frame, s_frame, h_frame, sh_frame, sign, glassk, house_fr, garage_fr, wide_fr, h_sur)
    rough = nn.mix(solid, wall_rough, (0.4, 0.4, 0.4, 1))
    rough = nn.mix(glass_all, rough, (0.04, 0.04, 0.04, 1))
    nn.link(rough, nn.bsdf.inputs["Roughness"])
    nn.normal(nn.mix(solid, wall_nor, (0.5, 0.5, 1.0, 1)))
    return m
