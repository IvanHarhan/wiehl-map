"""Run with --python before the first redraw: keep the region's 60 orthophoto tiles within a laptop GPU's memory."""
import bpy

bpy.context.preferences.system.gl_texture_limit = "CLAMP_2048"
