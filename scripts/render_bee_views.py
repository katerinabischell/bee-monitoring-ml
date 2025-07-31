#written by KC Seltmannn with ChatGPT near July 23, 2025
#gets 20 angles from a .glb file, exports as transparent png and adds them to a folder.
#better than .obj files b/c python importer
#This script uses blender and requires blender to be installed.
#currently has a shadow on the bee. Not sure if this is an improvement with the training data and may need to be removed.

import bpy
import os
import math
import random
import mathutils

# === CONFIGURATION ===
glb_path = "/Volumes/IMAGES/bombus_synthetic_image_data/bee_models/vos1.glb"
output_dir = "/Volumes/IMAGES/bombus_synthetic_image_data/extracted_bees/B_vos"
num_images = 20
image_width = 960
image_height = 540
camera_distance = 10

# === IMPORT GLB ===
bpy.ops.import_scene.gltf(filepath=glb_path)
bee_obj = bpy.context.selected_objects[0]
bee_obj.name = "Bee"
scene = bpy.context.scene

# === REMOVE ALL LIGHT SOURCES ===
for obj in bpy.data.objects:
    if obj.type == 'LIGHT':
        bpy.data.objects.remove(obj, do_unlink=True)

# === OPTIONAL: Disable shadow casting (if available) ===
if scene.render.engine == 'CYCLES':
    try:
        bee_obj.cycles_visibility.shadow = False
    except AttributeError:
        print("Note: bee_obj has no cycles_visibility.shadow attribute")

# === GET BOUNDING BOX CENTER ===
bbox_corners = [bee_obj.matrix_world @ mathutils.Vector(corner) for corner in bee_obj.bound_box]
bbox_center = sum(bbox_corners, mathutils.Vector()) / 8

# === RENDER SETTINGS ===
scene = bpy.context.scene
scene.render.engine = 'CYCLES'
scene.render.image_settings.file_format = 'PNG'
scene.render.image_settings.color_mode = 'RGBA'
scene.render.film_transparent = True
scene.render.resolution_x = image_width
scene.render.resolution_y = image_height
scene.render.resolution_percentage = 100

# === AMBIENT WORLD LIGHTING ===
if scene.world is None:
    scene.world = bpy.data.worlds.new("World")
scene.world.use_nodes = True
nodes = scene.world.node_tree.nodes
links = scene.world.node_tree.links
nodes.clear()

bg_node = nodes.new(type='ShaderNodeBackground')
bg_node.inputs[0].default_value = (1, 1, 1, 1)  # white
bg_node.inputs[1].default_value = 5.0           # brightness
out_node = nodes.new(type='ShaderNodeOutputWorld')
links.new(bg_node.outputs['Background'], out_node.inputs['Surface'])

# === CREATE CAMERA & TRACKER ===
cam_data = bpy.data.cameras.new("Camera")
cam = bpy.data.objects.new("Camera", cam_data)
scene.collection.objects.link(cam)
scene.camera = cam
cam.data.lens = 35

# Empty object for camera to track
target = bpy.data.objects.new("Target", None)
target.location = bbox_center
scene.collection.objects.link(target)

track = cam.constraints.new(type='TRACK_TO')
track.target = target
track.track_axis = 'TRACK_NEGATIVE_Z'
track.up_axis = 'UP_Y'

# === ENSURE OUTPUT DIR EXISTS ===
os.makedirs(output_dir, exist_ok=True)

# === RENDER MULTIPLE ANGLES ===
for i in range(num_images):
    theta = random.uniform(0, 2 * math.pi)
    phi = random.uniform(math.radians(20), math.radians(75))

    x = camera_distance * math.sin(phi) * math.cos(theta)
    y = camera_distance * math.sin(phi) * math.sin(theta)
    z = camera_distance * math.cos(phi)

    cam.location = bbox_center + mathutils.Vector((x, y, z))

    scene.render.filepath = os.path.join(output_dir, f"bee2_{i:03d}.png")#changes so dont overwrite if going to same folder
    bpy.ops.render.render(write_still=True)

print(f"✅ Rendered {num_images} views to {output_dir}")
