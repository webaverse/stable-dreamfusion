# using blender, open an input.obj file, and export it as output.glb
import bpy
# import input.obj file
bpy.ops.import_scene.obj(filepath='trial/mesh/mesh.obj')

# select all
bpy.ops.object.select_all(action='SELECT')

# export output.glb file
bpy.ops.export_scene.gltf(filepath="model.glb")
print('done')
