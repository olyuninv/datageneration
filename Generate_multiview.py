# Code based on the following sources:

# @INPROCEEDINGS{varol17_surreal,
#   title     = {Learning from Synthetic Humans},
#   author    = {Varol, G{\"u}l and Romero, Javier and Martin, Xavier and Mahmood, Naureen and Black, Michael J. and Laptev, Ivan and Schmid, Cordelia},
#   booktitle = {CVPR},
#   year      = {2017},
#   url       = {https://github.com/gulvarol/surreal}
# }

# @article{SMPL:2015,
#       author = {Loper, Matthew and Mahmood, Naureen and Romero, Javier and Pons-Moll, Gerard and Black, Michael J.},
#       title = {{SMPL}: A Skinned Multi-Person Linear Model},
#       journal = {ACM Trans. Graphics (Proc. SIGGRAPH Asia)},
#       month = oct,
#       number = {6},
#       pages = {248:1--248:16},
#       publisher = {ACM},
#       volume = {34},
#       year = {2015},
#       url = {http://smpl.is.tue.mpg.de}
#     }

import sys
import os
import random
import math
import bpy
import numpy as np
from os import getenv
from os import remove
from os.path import join, dirname, realpath, exists
from mathutils import Matrix, Vector, Quaternion, Euler
from glob import glob
from random import choice, randint, uniform, normalvariate
from pickle import load
from bpy_extras.object_utils import world_to_camera_view as world2cam

sys.path.insert(0, ".")

sorted_parts = ['hips','leftUpLeg','rightUpLeg','spine','leftLeg','rightLeg',
                'spine1','leftFoot','rightFoot','spine2','leftToeBase','rightToeBase',
                'neck','leftShoulder','rightShoulder','head','leftArm','rightArm',
                'leftForeArm','rightForeArm','leftHand','rightHand','leftHandIndex1' ,'rightHandIndex1']
# order
part_match = {'root':'root', 'bone_00':'Pelvis', 'bone_01':'L_Hip', 'bone_02':'R_Hip',
              'bone_03':'Spine1', 'bone_04':'L_Knee', 'bone_05':'R_Knee', 'bone_06':'Spine2',
              'bone_07':'L_Ankle', 'bone_08':'R_Ankle', 'bone_09':'Spine3', 'bone_10':'L_Foot',
              'bone_11':'R_Foot', 'bone_12':'Neck', 'bone_13':'L_Collar', 'bone_14':'R_Collar',
              'bone_15':'Head', 'bone_16':'L_Shoulder', 'bone_17':'R_Shoulder', 'bone_18':'L_Elbow',
              'bone_19':'R_Elbow', 'bone_20':'L_Wrist', 'bone_21':'R_Wrist', 'bone_22':'L_Hand', 'bone_23':'R_Hand'}

# local
camera_height_above_ground = 0.9

part2num = {part:(ipart+1) for ipart,part in enumerate(sorted_parts)}

def mkdir_safe(directory):
    try:
        os.makedirs(directory)
    except FileExistsError:
        pass

def setState0():
    for ob in bpy.data.objects.values():
        ob.select=False
    bpy.context.scene.objects.active = None

# creation of the spherical harmonics material, using an OSL script
def create_sh_material(tree, sh_path, img=None):
    # clear default nodes
    for n in tree.nodes:
        tree.nodes.remove(n)

    uv = tree.nodes.new('ShaderNodeTexCoord')
    uv.location = -800, 400

    uv_xform = tree.nodes.new('ShaderNodeVectorMath')
    uv_xform.location = -600, 400
    uv_xform.inputs[1].default_value = (0, 0, 1)
    uv_xform.operation = 'AVERAGE'

    uv_im = tree.nodes.new('ShaderNodeTexImage')
    uv_im.location = -400, 400
    if img is not None:
        uv_im.image = img

    rgb = tree.nodes.new('ShaderNodeRGB')
    rgb.location = -400, 200

    script = tree.nodes.new('ShaderNodeScript')
    script.location = -230, 400
    script.mode = 'EXTERNAL'
    script.filepath = sh_path  # 'spher_harm/sh.osl' #using the same file from multiple jobs causes white texture
    script.update()

    # the emission node makes it independent of the scene lighting
    emission = tree.nodes.new('ShaderNodeEmission')    ### TODO: may be delete this?
    emission.location = -60, 400

    mat_out = tree.nodes.new('ShaderNodeOutputMaterial')
    mat_out.location = 110, 400

    tree.links.new(uv.outputs[2], uv_im.inputs[0])
    tree.links.new(uv_im.outputs[0], script.inputs[0])
    tree.links.new(script.outputs[0], emission.inputs[0])
    tree.links.new(emission.outputs[0], mat_out.inputs[0])


# create the different passes that we render
def create_composite_nodes(tree, params, img=None, idx=0):
    res_paths = {k: join(params['tmp_path'], '%05d_%s' % (idx, k)) for k in params['output_types'] if
                 params['output_types'][k]}

    # clear default nodes
    for n in tree.nodes:
        tree.nodes.remove(n)

    # create node for foreground image
    layers = tree.nodes.new('CompositorNodeRLayers')
    layers.location = -300, 400

    # create node for background image
    bg_im = tree.nodes.new('CompositorNodeImage')
    bg_im.location = -300, 30
    if img is not None:
        bg_im.image = img

    if (params['output_types']['vblur']):
        # create node for computing vector blur (approximate motion blur)
        vblur = tree.nodes.new('CompositorNodeVecBlur')
        vblur.factor = params['vblur_factor']
        vblur.location = 240, 400

        # create node for saving output of vector blurred image
        vblur_out = tree.nodes.new('CompositorNodeOutputFile')
        vblur_out.format.file_format = 'PNG'
        vblur_out.base_path = res_paths['vblur']
        vblur_out.location = 460, 460

    # create node for mixing foreground and background images
    mix = tree.nodes.new('CompositorNodeMixRGB')
    mix.location = 40, 30
    mix.use_alpha = True

    # create node for the final output
    composite_out = tree.nodes.new('CompositorNodeComposite')
    composite_out.location = 240, 30

    # create node for saving depth
    if (params['output_types']['depth']):
        depth_out = tree.nodes.new('CompositorNodeOutputFile')
        depth_out.location = 40, 700
        depth_out.format.file_format = 'OPEN_EXR'
        depth_out.base_path = res_paths['depth']

    # create node for saving normals
    if (params['output_types']['normal']):
        normal_out = tree.nodes.new('CompositorNodeOutputFile')
        normal_out.location = 40, 600
        normal_out.format.file_format = 'OPEN_EXR'
        normal_out.base_path = res_paths['normal']

    # create node for saving foreground image
    if (params['output_types']['fg']):
        fg_out = tree.nodes.new('CompositorNodeOutputFile')
        fg_out.location = 170, 600
        fg_out.format.file_format = 'PNG'
        fg_out.base_path = res_paths['fg']

    # create node for saving ground truth flow
    if (params['output_types']['gtflow']):
        gtflow_out = tree.nodes.new('CompositorNodeOutputFile')
        gtflow_out.location = 40, 500
        gtflow_out.format.file_format = 'OPEN_EXR'
        gtflow_out.base_path = res_paths['gtflow']

    # create node for saving segmentation
    if (params['output_types']['segm']):
        segm_out = tree.nodes.new('CompositorNodeOutputFile')
        segm_out.location = 40, 400
        segm_out.format.file_format = 'OPEN_EXR'
        segm_out.base_path = res_paths['segm']

    # merge fg and bg images
    tree.links.new(bg_im.outputs[0], mix.inputs[1])
    tree.links.new(layers.outputs['Image'], mix.inputs[2])

    if (params['output_types']['vblur']):
        tree.links.new(mix.outputs[0], vblur.inputs[0])  # apply vector blur on the bg+fg image,
        tree.links.new(layers.outputs['Z'], vblur.inputs[1])  # using depth,
        tree.links.new(layers.outputs['Speed'], vblur.inputs[2])  # and flow.
        tree.links.new(vblur.outputs[0], vblur_out.inputs[0])  # save vblurred output

    tree.links.new(mix.outputs[0], composite_out.inputs[0])  # bg+fg image
    if (params['output_types']['fg']):
        tree.links.new(layers.outputs['Image'], fg_out.inputs[0])  # save fg
    # if(params['output_types']['depth']):
    #    tree.links.new(layers.outputs['Z'], depth_out.inputs[0])       # save depth
    if (params['output_types']['normal']):
        tree.links.new(layers.outputs['Normal'], normal_out.inputs[0])  # save normal
    # if(params['output_types']['gtflow']):
    #    tree.links.new(layers.outputs['Speed'], gtflow_out.inputs[0])  # save ground truth flow
    if (params['output_types']['segm']):
        tree.links.new(layers.outputs['IndexMA'], segm_out.inputs[0])  # save segmentation

    return (res_paths)

def setupCameras(scene, params):

    # camera_params = {'camera_dist_to_centre': camera_dist_to_centre,
    #                  'camera_height': camera_height,
    #                  'camera_yaw': camera_yaw,
    #                  'camera_pitch': camera_pitch,
    #                  'camera_roll_from_centerpoint': camera_roll_from_centerpoint,
    #                  'camera_offset': camera_offset,
    #                  'camera_offset_from_circle': camera_offset_from_circle
    #                  }

    # set camera properties and initial position
    bpy.ops.object.select_all(action='DESELECT')
    cam_ob = bpy.data.objects['Camera']
    scn = bpy.context.scene
    scn.objects.active = cam_ob

    radius = params['camera_dist_to_centre']
    camera_yaw = params['camera_yaw']
    camera_height = params['camera_height']
    camera_offset = params['camera_offset']

    print (str(camera_height))

    cam_ob.matrix_world = Matrix(((0., 0., 1, params['camera_dist_to_centre']),
                                  (0., -1, 0., -1.0),
                                  (-1., 0., 0., 0.),
                                  (0.0, 0.0, 0.0, 1.0)))

    gt_cam_location = Vector((radius * np.cos(camera_yaw),
                                        -camera_height,
                                        radius * np.sin(camera_yaw)))

    cam_ob.location = Vector((gt_cam_location.x,
                             gt_cam_location.y,
                             gt_cam_location.z + 1.0))  # move up along z-axis

    camera_roll = -(np.arctan((camera_height - camera_height_above_ground) / radius))
    camera_roll = camera_roll + params['camera_roll_from_centerpoint']

    camera_yaw_adjusted = np.arctan(cam_ob.location.x /(cam_ob.location.z - 1.0))

    if camera_yaw < np.pi:
        camera_yaw_adjusted + np.pi

    cam_ob.rotation_euler = Euler((camera_roll,
                                    camera_yaw_adjusted,
                                    0.0), 'XYZ')

    cam_ob.data.angle = math.radians(40)
    cam_ob.data.lens = 35   # was 60
    cam_ob.data.clip_start = 0.1
    cam_ob.data.sensor_width = 35 # 32

    ## 2nd camera
    tangent = Vector((0.0, 1.0, 0.0)).cross(gt_cam_location)

    tangent = tangent.normalized()

    print(tangent)

    cam1_location = gt_cam_location + tangent * camera_offset
    cam3_location = gt_cam_location - tangent * camera_offset

    bpy.ops.object.camera_add(view_align=False,
                              location= [0.0, 0.0, 0.0],
                              rotation = [0.0, 0.0, 0.0])
    cam_ob1 = bpy.context.object
    cam_ob1.name = 'Camera1'
    cam_ob1.matrix_world = Matrix(((0., 0., 1, params['camera_dist_to_centre']),
                                  (0., -1, 0., -1.0),
                                  (-1., 0., 0., 0.),
                                  (0.0, 0.0, 0.0, 1.0)))
    cam_ob1.location = Vector((cam1_location.x, cam1_location.y, cam1_location.z + 1.0))
    cam_ob1.rotation_euler = Euler((camera_roll, camera_yaw_adjusted, 0.0), 'XYZ')
    cam_ob1.data.angle = math.radians(40)
    cam_ob1.data.lens = 35  # was 60
    cam_ob1.data.clip_start = 0.1
    cam_ob1.data.sensor_width = 35  # 32

    ## 3rd camera
    bpy.ops.object.camera_add(view_align=False,
                              location=[0.0, 0.0, 0.0],
                              rotation=[0.0, 0.0, 0.0])
    cam_ob3 = bpy.context.object
    cam_ob3.name = 'Camera3'
    cam_ob3.matrix_world = Matrix(((0., 0., 1, params['camera_dist_to_centre']),
                                   (0., -1, 0., -1.0),
                                   (-1., 0., 0., 0.),
                                   (0.0, 0.0, 0.0, 1.0)))
    cam_ob3.location = Vector((cam3_location.x, cam3_location.y, cam3_location.z + 1.0))
    cam_ob3.rotation_euler = Euler((camera_roll, camera_yaw_adjusted, 0.0), 'XYZ')
    cam_ob3.data.angle = math.radians(40)
    cam_ob3.data.lens = 35  # was 60
    cam_ob3.data.clip_start = 0.1
    cam_ob3.data.sensor_width = 35  # 32

    return  cam_ob, cam_ob1, cam_ob3

# create one material per part as defined in a pickle with the segmentation
# this is useful to render the segmentation in a material pass
def create_segmentation(ob, params):
    materials = {}
    vgroups = {}
    with open('pkl/segm_per_v_overlap.pkl', 'rb') as f:
        vsegm = load(f)
    bpy.ops.object.material_slot_remove()
    parts = sorted(vsegm.keys())
    for part in parts:
        vs = vsegm[part]
        vgroups[part] = ob.vertex_groups.new(part)
        vgroups[part].add(vs, 1.0, 'ADD')
        bpy.ops.object.vertex_group_set_active(group=part)
        materials[part] = bpy.data.materials['Material'].copy()
        materials[part].pass_index = part2num[part]
        bpy.ops.object.material_slot_add()
        ob.material_slots[-1].material = materials[part]
        bpy.ops.object.mode_set(mode='EDIT')
        bpy.ops.mesh.select_all(action='DESELECT')
        bpy.ops.object.vertex_group_select()
        bpy.ops.object.material_slot_assign()
        bpy.ops.object.mode_set(mode='OBJECT')
    return(materials)

# load poses and shapes
def load_body_data(smpl_data, ob, obname, gender='female', idx=0):
    # load MoSHed data from CMU Mocap (only the given idx is loaded)

    # create a dictionary with key the sequence name and values the pose and trans
    cmu_keys = []
    for seq in smpl_data.files:
        if seq.startswith('pose_'):
            cmu_keys.append(seq.replace('pose_', ''))

    name = sorted(cmu_keys)[idx % len(cmu_keys)]

    cmu_parms = {}
    for seq in smpl_data.files:
        if seq == ('pose_' + name):
            cmu_parms[seq.replace('pose_', '')] = {'poses': smpl_data[seq],
                                                   'trans': smpl_data[seq.replace('pose_', 'trans_')]}

    # compute the number of shape blendshapes in the model
    n_sh_bshapes = len([k for k in ob.data.shape_keys.key_blocks.keys()
                        if k.startswith('Shape')])

    # load all SMPL shapes
    fshapes = smpl_data['%sshapes' % gender][:, :n_sh_bshapes]

    return (cmu_parms, fshapes, name)

# computes rotation matrix through Rodrigues formula as in cv2.Rodrigues
def Rodrigues(rotvec):
    theta = np.linalg.norm(rotvec)
    r = (rotvec/theta).reshape(3, 1) if theta > 0. else rotvec
    cost = np.cos(theta)
    mat = np.asarray([[0, -r[2], r[1]],
                      [r[2], 0, -r[0]],
                      [-r[1], r[0], 0]])

    return(cost*np.eye(3) + (1-cost)*r.dot(r.T) + np.sin(theta)*mat)

# transformation between pose and blendshapes
def rodrigues2bshapes(pose):
    rod_rots = np.asarray(pose).reshape(24, 3)
    mat_rots = [Rodrigues(rod_rot) for rod_rot in rod_rots]
    bshapes = np.concatenate([(mat_rot - np.eye(3)).ravel()
                              for mat_rot in mat_rots[1:]])
    return(mat_rots, bshapes)

# apply trans pose and shape to character
def apply_trans_pose_shape(trans, pose, shape, ob, arm_ob, obname, scene, cam_ob, frame=None):
    # transform pose into rotation matrices (for pose) and pose blendshapes
    mrots, bsh = rodrigues2bshapes(pose)

    # set the location of the first bone to the translation parameter
    arm_ob.pose.bones[obname+'_Pelvis'].location = trans
    if frame is not None:
        arm_ob.pose.bones[obname+'_root'].keyframe_insert('location', frame=frame)
    # set the pose of each bone to the quaternion specified by pose
    for ibone, mrot in enumerate(mrots):
        bone = arm_ob.pose.bones[obname+'_'+part_match['bone_%02d' % ibone]]
        bone.rotation_quaternion = Matrix(mrot).to_quaternion()
        if frame is not None:
            bone.keyframe_insert('rotation_quaternion', frame=frame)
            bone.keyframe_insert('location', frame=frame)

    # apply pose blendshapes
    for ibshape, bshape in enumerate(bsh):
        ob.data.shape_keys.key_blocks['Pose%03d' % ibshape].value = bshape
        if frame is not None:
            ob.data.shape_keys.key_blocks['Pose%03d' % ibshape].keyframe_insert('value', index=-1, frame=frame)

    # apply shape blendshapes
    for ibshape, shape_elem in enumerate(shape):
        ob.data.shape_keys.key_blocks['Shape%03d' % ibshape].value = shape_elem
        if frame is not None:
            ob.data.shape_keys.key_blocks['Shape%03d' % ibshape].keyframe_insert('value', index=-1, frame=frame)

# reset the joint positions of the character according to its new shape
def reset_joint_positions(orig_trans, shape, ob, arm_ob, obname, scene, cam_ob, reg_ivs, joint_reg):
    # since the regression is sparse, only the relevant vertex
    #     elements (joint_reg) and their indices (reg_ivs) are loaded
    reg_vs = np.empty((len(reg_ivs), 3))  # empty array to hold vertices to regress from
    # zero the pose and trans to obtain joint positions in zero pose
    apply_trans_pose_shape(orig_trans, np.zeros(72), shape, ob, arm_ob, obname, scene, cam_ob)

    # obtain a mesh after applying modifiers
    bpy.ops.wm.memory_statistics()
    # me holds the vertices after applying the shape blendshapes
    me = ob.to_mesh(scene, True, 'PREVIEW')

    # fill the regressor vertices matrix
    for iiv, iv in enumerate(reg_ivs):
        reg_vs[iiv] = me.vertices[iv].co
    bpy.data.meshes.remove(me)

    # regress joint positions in rest pose
    joint_xyz = joint_reg.dot(reg_vs)
    # adapt joint positions in rest pose
    arm_ob.hide = False
    bpy.ops.object.mode_set(mode='EDIT')
    arm_ob.hide = True
    for ibone in range(24):
        bb = arm_ob.data.edit_bones[obname+'_'+part_match['bone_%02d' % ibone]]
        bboffset = bb.tail - bb.head
        bb.head = joint_xyz[ibone]
        bb.tail = bb.head + bboffset
    bpy.ops.object.mode_set(mode='OBJECT')
    return(shape)

def get_bone_locs(obname, arm_ob, scene, cam_ob):
    n_bones = 24
    render_scale = scene.render.resolution_percentage / 100
    render_size = (int(scene.render.resolution_x * render_scale),
                   int(scene.render.resolution_y * render_scale))
    bone_locations_2d = np.empty((n_bones, 2))
    bone_locations_3d = np.empty((n_bones, 3), dtype='float32')

    # obtain the coordinates of each bone head in image space
    for ibone in range(n_bones):
        bone = arm_ob.pose.bones[obname+'_'+part_match['bone_%02d' % ibone]]
        co_2d = world2cam(scene, cam_ob, arm_ob.matrix_world * bone.head)
        co_3d = arm_ob.matrix_world * bone.head
        bone_locations_3d[ibone] = (co_3d.x,
                                 co_3d.y,
                                 co_3d.z)
        bone_locations_2d[ibone] = (round(co_2d.x * render_size[0]),
                                 round(co_2d.y * render_size[1]))
    return(bone_locations_2d, bone_locations_3d)

import time
start_time = None
def log_message(message):
    elapsed_time = time.time() - start_time
    print("[%.2f s] %s" % (elapsed_time, message))

def main():
    print("IN MAIN")
    global start_time
    start_time = time.time()

    # parse commandline arguments
    import argparse
    log_message(sys.argv)
    parser = argparse.ArgumentParser(description='Generate synth dataset images.')
    parser.add_argument('--idx', type=int,
                        help='idx of the requested sequence')
    args = parser.parse_args(sys.argv[sys.argv.index("--") + 1:])
    idx = args.idx

    log_message("Loading models file..")
    idx_info = load(open("pkl/idx_info.pickle", 'rb'))

    length_idx_info = len(idx_info)
    log_message("idx_info length: %d " % length_idx_info)

    bg_path = '/home/lera/Documents/surreal-master/datageneration/misc/background'
    output_path = '/home/lera/Documents/CGModel/Surreal/output_mv'
    output_path1 = join(output_path, 'cam%d' % 1)
    output_path2 = join(output_path, 'cam%d' % 2)
    output_path3 = join(output_path, 'cam%d' % 3)

    # create output directory
    if not exists(output_path1):
        mkdir_safe(output_path1)
    if not exists(output_path2):
        mkdir_safe(output_path2)
    if not exists(output_path3):
        mkdir_safe(output_path3)

    tmp_path = '/home/lera/Documents/CGModel/Surreal/tmp_mv'

    if not exists(output_path1):
        mkdir_safe(output_path1)

    if not exists(output_path2):
        mkdir_safe(output_path2)

    if not exists(output_path3):
        mkdir_safe(output_path3)

    # config
    resy = 320 #1280
    resx = 280 #720
    stepsize = 20
    stride = 50
    clipsize = 10

    log_message("Deleting cube")  ## TODO: may need to refactor if this holds material
    # delete the default cube (which held the material)
    bpy.ops.object.select_all(action='DESELECT')
    bpy.data.objects['Cube'].select = True
    bpy.ops.object.delete(use_global=False)

    log_message("=========================")
    log_message("Generating random index..")
    idx = randint(0, length_idx_info - 1)
    #idx = 32

    log_message("idx = %d" % idx)

    log_message("Setting selected data")
    selected_idx_info = idx_info[idx]

    name = selected_idx_info['name']
    log_message("name: %s" % name)
    log_message("nb_frames: %f" % selected_idx_info['nb_frames'])

    use_split = selected_idx_info['use_split']
    #log_message("test or train: %s" % use_split)

    # initialize RNG with seeds from sequence id
    import hashlib
    s = "synth_data:%d:%d:%d" % (idx, 0, 0)
    seed_number = int(hashlib.sha1(s.encode('utf-8')).hexdigest(), 16) % (10 ** 8)
    log_message("GENERATED SEED %d from string '%s'" % (seed_number, s))
    random.seed(seed_number)
    np.random.seed(seed_number)

    #output_path_f1 = join(output_path, 'cam%d' % 1)
    #output_path_f2 = join(output_path, 'cam%d' % 2)
    #output_path_f3 = join(output_path, 'cam%d' % 3)

    log_message("Creating temp directory")
    tmp_path_f = join(tmp_path, 'run%d_%s_c%04d' % (0, idx, (0 + 1)))
    if exists(tmp_path_f) and tmp_path_f != "" and tmp_path_f != "/":
        os.system('rm -rf %s' % tmp_path_f)
    if not exists(tmp_path_f):
        mkdir_safe(tmp_path_f)

    log_message("Copying spherical harmonics directory")
    sh_dir = join(tmp_path_f, 'spher_harm')
    if not exists(sh_dir):
        mkdir_safe(sh_dir)
    sh_dst = join(sh_dir, 'sh_%02d_%05d.osl' % (0, idx))
    os.system('cp spher_harm/sh.osl %s' % sh_dst)

    log_message("Generating human")
    genders = {0: 'female', 1: 'male'}
    # pick random gender
    gender = choice(genders)
    log_message("gender: %s" % gender)

    scene = bpy.data.scenes['Scene']
    scene.render.engine = 'CYCLES'
    bpy.data.materials['Material'].use_nodes = True
    scene.cycles.shading_system = True
    scene.use_nodes = True

    log_message("Creating background")
    bg_img_name = join(bg_path, 'UseAsBk.jpg')
    bg_img = bpy.data.images.load(bg_img_name)

    log_message("Creating clothes")
    clothing_option = 'all'
    log_message("clothing: %s" % clothing_option)

    smpl_data_folder = 'smpl_data'
    with open(join(smpl_data_folder, 'textures', '%s_%s.txt' % (gender, use_split))) as f:
        txt_paths = f.read().splitlines()

    # if using only one source of clothing
    if clothing_option == 'nongrey':
        txt_paths = [k for k in txt_paths if 'nongrey' in k]
    elif clothing_option == 'grey':
        txt_paths = [k for k in txt_paths if 'nongrey' not in k]

    # random clothing texture
    cloth_img_name = choice(txt_paths)
    log_message("Selected clothes image file: %s" % cloth_img_name)
    cloth_img_name = join(smpl_data_folder, cloth_img_name)
    cloth_img = bpy.data.images.load(cloth_img_name)

    log_message("Loading parts segmentation")
    beta_stds = np.load(join(smpl_data_folder, ('%s_beta_stds.npy' % gender)))

    log_message("Building materials tree")
    materials_tree = bpy.data.materials['Material'].node_tree
    create_sh_material(materials_tree, sh_dst, cloth_img)

    log_message("Creating nodes")
    output_types = {'depth': False, 'fg': False, 'gtflow': False, 'normal': False, 'segm': False, 'vblur': False}
    params = {'resx':resx, 'resy': resy, 'tmp_path': tmp_path_f, 'output_types':output_types}

    res_paths = create_composite_nodes(scene.node_tree, params, img=bg_img, idx=idx)

    log_message("Loading smpl data")
    smpl_data = np.load(join(smpl_data_folder, 'smpl_data.npz'))

    log_message("Initializing scene")
    log_message("Loading fbx")
    bpy.ops.import_scene.fbx(
        filepath=join('smpl_data', 'basicModel_%s_lbs_10_207_0_v1.0.2.fbx' % gender[0]),
        axis_forward='Y', axis_up='Z', global_scale=100)
    obname = '%s_avg' % gender[0]
    ob = bpy.data.objects[obname]
    ob.data.use_auto_smooth = False  # autosmooth creates artifacts

    # assign the existing spherical harmonics material
    ob.active_material = bpy.data.materials['Material']

    log_message("Initializing cameras")
    camera_dist_to_centre = 2.5 + 2.5 * np.random.rand()
    camera_height = camera_height_above_ground + np.random.rand()
    camera_yaw  = 2 * np.pi * np.random.rand()   #random_zrot = randint(0, 360)
    camera_pitch = -0.092 * np.pi + 0.184 * np.pi * np.random.rand() # 33 degrees
    camera_roll_from_centerpoint = -0.0277 * np.pi + 0.0555 * np.pi * np.random.rand() # 10 degrees
    camera_offset = uniform(0.05, 1.5)
    camera_offset_from_circle = randint(0,100)  # 100% - 1st and 3rd cameras on a tangent line through 2nd camera, 0% - 1st and 3rd camera on the circle with same radius as 2nd camera

    camera_params = {'camera_dist_to_centre': camera_dist_to_centre,
                     'camera_height': camera_height,
                     'camera_yaw': camera_yaw,
                     'camera_pitch': camera_pitch,
                     'camera_roll_from_centerpoint': camera_roll_from_centerpoint,
                     'camera_offset':camera_offset,
                     'camera_offset_from_circle':camera_offset_from_circle
                     }

    print(camera_params)

    (camera, cam_ob1, cam_ob3) = setupCameras(scene, camera_params)

    # setup an empty object in the center which will be the parent of the Camera
    # this allows to easily rotate an object around the origin
    scn = bpy.context.scene
    scn.cycles.film_transparent = True
    scn.render.layers["RenderLayer"].use_pass_vector = True
    scn.render.layers["RenderLayer"].use_pass_normal = True
    scene.render.layers['RenderLayer'].use_pass_emit = True
    scene.render.layers['RenderLayer'].use_pass_emit = True
    scene.render.layers['RenderLayer'].use_pass_material_index = True

    # set render size
    scn.render.resolution_x = params['resy']
    scn.render.resolution_y = params['resx']
    scn.render.resolution_percentage = 100
    scn.render.image_settings.file_format = 'PNG'

    # clear existing animation data
    ob.data.shape_keys.animation_data_clear()
    arm_ob = bpy.data.objects['Armature']
    arm_ob.animation_data_clear()

    log_message("After scene initialised")
    setState0()
    ob.select = True
    bpy.context.scene.objects.active = ob
    segmented_materials = True  # True: 0-24, False: expected to have 0-1 bg/fg

    log_message("Creating materials segmentation")
    # create material segmentation
    if segmented_materials:
        materials = create_segmentation(ob, params)
        prob_dressed = {'leftLeg': .5, 'leftArm': .9, 'leftHandIndex1': .01,
                        'rightShoulder': .8, 'rightHand': .01, 'neck': .01,
                        'rightToeBase': .9, 'leftShoulder': .8, 'leftToeBase': .9,
                        'rightForeArm': .5, 'leftHand': .01, 'spine': .9,
                        'leftFoot': .9, 'leftUpLeg': .9, 'rightUpLeg': .9,
                        'rightFoot': .9, 'head': .01, 'leftForeArm': .5,
                        'rightArm': .5, 'spine1': .9, 'hips': .9,
                        'rightHandIndex1': .01, 'spine2': .9, 'rightLeg': .5}
    else:
        materials = {'FullBody': bpy.data.materials['Material']}
        prob_dressed = {'FullBody': .6}

    orig_pelvis_loc = (arm_ob.matrix_world.copy() * arm_ob.pose.bones[obname + '_Pelvis'].head.copy()) - Vector(
            (-1., 1., 1.))
    orig_cam_loc = camera.location.copy()

    # unblocking both the pose and the blendshape limits
    for k in ob.data.shape_keys.key_blocks.keys():
            bpy.data.shape_keys["Key"].key_blocks[k].slider_min = -10
            bpy.data.shape_keys["Key"].key_blocks[k].slider_max = 10

    log_message("Loading body data")
    cmu_parms, fshapes, name = load_body_data(smpl_data, ob, obname, idx=idx, gender=gender)

    log_message("Loaded body data for %s" % name)

    nb_fshapes = len(fshapes)

    ## TODO: train vs test?
    #if idx_info['use_split'] == 'train':
    #    fshapes = fshapes[:int(nb_fshapes * 0.8)]
    #elif idx_info['use_split'] == 'test':
    #    fshapes = fshapes[int(nb_fshapes * 0.8):]

    # pick random real body shape
    log_message("Selecting random body shape")
    shape = choice(fshapes)  # +random_shape(.5) can add noise

    ndofs = 10

    scene.objects.active = arm_ob
    orig_trans = np.asarray(arm_ob.pose.bones[obname + '_Pelvis'].location).copy()  ## TODO: this line fails

    # spherical harmonics material needs a script to be loaded and compiled
    scs = []
    for mname, material in materials.items():
        scs.append(material.node_tree.nodes['Script'])
        scs[-1].filepath = sh_dst
        scs[-1].update()

    rgb_dirname = name.replace(" ", "") + '_c%04d.mp4' % (0 + 1)
    rgb_path = join(tmp_path_f, rgb_dirname)

    data = cmu_parms[name]

    fbegin = 0 * stepsize * stride
    fend = min(0 * stepsize * stride + stepsize * clipsize, len(data['poses']))

    log_message("Computing how many frames to allocate")
    N = len(data['poses'][fbegin:fend:stepsize])
    log_message("Allocating %d frames in mat file" % N)

    # force recomputation of joint angles unless shape is all zeros
    curr_shape = np.zeros_like(shape)
    nframes = len(data['poses'][::stepsize])

    matfile_info = join(output_path, name.replace(" ", "") + "_c%04d_info.mat" % (0 + 1))
    log_message('Working on %s' % matfile_info)

    # allocate
    dict_info = {}
    dict_info['bg'] = np.zeros((N,), dtype=np.object)  # background image path
    dict_info['bg1'] = np.zeros((N,), dtype=np.object)  # background image path
    dict_info['bg3'] = np.zeros((N,), dtype=np.object)  # background image path
    dict_info['camLoc'] = np.empty(3)  # (1, 3)
    dict_info['camLoc1'] = np.empty(3)  # (1, 3)
    dict_info['camLoc3'] = np.empty(3)  # (1, 3)
    dict_info['clipNo'] = 0 + 1
    dict_info['cloth'] = np.zeros((N,), dtype=np.object)  # clothing texture image path
    dict_info['cloth1'] = np.zeros((N,), dtype=np.object)  # clothing texture image path
    dict_info['cloth3'] = np.zeros((N,), dtype=np.object)  # clothing texture image path
    dict_info['gender'] = np.empty(N, dtype='uint8')  # 0 for male, 1 for female
    dict_info['joints2D'] = np.empty((2, 24, N), dtype='float32')  # 2D joint positions in pixel space
    dict_info['joints3D'] = np.empty((3, 24, N), dtype='float32')  # 3D joint positions in world coordinates
    dict_info['light'] = np.empty((9, N), dtype='float32')
    dict_info['light1'] = np.empty((9, N), dtype='float32')
    dict_info['light3'] = np.empty((9, N), dtype='float32')
    dict_info['pose'] = np.empty((data['poses'][0].size, N), dtype='float32')  # joint angles from SMPL (CMU)
    dict_info['sequence'] = name.replace(" ", "") + "_c%04d" % (0 + 1)
    dict_info['shape'] = np.empty((ndofs, N), dtype='float32')
    dict_info['zrot'] = np.empty(N, dtype='float32')
    dict_info['camDist'] = camera_dist_to_centre
    dict_info['stride'] = stride

    if name.replace(" ", "").startswith('h36m'):
        dict_info['source'] = 'h36m'
    else:
        dict_info['source'] = 'cmu'

    if (output_types['vblur']):
        dict_info['vblur_factor'] = np.empty(N, dtype='float32')

    # for each clipsize'th frame in the sequence
    get_real_frame = lambda ifr: ifr
    reset_loc = False
    batch_it = 0
    curr_shape = reset_joint_positions(orig_trans, shape, ob, arm_ob, obname, scene,
                                       camera, smpl_data['regression_verts'], smpl_data['joint_regressor'])

    arm_ob.animation_data_clear()
    camera.animation_data_clear()
    cam_ob1.animation_data_clear()
    cam_ob3.animation_data_clear()

    #bpy.ops.wm.save_as_mainfile(filepath=join(tmp_path_f, 'pre.blend'))

    # create a keyframe animation with pose, translation, blendshapes and camera motion
    # LOOP TO CREATE 3D ANIMATION
    for seq_frame, (pose, trans) in enumerate(
            zip(data['poses'][fbegin:fend:stepsize], data['trans'][fbegin:fend:stepsize])):
        iframe = seq_frame
        scene.frame_set(get_real_frame(seq_frame))

        # apply the translation, pose and shape to the character
        apply_trans_pose_shape(Vector(trans), pose, shape, ob, arm_ob, obname, scene, camera,
                               get_real_frame(seq_frame))
        dict_info['shape'][:, iframe] = shape[:ndofs]
        dict_info['pose'][:, iframe] = pose
        dict_info['gender'][iframe] = list(genders)[list(genders.values()).index(gender)]
        #if (output_types['vblur']):
        #    dict_info['vblur_factor'][iframe] = vblur_factor

        arm_ob.pose.bones[obname + '_root'].rotation_quaternion = Quaternion(Euler((0, 0, 0), 'XYZ'))  ## TODO: camera yaw
        arm_ob.pose.bones[obname + '_root'].keyframe_insert('rotation_quaternion', frame=get_real_frame(seq_frame))
        dict_info['zrot'][iframe] = 0   ## TODO: rotation

        scene.update()

        # Bodies centered only in each minibatch of clipsize frames
        if seq_frame == 0 or reset_loc:
            reset_loc = False
            new_pelvis_loc = arm_ob.matrix_world.copy() * arm_ob.pose.bones[obname + '_Pelvis'].head.copy()
            print (new_pelvis_loc.copy() - orig_pelvis_loc.copy())

            #camera.location = orig_cam_loc.copy() + (new_pelvis_loc.copy() - orig_pelvis_loc.copy())

            camera.keyframe_insert('location', frame=get_real_frame(seq_frame))
            cam_ob1.keyframe_insert('location', frame=get_real_frame(seq_frame))
            cam_ob3.keyframe_insert('location', frame=get_real_frame(seq_frame))
            dict_info['camLoc'] = np.array(camera.location)
            dict_info['camLoc1'] = np.array(cam_ob1.location)
            dict_info['camLoc3'] = np.array(cam_ob3.location)

    scene.node_tree.nodes['Image'].image = bg_img

    for part, material in materials.items():
        material.node_tree.nodes['Vector Math'].inputs[1].default_value[:2] = (0, 0)

    # random light
    sh_coeffs = .7 * (2 * np.random.rand(9) - 1)
    sh_coeffs[
        0] = .5 + .9 * np.random.rand()  # Ambient light (first coeff) needs a minimum  is ambient. Rest is uniformly distributed, higher means brighter.
    sh_coeffs[1] = -.7 * np.random.rand()

    for ish, coeff in enumerate(sh_coeffs):
        for sc in scs:
            sc.inputs[ish + 1].default_value = coeff

    # iterate over the keyframes and render
    # LOOP TO RENDER
    for seq_frame, (pose, trans) in enumerate(
            zip(data['poses'][fbegin:fend:stepsize], data['trans'][fbegin:fend:stepsize])):
        scene.frame_set(get_real_frame(seq_frame))
        iframe = seq_frame

        dict_info['bg'][iframe] = bg_img_name
        dict_info['cloth'][iframe] = cloth_img_name
        dict_info['light'][:, iframe] = sh_coeffs

        scene.render.use_antialiasing = False
        scene.render.filepath = join(rgb_path, 'Image%04d.png' % get_real_frame(seq_frame))

        log_message("Rendering frame %d" % seq_frame)

        # disable render output
        logfile = '/dev/null'
        open(logfile, 'a').close()
        old = os.dup(1)
        sys.stdout.flush()
        os.close(1)
        os.open(logfile, os.O_WRONLY)

        # Render
        bpy.ops.render.render(write_still=True)

        # disable output redirection
        os.close(1)
        os.dup(old)
        os.close(old)

        # NOTE:
        # ideally, pixels should be readable from a viewer node, but I get only zeros
        # --> https://ammous88.wordpress.com/2015/01/16/blender-access-render-results-pixels-directly-from-python-2/
        # len(np.asarray(bpy.data.images['Render Result'].pixels) is 0
        # Therefore we write them to temporary files and read with OpenEXR library (available for python2) in main_part2.py
        # Alternatively, if you don't want to use OpenEXR library, the following commented code does loading with Blender functions, but it can cause memory leak.
        # If you want to use it, copy necessary lines from main_part2.py such as definitions of dict_normal, matfile_normal...

        # for k, folder in res_paths.items():
        #   if not k== 'vblur' and not k=='fg':
        #       path = join(folder, 'Image%04d.exr' % get_real_frame(seq_frame))
        #       render_img = bpy.data.images.load(path)
        #       # render_img.pixels size is width * height * 4 (rgba)
        #       arr = np.array(render_img.pixels[:]).reshape(resx, resy, 4)[::-1,:, :] # images are vertically flipped
        #       if k == 'normal':# 3 channels, original order
        #           mat = arr[:,:, :3]
        #           dict_normal['normal_%d' % (iframe + 1)] = mat.astype(np.float32, copy=False)
        #       elif k == 'gtflow':
        #           mat = arr[:,:, 1:3]
        #           dict_gtflow['gtflow_%d' % (iframe + 1)] = mat.astype(np.float32, copy=False)
        #       elif k == 'depth':
        #           mat = arr[:,:, 0]
        #           dict_depth['depth_%d' % (iframe + 1)] = mat.astype(np.float32, copy=False)
        #       elif k == 'segm':
        #           mat = arr[:,:,0]
        #           dict_segm['segm_%d' % (iframe + 1)] = mat.astype(np.uint8, copy=False)
        #
        #       # remove the image to release memory, object handles, etc.
        #       render_img.user_clear()
        #       bpy.data.images.remove(render_img)

        # bone locations should be saved after rendering so that the bones are updated
        bone_locs_2D, bone_locs_3D = get_bone_locs(obname, arm_ob, scene, camera)
        dict_info['joints2D'][:, :, iframe] = np.transpose(bone_locs_2D)
        dict_info['joints3D'][:, :, iframe] = np.transpose(bone_locs_3D)

        reset_loc = (bone_locs_2D.max(axis=-1) > 256).any() or (bone_locs_2D.min(axis=0) < 0).any()
        arm_ob.pose.bones[obname + '_root'].rotation_quaternion = Quaternion((1, 0, 0, 0))

    ## Use camera 1
    bpy.context.scene.camera = cam_ob1

    # iterate over the keyframes and render
    # LOOP TO RENDER
    for seq_frame, (pose, trans) in enumerate(
            zip(data['poses'][fbegin:fend:stepsize], data['trans'][fbegin:fend:stepsize])):
        scene.frame_set(get_real_frame(seq_frame))
        iframe = seq_frame

        dict_info['bg1'][iframe] = bg_img_name
        dict_info['cloth1'][iframe] = cloth_img_name
        dict_info['light1'][:, iframe] = sh_coeffs

        scene.render.use_antialiasing = False
        scene.render.filepath = join(rgb_path, 'Image1%04d.png' % get_real_frame(seq_frame))

        log_message("Rendering frame %d" % seq_frame)

        # disable render output
        logfile = '/dev/null'
        open(logfile, 'a').close()
        old = os.dup(1)
        sys.stdout.flush()
        os.close(1)
        os.open(logfile, os.O_WRONLY)

        # Render
        bpy.ops.render.render(write_still=True)

        # disable output redirection
        os.close(1)
        os.dup(old)
        os.close(old)

    ## Use camera 3
    bpy.context.scene.camera = cam_ob3

    # iterate over the keyframes and render
    # LOOP TO RENDER
    for seq_frame, (pose, trans) in enumerate(
            zip(data['poses'][fbegin:fend:stepsize], data['trans'][fbegin:fend:stepsize])):
        scene.frame_set(get_real_frame(seq_frame))
        iframe = seq_frame

        dict_info['bg3'][iframe] = bg_img_name
        dict_info['cloth3'][iframe] = cloth_img_name
        dict_info['light3'][:, iframe] = sh_coeffs

        scene.render.use_antialiasing = False
        scene.render.filepath = join(rgb_path, 'Image3%04d.png' % get_real_frame(seq_frame))

        log_message("Rendering frame %d" % seq_frame)

        # disable render output
        logfile = '/dev/null'
        open(logfile, 'a').close()
        old = os.dup(1)
        sys.stdout.flush()
        os.close(1)
        os.open(logfile, os.O_WRONLY)

        # Render
        bpy.ops.render.render(write_still=True)

        # disable output redirection
        os.close(1)
        os.dup(old)
        os.close(old)

    bpy.ops.wm.save_as_mainfile(filepath=join(tmp_path_f, 'post.blend'))

    ## TODO: need to delete model before the next iteration


if __name__ == '__main__':
    main()
