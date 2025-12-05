#Modified From the rlbench: https://github.com/stepjam/RLBench
from copy import deepcopy
from os import error
import os
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
from PIL import Image
from pyrep.backend import utils as pyrep_utils
from pyrep.backend._sim_cffi import ffi, lib
from pyrep.const import PYREP_SCRIPT_TYPE, JointType
from pyrep.objects.dummy import Dummy
from pyrep.objects.object import Object
from pyrep.objects.shape import Shape
from pyrep.objects.joint import Joint
from pyrep.pyrep import PyRep
from pyrep.robots.arms.arm import Arm
from pyrep.robots.end_effectors.gripper import Gripper
from amsolver.backend.robot import Robot
from pyrep.objects.cartesian_path import CartesianPath
from amsolver.const import colors
from amsolver.backend.spawn_boundary import BoundaryObject, BoundingBox, SpawnBoundary
from copy import deepcopy
from pyrep.backend import sim
from scipy.spatial.transform import Rotation as R

from tools.grasploc import Grasploc, define_default_args

def ClipFloatValues(float_array, min_value, max_value):
  """Clips values to the range [min_value, max_value].

  First checks if any values are out of range and prints a message.
  Then clips all values to the given range.

  Args:
    float_array: 2D array of floating point values to be clipped.
    min_value: Minimum value of clip range.
    max_value: Maximum value of clip range.

  Returns:
    The clipped array.

  """
  if float_array.min() < min_value or float_array.max() > max_value:
    float_array = np.clip(float_array, min_value, max_value)
  return float_array


DEFAULT_RGB_SCALE_FACTOR = 256000.0


def float_array_to_rgb_image(float_array,
                             scale_factor=DEFAULT_RGB_SCALE_FACTOR,
                             drop_blue=False):
  """Convert a floating point array of values to an RGB image.

  Convert floating point values to a fixed point representation where
  the RGB bytes represent a 24-bit integer.
  R is the high order byte.
  B is the low order byte.
  The precision of the depth image is 1/256 mm.

  Floating point values are scaled so that the integer values cover
  the representable range of depths.

  This image representation should only use lossless compression.

  Args:
    float_array: Input array of floating point depth values in meters.
    scale_factor: Scale value applied to all float values.
    drop_blue: Zero out the blue channel to improve compression, results in 1mm
      precision depth values.

  Returns:
    24-bit RGB PIL Image object representing depth values.
  """
  # Scale the floating point array.
  scaled_array = np.floor(float_array * scale_factor + 0.5)
  # Convert the array to integer type and clip to representable range.
  min_inttype = 0
  max_inttype = 2**24 - 1
  scaled_array = ClipFloatValues(scaled_array, min_inttype, max_inttype)
  int_array = scaled_array.astype(np.uint32)
  # Calculate:
  #   r = (f / 256) / 256  high byte
  #   g = (f / 256) % 256  middle byte
  #   b = f % 256          low byte
  rg = np.divide(int_array, 256)
  r = np.divide(rg, 256)
  g = np.mod(rg, 256)
  image_shape = int_array.shape
  rgb_array = np.zeros((image_shape[0], image_shape[1], 3), dtype=np.uint8)
  rgb_array[..., 0] = r
  rgb_array[..., 1] = g
  if not drop_blue:
    # Calculate the blue channel and add it to the array.
    b = np.mod(int_array, 256)
    rgb_array[..., 2] = b
  image_mode = 'RGB'
  image = Image.fromarray(rgb_array, mode=image_mode)
  return image


DEFAULT_GRAY_SCALE_FACTOR = {np.uint8: 100.0,
                             np.uint16: 1000.0,
                             np.int32: DEFAULT_RGB_SCALE_FACTOR}


def float_array_to_grayscale_image(float_array, scale_factor=None, image_dtype=np.uint8):
  """Convert a floating point array of values to an RGB image.

  Convert floating point values to a fixed point representation with
  the given bit depth.

  The precision of the depth image with default scale_factor is:
    uint8: 1cm, with a range of [0, 2.55m]
    uint16: 1mm, with a range of [0, 65.5m]
    int32: 1/256mm, with a range of [0, 8388m]

  Right now, PIL turns uint16 images into a very strange format and
  does not decode int32 images properly.  Only uint8 works correctly.

  Args:
    float_array: Input array of floating point depth values in meters.
    scale_factor: Scale value applied to all float values.
    image_dtype: Image datatype, which controls the bit depth of the grayscale
      image.

  Returns:
    Grayscale PIL Image object representing depth values.

  """
  # Ensure that we have a valid numeric type for the image.
  if image_dtype == np.uint16:
    image_mode = 'I;16'
  elif image_dtype == np.int32:
    image_mode = 'I'
  else:
    image_dtype = np.uint8
    image_mode = 'L'
  if scale_factor is None:
    scale_factor = DEFAULT_GRAY_SCALE_FACTOR[image_dtype]
  # Scale the floating point array.
  scaled_array = np.floor(float_array * scale_factor + 0.5)
  # Convert the array to integer type and clip to representable range.
  min_dtype = np.iinfo(image_dtype).min
  max_dtype = np.iinfo(image_dtype).max
  scaled_array = ClipFloatValues(scaled_array, min_dtype, max_dtype)

  image_array = scaled_array.astype(image_dtype)
  image = Image.fromarray(image_array, mode=image_mode)
  return image


def image_to_float_array(image, scale_factor=None):
  """Recovers the depth values from an image.

  Reverses the depth to image conversion performed by FloatArrayToRgbImage or
  FloatArrayToGrayImage.

  The image is treated as an array of fixed point depth values.  Each
  value is converted to float and scaled by the inverse of the factor
  that was used to generate the Image object from depth values.  If
  scale_factor is specified, it should be the same value that was
  specified in the original conversion.

  The result of this function should be equal to the original input
  within the precision of the conversion.

  Args:
    image: Depth image output of FloatArrayTo[Format]Image.
    scale_factor: Fixed point scale factor.

  Returns:
    A 2D floating point numpy array representing a depth image.

  """
  image_array = np.array(image)
  image_dtype = image_array.dtype
  image_shape = image_array.shape

  channels = image_shape[2] if len(image_shape) > 2 else 1
  assert 2 <= len(image_shape) <= 3
  if channels == 3:
    # RGB image needs to be converted to 24 bit integer.
    float_array = np.sum(image_array * [65536, 256, 1], axis=2)
    if scale_factor is None:
      scale_factor = DEFAULT_RGB_SCALE_FACTOR
  else:
    if scale_factor is None:
      scale_factor = DEFAULT_GRAY_SCALE_FACTOR[image_dtype.type]
    float_array = image_array.astype(np.float32)
  scaled_array = float_array / scale_factor
  return scaled_array


def task_file_to_task_class(task_file, parent_folder = 'amsolver'):
  import importlib
  name = task_file.replace('.py', '')
  class_name = ''.join([w[0].upper() + w[1:] for w in name.split('_')])
  mod = importlib.import_module(parent_folder+".tasks.%s" % name)
  mod = importlib.reload(mod)
  task_class = getattr(mod, class_name)
  return task_class


def rgb_handles_to_mask(rgb_coded_handles):
  # rgb_coded_handles should be (w, h, c)
  # Handle encoded as : handle = R + G * 256 + B * 256 * 256
  rgb_coded_handles *= 255  # takes rgb range to 0 -> 255
  rgb_coded_handles = rgb_coded_handles.astype(int)
  return (rgb_coded_handles[:, :, 0] +
          rgb_coded_handles[:, :, 1] * 256 +
          rgb_coded_handles[:, :, 2] * 256 * 256)

'''
New functions
'''
def scale_object(obj, scale_factor: float, scale_position: bool = True) -> None:
    objectHandle = ffi.new('int[1]', [obj._handle])
    current_scale = lib.simGetObjectSizeFactor(ffi.cast('int',obj._handle))
    if hasattr(obj, "scale_factor"):
      current_scale /= obj.scale_factor
    relative_factor = scale_factor/current_scale
    if abs(relative_factor-1)>1e-2:
      lib.simScaleObjects(objectHandle, ffi.cast('int',1), ffi.cast('float', scale_factor/current_scale), ffi.cast('bool', scale_position))
    return relative_factor
    
def get_relative_position_xy(object1, object2, robot):
    # get relative position of object2 relative to object1 in robot coordinate frame
    # y+/-: right/left; z+/-: back/front
    rel_pos_xyz = object2.get_position(robot) - object1.get_position(robot)
    z, x, y = rel_pos_xyz[0], rel_pos_xyz[1], rel_pos_xyz[2]
    if z>0.05:
      direction = 'top'
    else:
      if y > x and y > -x:
          direction = 'front'
      elif y > x and y < -x:
          direction = 'left'
      elif y < x and y > -x:
          direction = 'right'
      else:
          direction = 'rear'
    return direction
  
def exchange_objects(target: Shape, source: Shape):
    visual_name = '{}_visual'.format(target.get_name())
    target_visual = Shape(visual_name)
    target_bbox = np.array(target_visual.get_bounding_box())
    source_bbox = np.array(source.get_bounding_box())
    scaling_factor = (np.prod(target_bbox[1:6:2]-target_bbox[0:6:2])/np.prod(source_bbox[1:6:2]-source_bbox[0:6:2]))**(1/3)
    scale_object(source, scaling_factor)

    source.set_pose(target.get_pose())

    source.set_parent(target.get_parent())
    for obj in target.get_objects_in_tree(first_generation_only=True):
      if 'visual' not in obj.get_name():
        obj.set_parent(source)
        if obj.get_type()=='DUMMY':
          obj.set_pose(obj.get_pose(relative_to=target_visual), relative_to=source)
    
    target.remove()
    if Shape.exists(visual_name):
      Shape(visual_name).remove()
    
    return source, Shape('{}_visual'.format(source.get_name()))

def WriteCustomDataBlock(objectHandle, tagName, data):
  # dataSize = ffi.cast('int', len(data))
  # objectHandle = ffi.cast('int', objectHandle)
  # tagName = ffi.new('char[]', tagName.encode('ascii'))
  # data = ffi.new('char[]', data.encode('ascii'))
  # lib.simWriteCustomDataBlock(objectHandle, tagName, data, dataSize)
  pyrep_utils.script_call('_WriteCustomDataBlock@PyRep', PYREP_SCRIPT_TYPE, 
      ints=[objectHandle], strings=[tagName, data])

def ReadCustomDataBlock(objectHandle, tagName):
  # dataSize = ffi.new('int*')
  # objectHandle = ffi.cast('int', objectHandle)
  # tagName = ffi.new('char[]', tagName.encode('ascii'))
  # data = lib.simReadCustomDataBlock(objectHandle, tagName, dataSize)
  data_string = None
  try:
    # data_string = ffi.string(data).decode('utf-8')
    data = pyrep_utils.script_call('_ReadCustomDataBlock@PyRep', PYREP_SCRIPT_TYPE, 
      ints=[objectHandle], strings=[tagName])
    data_string = data[2][0]
  except:
    pass
  return data_string

def ReadCustomDataBlockTags(objectHandle):
  # tagCount = ffi.new('int*')
  # objectHandle = ffi.cast('int', objectHandle)
  # data = lib.simReadCustomDataBlockTags(objectHandle, tagCount)
  data_string = None
  try:
    data = pyrep_utils.script_call('_ReadCustomDataBlockTags@PyRep', PYREP_SCRIPT_TYPE, 
      ints=[objectHandle])
    data_string = data[2]
  except:
    pass
  return data_string

def test_reachability(arm: Arm, pose, try_ik_sampling=False, linear=False, ignore_collisions=False):
  new_target = Dummy.create()
  new_target.set_matrix(pose)
  pos, ori = new_target.get_position(), new_target.get_orientation()
  res, path = False, None
  success = False
  try:
      _ = arm.solve_ik_via_jacobian(pos, ori)
      success = True
  except:
    if try_ik_sampling:
      try:
          _ = arm.solve_ik_via_sampling(pos, ori) # much slower than jacobian
          success = True
      except:
          pass
    else:
      pass
  if success:
      try:
          path = arm.get_linear_path(pos, ori, ignore_collisions=ignore_collisions) if linear else arm.get_path(pos, ori, ignore_collisions=ignore_collisions)
          if sum(path._get_path_point_lengths()) == 0:
            res = False
          else:
            res = True
      except:
          pass
  new_target.remove()
  return res, path

# execute_path, grasp, release are copied from amsolver/backend/scene.py
def execute_path(path, pyrep):
  # print('executing path')
  if sum(path._get_path_point_lengths()) == 0:
    return None
  done = False
  while not done:
      done = path.step()
      for _ in range(2):
        pyrep.step()
  # add additional steps to ensure finish of current path
  # for _ in range(100):
  #   pyrep.step()
  path._path_done = False
  path._rml_handle = None
  return path

def execute_grasp(gripper: Gripper, obj: Object, pyrep: PyRep, release=False):
  # print('executing grasp')
  done = False
  step = 0.2
  success_grasp_distance = 0.015
  action = 1 if release else 0
  while not done:
      done = gripper.actuate(action, step)
      pyrep.step()
  done = False
  if release:
    gripper.release()
  else:
    if hasattr(obj, 'graspable'): # all True for graspable objects
      if obj.graspable == True:
        done = gripper._proximity_sensor.is_detected(obj) and sum(gripper.get_joint_positions()) > success_grasp_distance
        if done:
          gripper.grasp(obj)
      else: #for real-grasp
        done = sum(gripper.get_joint_positions()) > success_grasp_distance
    else:
      done = sum(gripper.get_joint_positions()) > success_grasp_distance
  return done

def exportMesh(objects, output_format, output_path):
  output_format_list = ['obj','text_stl','binary_stl','dae','text_ply','binary_ply']
  assert output_format in output_format_list

  if 'obj' in output_format:
    suffix = '.obj'
  elif 'stl' in output_format:
    suffix = '.stl'
  elif 'ply' in output_format:
    suffix = '.ply'
  elif 'dae' in output_format:
    suffix = '.dae'
  else:
    raise ValueError('output format should choose inside [obj,stl,ply,dae]')
  output_path = output_path+suffix
  object_handles = []
  if type(objects) is list:
    for obj in objects:
      object_handles.append(obj._handle)
  else:
    object_handles.append(objects._handle)
  pyrep_utils.script_call('_exportMesh@PyRep', 
                  PYREP_SCRIPT_TYPE, ints=object_handles, strings=[output_format,output_path])

def pose_differences(pose1, pose2):
  q1 = R.from_matrix(pose1[:3, :3]).as_quat()
  q2 = R.from_matrix(pose2[:3, :3]).as_quat()
  quat_diff = 1-abs(q1.dot(q2))
  translate_diff = ((pose1[:3, 3]-pose2[:3, 3])**2).sum()**(1/2)
  return translate_diff, quat_diff
  
def get_sorted_grasp_pose(obj_pose, local_grasp_pose, sort_key="vertical"):
  grasp_pose = np.einsum('ij,kjl->kil', obj_pose, local_grasp_pose)
  angle_z_z_axis = grasp_pose[:,2, 2]
  angle_z_x_axis = grasp_pose[:,0, 2]
  grasp_pose = grasp_pose[np.logical_and(angle_z_z_axis<0.1, angle_z_x_axis>-0.1)]
  np.random.shuffle(grasp_pose)
  if sort_key=="vertical":
    axis_angle = grasp_pose[:,2, 2]
  elif sort_key=="horizontal":
    axis_angle = abs(grasp_pose[:,2, 2])-grasp_pose[:,0, 2]**2
    # sort_select = np.argsort(np.abs(angle_z_z_axis))
  elif type(sort_key)==list:
    # New axis(sort_key[0]) in the world axis(sort_key[1])
    axis = ["x", "y", "z"]
    axis_angle = grasp_pose[:,axis.index(sort_key[0]), axis.index(sort_key[1])]
    if "abs" in sort_key[2]:
      axis_angle = abs(axis_angle)
    elif "neg" in sort_key[2]:
      axis_angle = -axis_angle
  else:
    raise ValueError('sort_key can only be vertical first or horizontal first')
  sort_select = np.argsort(axis_angle)
  grasp_pose = grasp_pose[sort_select]
  return grasp_pose

def get_local_grasp_pose(obj: Object, ply_file: str, grasp_pose_path = './vlm/grasp_poses/', need_rebuild=False, use_meshlab=True, crop_box:Object =None):
  args = define_default_args()
  args.input_file = ply_file
  args.output_file = os.path.join(grasp_pose_path, ply_file.split('/')[-1][:-4] + '.pkl')
  args.use_meshlab = use_meshlab
  obj_origin_matrix = obj.get_matrix()
  if crop_box is not None:
    args.no_crop = False
    crop_boundary = crop_box.get_bounding_box()
    crop_origin_matrix = crop_box.get_matrix()
    args.crop_box_transform = crop_origin_matrix
    args.crop_x_min = crop_boundary[0]
    args.crop_y_min = crop_boundary[2]
    args.crop_z_min = crop_boundary[4]
    args.crop_x_max = crop_boundary[1]
    args.crop_y_max = crop_boundary[3]
    args.crop_z_max = crop_boundary[5]
  gl = Grasploc(args)
  # scale_object(self.toy, 0.5)
  
  if not os.path.exists(args.output_file) or need_rebuild:
    exportMesh(obj, 'binary_ply', ply_file[:-4])
    gl.run(obj_origin_matrix)
  else:
    gl.load_result()
  return gl.se3_output

def add_joint(jointType, jointMode=sim.sim_jointmode_force, length=0.2, diameter=0.02): # joint axis is along Z
    jointMode = ffi.cast('int', jointMode)
    if jointType == 'revolute':
        j_t = JointType.REVOLUTE
    elif jointType == 'prismatic':
        j_t = JointType.PRISMATIC
    elif jointType == 'spherical':
        j_t = JointType.SPHERICAL
    else:
        raise ValueError('the wrong joint type')
    j_t = ffi.cast('int', j_t.value)
    sizes = ffi.new('float[]',[length, diameter])
    color_A = ffi.new('float[]',[1,0,0])
    color_B = ffi.new('float[]',[0,0,1])
    handle = lib.simCreateJoint(j_t, jointMode, ffi.cast('int',0), sizes, color_A, color_B)
    return Joint(int(handle))

def create_rotation_joint(container_pour, container_recv):
  joint = add_joint('revolute', length=0.002, diameter=0.002)
  joint.set_joint_interval(False, [-np.pi, np.pi])
  joint.set_model_dynamic(True)
  joint_mat = np.zeros((3, 4)) # Z to be tangent, Y to be pointing from container_pour to container_recv in horizontal plane, X is always (0, 0, -1)
  joint_mat[:, 0] = [0, 0, -1]
  joint_mat[:, 1] = container_pour.get_position() - container_recv.get_position()
  joint_mat[2, 1] = 0
  joint_mat[:, 2] = np.cross(joint_mat[:, 0], joint_mat[:, 1])
  joint_mat[:, 3] = container_pour.get_position()
  joint.set_matrix(joint_mat)
  joint.set_parent(container_pour.get_parent())
  container_pour.set_parent(joint)
  return joint

def select_color(index, other_color_numbers, replace=True):
  target_color_name, target_rgb = colors[index]

  color_names, rgbs = [target_color_name], [target_rgb]
  if other_color_numbers!=0:
    random_idx = np.random.choice(len(colors), other_color_numbers, replace=replace)
    while index in random_idx:
      random_idx = np.random.choice(len(colors), other_color_numbers, replace=replace)
    for i in random_idx:
      name, rgb = colors[i]
      color_names.append(name)
      rgbs.append(rgb)
  return color_names, rgbs

def import_distractors(pyrep: PyRep, path = './vlm/asset', select_number=5, scale=1e-4):
  models_path = []
  for path in Path(path).rglob('*.ttm'):
    models_path.append(path)
  idx = np.random.choice(len(models_path), size=select_number)
  models = []
  for i in idx:
    model = pyrep.import_model(str(models_path[i]))
    model.set_model_dynamic(True)
    model.set_model_collidable(True)
    bbox = np.array(model.get_bounding_box())
    scaling_factor = (scale/np.prod(bbox[1:6:2]-bbox[0:6:2]))**(1/3)
    scale_object(model, scaling_factor)
    models.append(model)
  return models
    
