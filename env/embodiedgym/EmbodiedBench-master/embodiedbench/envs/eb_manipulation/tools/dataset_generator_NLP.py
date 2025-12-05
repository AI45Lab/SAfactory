from multiprocessing import Process, Manager
from time import time

from pyrep.const import RenderMode
from os.path import join, dirname, abspath, isfile
import sys
CURRENT_DIR = dirname(abspath(__file__))
sys.path.insert(0, join(CURRENT_DIR, '..'))  # Use local amsolver rather than installed
from amsolver import ObservationConfig
from amsolver.action_modes import ActionMode
from amsolver.backend.utils import task_file_to_task_class
from amsolver.environment import Environment
import amsolver.backend.task as task

import os
import pickle
from PIL import Image
from amsolver.backend import utils
from amsolver.backend.const import *
import numpy as np
from pathlib import Path

from absl import app
from absl import flags

"""
Change to different folder for different split. 
If you want to generate unseen data, please set the colors and objects to the unseen collections in amsolver/constant.py
"""

FLAGS = flags.FLAGS

flags.DEFINE_string('save_path',
                    './rlbench_data/train',
                    'Where to save the demos.')
flags.DEFINE_list('tasks', [
                            'drop_pen_color', 'drop_pen_relative', 'drop_pen_size',
                            'wipe_table_color', 'wipe_table_relative', 'wipe_table_shape', 'wipe_table_size', 'wipe_table_direction',
                            'pour_demo_color', 'pour_demo_relative', 'pour_demo_size',
                            'pick_cube_color', 'pick_cube_relative', 'pick_cube_shape', 'pick_cube_size',
                            'stack_cubes_color', 'stack_cubes_size',
                            'stack_cubes_relative', 'stack_cubes_shape',
                            'place_into_shape_sorter_color', 'place_into_shape_sorter_shape', 'place_into_shape_sorter_relative',
                            'open_drawer',
                            'open_door_complex'
                            ],
                  'The tasks to collect. If empty, all tasks are collected.')
flags.DEFINE_list('image_size', [360, 360],
                  'The size of the images tp save.')
flags.DEFINE_enum('renderer',  'opengl', ['opengl', 'opengl3'],
                  'The renderer to use. opengl does not include shadows, '
                  'but is faster.')
flags.DEFINE_integer('processes', 8,
                     'The number of parallel processes during collection.')
flags.DEFINE_integer('episodes_per_task', 5,
                     'The number of episodes to collect per task.')
flags.DEFINE_integer('variations', -1,
                     'Number of variations to collect per task. -1 for all.')
flags.DEFINE_bool('save_configs', True,
                     'whether also save the config for replay.')


def check_and_make(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)

def save_configs(task_base, waypoint_sets, config, example_path):
    check_and_make(example_path)
    # Save the low-dimension data
    with open(os.path.join(example_path, "configs.pkl"), 'wb') as f:
        pickle.dump(config, f)
    task_base.save_model(os.path.join(example_path, "task_base.ttm"))
    waypoint_sets.save_model(os.path.join(example_path, "waypoint_sets.ttm"))

def save_demo(demo, example_path):

    # Save image data first, and then None the image data, and pickle
    left_shoulder_rgb_path = os.path.join(
        example_path, LEFT_SHOULDER_RGB_FOLDER)
    left_shoulder_depth_path = os.path.join(
        example_path, LEFT_SHOULDER_DEPTH_FOLDER)
    left_shoulder_mask_path = os.path.join(
        example_path, LEFT_SHOULDER_MASK_FOLDER)
    right_shoulder_rgb_path = os.path.join(
        example_path, RIGHT_SHOULDER_RGB_FOLDER)
    right_shoulder_depth_path = os.path.join(
        example_path, RIGHT_SHOULDER_DEPTH_FOLDER)
    right_shoulder_mask_path = os.path.join(
        example_path, RIGHT_SHOULDER_MASK_FOLDER)
    overhead_rgb_path = os.path.join(
        example_path, OVERHEAD_RGB_FOLDER)
    overhead_depth_path = os.path.join(
        example_path, OVERHEAD_DEPTH_FOLDER)
    overhead_mask_path = os.path.join(
        example_path, OVERHEAD_MASK_FOLDER)
    wrist_rgb_path = os.path.join(example_path, WRIST_RGB_FOLDER)
    wrist_depth_path = os.path.join(example_path, WRIST_DEPTH_FOLDER)
    wrist_mask_path = os.path.join(example_path, WRIST_MASK_FOLDER)
    front_rgb_path = os.path.join(example_path, FRONT_RGB_FOLDER)
    front_depth_path = os.path.join(example_path, FRONT_DEPTH_FOLDER)
    front_mask_path = os.path.join(example_path, FRONT_MASK_FOLDER)

    check_and_make(left_shoulder_rgb_path)
    check_and_make(left_shoulder_depth_path)
    check_and_make(left_shoulder_mask_path)
    check_and_make(right_shoulder_rgb_path)
    check_and_make(right_shoulder_depth_path)
    check_and_make(right_shoulder_mask_path)
    check_and_make(overhead_rgb_path)
    check_and_make(overhead_depth_path)
    check_and_make(overhead_mask_path)
    check_and_make(wrist_rgb_path)
    check_and_make(wrist_depth_path)
    check_and_make(wrist_mask_path)
    check_and_make(front_rgb_path)
    check_and_make(front_depth_path)
    check_and_make(front_mask_path)

    for i, obs in enumerate(demo):
        left_shoulder_rgb = Image.fromarray(obs.left_shoulder_rgb)
        left_shoulder_depth = utils.float_array_to_rgb_image(
            obs.left_shoulder_depth, scale_factor=DEPTH_SCALE)
        left_shoulder_mask = Image.fromarray(
            (obs.left_shoulder_mask * 255).astype(np.uint8))
        right_shoulder_rgb = Image.fromarray(obs.right_shoulder_rgb)
        right_shoulder_depth = utils.float_array_to_rgb_image(
            obs.right_shoulder_depth, scale_factor=DEPTH_SCALE)
        right_shoulder_mask = Image.fromarray(
            (obs.right_shoulder_mask * 255).astype(np.uint8))
        overhead_rgb = Image.fromarray(obs.overhead_rgb)
        overhead_depth = utils.float_array_to_rgb_image(
            obs.overhead_depth, scale_factor=DEPTH_SCALE)
        overhead_mask = Image.fromarray(
            (obs.overhead_mask * 255).astype(np.uint8))
        wrist_rgb = Image.fromarray(obs.wrist_rgb)
        wrist_depth = utils.float_array_to_rgb_image(
            obs.wrist_depth, scale_factor=DEPTH_SCALE)
        wrist_mask = Image.fromarray((obs.wrist_mask * 255).astype(np.uint8))
        front_rgb = Image.fromarray(obs.front_rgb)
        front_depth = utils.float_array_to_rgb_image(
            obs.front_depth, scale_factor=DEPTH_SCALE)
        front_mask = Image.fromarray((obs.front_mask * 255).astype(np.uint8))

        left_shoulder_rgb.save(
            os.path.join(left_shoulder_rgb_path, IMAGE_FORMAT % i))
        left_shoulder_depth.save(
            os.path.join(left_shoulder_depth_path, IMAGE_FORMAT % i))
        left_shoulder_mask.save(
            os.path.join(left_shoulder_mask_path, IMAGE_FORMAT % i))
        right_shoulder_rgb.save(
            os.path.join(right_shoulder_rgb_path, IMAGE_FORMAT % i))
        right_shoulder_depth.save(
            os.path.join(right_shoulder_depth_path, IMAGE_FORMAT % i))
        right_shoulder_mask.save(
            os.path.join(right_shoulder_mask_path, IMAGE_FORMAT % i))
        overhead_rgb.save(
            os.path.join(overhead_rgb_path, IMAGE_FORMAT % i))
        overhead_depth.save(
            os.path.join(overhead_depth_path, IMAGE_FORMAT % i))
        overhead_mask.save(
            os.path.join(overhead_mask_path, IMAGE_FORMAT % i))
        wrist_rgb.save(os.path.join(wrist_rgb_path, IMAGE_FORMAT % i))
        wrist_depth.save(os.path.join(wrist_depth_path, IMAGE_FORMAT % i))
        wrist_mask.save(os.path.join(wrist_mask_path, IMAGE_FORMAT % i))
        front_rgb.save(os.path.join(front_rgb_path, IMAGE_FORMAT % i))
        front_depth.save(os.path.join(front_depth_path, IMAGE_FORMAT % i))
        front_mask.save(os.path.join(front_mask_path, IMAGE_FORMAT % i))

        # We save the images separately, so set these to None for pickling.
        obs.left_shoulder_rgb = None
        obs.left_shoulder_depth = None
        obs.left_shoulder_point_cloud = None
        obs.left_shoulder_mask = None
        obs.right_shoulder_rgb = None
        obs.right_shoulder_depth = None
        obs.right_shoulder_point_cloud = None
        obs.right_shoulder_mask = None
        obs.overhead_rgb = None
        obs.overhead_depth = None
        obs.overhead_point_cloud = None
        obs.overhead_mask = None
        obs.wrist_rgb = None
        obs.wrist_depth = None
        obs.wrist_point_cloud = None
        obs.wrist_mask = None
        obs.front_rgb = None
        obs.front_depth = None
        obs.front_point_cloud = None
        obs.front_mask = None

    # Save the low-dimension data
    with open(os.path.join(example_path, LOW_DIM_PICKLE), 'wb') as f:
        pickle.dump(demo, f)


def run(i, lock, task_index, variation_count, results, file_lock, tasks):
    """Each thread will choose one task and variation, and then gather
    all the episodes_per_task for that variation."""

    # Initialise each thread with random seed
    np.random.seed(None)
    num_tasks = len(tasks)

    img_size = list(map(int, FLAGS.image_size))

    obs_config = ObservationConfig()
    obs_config.set_all(True)
    obs_config.right_shoulder_camera.image_size = img_size
    obs_config.left_shoulder_camera.image_size = img_size
    obs_config.overhead_camera.image_size = img_size
    obs_config.wrist_camera.image_size = img_size
    obs_config.front_camera.image_size = img_size

    # Store depth as 0 - 1
    obs_config.right_shoulder_camera.depth_in_meters = False
    obs_config.left_shoulder_camera.depth_in_meters = False
    obs_config.overhead_camera.depth_in_meters = False
    obs_config.wrist_camera.depth_in_meters = False
    obs_config.front_camera.depth_in_meters = False

    # We want to save the masks as rgb encodings.
    obs_config.left_shoulder_camera.masks_as_one_channel = False
    obs_config.right_shoulder_camera.masks_as_one_channel = False
    obs_config.overhead_camera.masks_as_one_channel = False
    obs_config.wrist_camera.masks_as_one_channel = False
    obs_config.front_camera.masks_as_one_channel = False

    if FLAGS.renderer == 'opengl':
        obs_config.right_shoulder_camera.render_mode = RenderMode.OPENGL
        obs_config.left_shoulder_camera.render_mode = RenderMode.OPENGL
        obs_config.overhead_camera.render_mode = RenderMode.OPENGL
        obs_config.wrist_camera.render_mode = RenderMode.OPENGL
        obs_config.front_camera.render_mode = RenderMode.OPENGL

    amsolver_env = Environment(
        action_mode=ActionMode(),
        obs_config=obs_config,
        headless=True) # set headless=False, if user want to visualize the simulator 
    amsolver_env.launch()

    task_env = None

    tasks_with_problems = results[i] = ''

    while True:
        # Figure out what task/variation this thread is going to do
        with lock:

            if task_index.value >= num_tasks:
                print('Process', i, 'finished')
                break

            my_variation_count = variation_count.value
            t = tasks[task_index.value]
            task_env = amsolver_env.get_task(t)
            var_target = task_env.variation_count()
            if FLAGS.variations >= 0:
                var_target = np.minimum(FLAGS.variations, var_target)
            if my_variation_count >= var_target:
                # If we have reached the required number of variations for this
                # task, then move on to the next task.
                variation_count.value = my_variation_count = 0
                task_index.value += 1

            variation_count.value += 1
            if task_index.value >= num_tasks:
                print('Process', i, 'finished')
                break
            t = tasks[task_index.value]

        task_env = amsolver_env.get_task(t)
        task_env.set_variation(my_variation_count)

        variation_path = os.path.join(
            FLAGS.save_path, task_env.get_name(),
            VARIATIONS_FOLDER % my_variation_count)

        check_and_make(variation_path)

        episodes_path = os.path.join(variation_path, EPISODES_FOLDER)
        check_and_make(episodes_path)
        current_episodes = list(Path(episodes_path).rglob("low_dim*"))

        fails_path = os.path.join(variation_path, 'fail_cases')
        check_and_make(fails_path)

        abort_variation = False
        ex_idx = len(current_episodes)
        fail_idx = 0
        need_fail_case = False
        # for ex_idx in range(FLAGS.episodes_per_task):
        while ex_idx<FLAGS.episodes_per_task:
            print('Process', i, '// Task:', task_env.get_name(),
                  '// Variation:', my_variation_count, '// Demo:', ex_idx)
            attempts = 10
            while attempts > 0:
                try:
                    # TODO: for now we do the explicit looping.
                    t0 = time()
                    demo_success= task_env.get_demos(
                        amount=1,
                        live_demos=True)
                    if len(demo_success)==2:
                        demo = demo_success[0][0]
                        success = demo_success[1][0]
                    else:
                        demo = demo_success[0][0]
                    print(f"one demo for {task_env.get_name()} // Variation {my_variation_count}: {time()-t0}, success: {success}")
                except Exception as e:
                    print(e)
                    attempts -= 1
                    if attempts > 0:
                        continue
                    problem = (
                        'Process %d failed collecting task %s (variation: %d, '
                        'example: %d). Skipping this task/variation.\n%s\n' % (
                            i, task_env.get_name(), my_variation_count, ex_idx,
                            str(e))
                    )
                    print(problem)
                    tasks_with_problems += problem
                    abort_variation = True
                    break
                if success:
                    episode_path = os.path.join(episodes_path, EPISODE_FOLDER % (ex_idx))
                    # with file_lock:
                    save_demo(demo, episode_path)
                    if FLAGS.save_configs:
                        task_base, waypoint_sets, config = task_env.read_config(demo.high_level_instructions)
                        # with file_lock:
                        save_configs(task_base, waypoint_sets, config, episode_path)
                    ex_idx += 1
                    break
                else:
                    if need_fail_case:
                        if fail_idx<FLAGS.episodes_per_task:
                            fail_path = os.path.join(fails_path, EPISODE_FOLDER % fail_idx)
                            with file_lock:
                                save_demo(demo, fail_path)
                            fail_idx+=1
                        else:
                            attempts -= 1
                            if attempts > 0:
                                continue
                            problem = (
                            'Process %d failed collecting task %s (variation: %d, '
                            'example: %d). Skipping this task/variation.\n' % (
                                i, task_env.get_name(), my_variation_count, ex_idx)
                            )
                            print(problem)
                            tasks_with_problems += problem
                            abort_variation = True
                    else:
                        attempts -= 1
            if abort_variation:
                break

    results[i] = tasks_with_problems
    amsolver_env.shutdown()


def main(argv):

    tasks_path = './vlm/tasks'
    task_files = [t.replace('.py', '') for t in os.listdir(tasks_path)
                  if t != '__init__.py' and t.endswith('.py')]
    
    if len(FLAGS.tasks) > 0:
        for t in FLAGS.tasks:
            if t not in task_files:
                raise ValueError('Task %s not recognised!.' % t)
        task_files = FLAGS.tasks

    tasks = [task_file_to_task_class(t, parent_folder = 'vlm') for t in task_files]

    manager = Manager()

    result_dict = manager.dict()
    file_lock = manager.Lock()

    task_index = manager.Value('i', 0)
    variation_count = manager.Value('i', 0)
    lock = manager.Lock()

    check_and_make(FLAGS.save_path)

    processes = [Process(
        target=run, args=(
            i, lock, task_index, variation_count, result_dict, file_lock,
            tasks))
        for i in range(FLAGS.processes)]
    [t.start() for t in processes]
    [t.join() for t in processes]

    print('Data collection done!')
    for i in range(FLAGS.processes):
        print(result_dict[i])


if __name__ == '__main__':
  app.run(main)
