from multiprocessing import Process, Manager
from time import time
import cv2
from pyrep.objects.dummy import Dummy
from pyrep.objects.vision_sensor import VisionSensor
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
import random

from absl import app
from absl import flags

def set_seed(seed, torch=False):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)

    if torch:
        import torch
        torch.manual_seed(seed)
random_seed = random.randint(1, 10000)
print(f'random seed: {random_seed}')
set_seed(random_seed)
FLAGS = flags.FLAGS

flags.DEFINE_string('save_path',
                    './rlbench_data/test/seen',
                    'Where to save the demos.')
flags.DEFINE_list('tasks', [
                            'place_into_shape_sorter_color',
                            'place_into_shape_sorter_shape', 'place_into_shape_sorter_relative',
                            'drop_pen_color', 'drop_pen_relative', 'drop_pen_size',
                            'wipe_table_color', 'wipe_table_relative', 'wipe_table_shape', 'wipe_table_size', 'wipe_table_direction',
                            'pour_demo_color', 'pour_demo_relative', 'pour_demo_size',
                            'pick_cube_color', 'pick_cube_relative', 'pick_cube_shape', 'pick_cube_size',
                            'stack_cubes_color', 'stack_cubes_size',
                            'stack_cubes_relative', 'stack_cubes_shape',
                            'open_door_complex',
                            'open_drawer'
],
                  'The tasks to collect. If empty, all tasks are collected.')
flags.DEFINE_list('image_size', [360, 360],
                  'The size of the images tp save.')
flags.DEFINE_enum('renderer',  'opengl', ['opengl', 'opengl3'],
                  'The renderer to use. opengl does not include shadows, '
                  'but is faster.')
flags.DEFINE_integer('processes', 16,
                     'The number of parallel processes during collection.')
flags.DEFINE_integer('episodes_per_task', 5,
                     'The number of episodes to collect per task.')
flags.DEFINE_integer('episodes_per_task_all_variations', 100,
                     'The number of episodes to collect per task.')
flags.DEFINE_integer('variations', -1,
                     'Number of variations to collect per task. -1 for all.')

class Recorder(object):
    def __init__(self) -> None:
        cam_placeholder = Dummy('cam_cinematic_placeholder')
        self.cam = VisionSensor.create([320, 320], view_angle=30)
        self.cam.set_pose(cam_placeholder.get_pose())
        self.cam.set_parent(cam_placeholder)
        self._snaps = []
        self._fps=30

    def take_snap(self):
        self._snaps.append(
            (self.cam.capture_rgb() * 255.).astype(np.uint8))
    
    def save_video(self, path):
        print('Converting to video ...')
        os.makedirs(os.path.dirname(path), exist_ok=True)
        video = cv2.VideoWriter(
                path, cv2.VideoWriter_fourcc(*'MJPG'), self._fps,
                tuple(self.cam.get_resolution()))
        for image in self._snaps:
            video.write(cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        video.release()
        self._snaps = []

    def save_image(self, path):
        print('Saving the image ...')
        cv2.imwrite(path, cv2.cvtColor(self._snaps[0], cv2.COLOR_RGB2BGR))
        self._snaps = []

def check_and_make(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)


def save_demo(task_base, waypoint_sets, config, example_path):
    check_and_make(example_path)
    # Save the low-dimension data
    with open(os.path.join(example_path, "configs.pkl"), 'wb') as f:
        pickle.dump(config, f)
    task_base.save_model(os.path.join(example_path, "task_base.ttm"))
    waypoint_sets.save_model(os.path.join(example_path, "waypoint_sets.ttm"))

def run(i, lock, task_index, variation_count, results, file_lock, tasks):
    """Each thread will choose one task and variation, and then gather
    all the episodes_per_task for that variation."""

    # Initialise each thread with random seed
    np.random.seed(None)
    num_tasks = len(tasks)

    img_size = list(map(int, FLAGS.image_size))

    obs_config = ObservationConfig()
    obs_config.set_all_high_dim(False)
    obs_config.set_all_low_dim(True)

    amsolver_env = Environment(
        action_mode=ActionMode(),
        obs_config=obs_config,
        headless=True)
    amsolver_env.launch()
    # recorder = Recorder()
    recorder = None
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

        if FLAGS.episodes_per_task_all_variations>0:
            FLAGS.episodes_per_task = (FLAGS.episodes_per_task_all_variations // task_env.variation_count())
        variation_path = os.path.join(
            FLAGS.save_path, task_env.get_name(),
            VARIATIONS_FOLDER % my_variation_count)

        check_and_make(variation_path)

        episodes_path = os.path.join(variation_path, EPISODES_FOLDER)
        check_and_make(episodes_path)
        current_episodes = os.listdir(episodes_path)

        abort_variation = False
        ex_idx = len(current_episodes)
        # for ex_idx in range(FLAGS.episodes_per_task):
        while ex_idx<FLAGS.episodes_per_task:
            print('Process', i, '// Task:', task_env.get_name(),
                  '// Variation:', my_variation_count, '// Demo:', ex_idx)
            attempts = 10
            while attempts > 0:
                try:
                    task_base, waypoint_sets, config = task_env.save_config()
                    if recorder is not None:
                        recorder.take_snap()
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
                episode_path = os.path.join(episodes_path, EPISODE_FOLDER % (ex_idx))
                with file_lock:
                    save_demo(task_base, waypoint_sets, config, episode_path)
                    if recorder is not None:
                        recorder.save_image(os.path.join(episode_path, "image.png"))
                ex_idx += 1
                break
            if abort_variation:
                break

    results[i] = tasks_with_problems
    amsolver_env.shutdown()


def main(argv):
    # tasks_path = './language_tasks/tasks'
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
