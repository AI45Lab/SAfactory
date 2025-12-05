from gymnasium.envs.registration import register
import amsolver.backend.task as task
import os
from amsolver.utils import name_to_task_class
from amsolver.gym.vlmbench_env import VLMBenchEnv

TASKS = [t for t in os.listdir(task.TASKS_PATH)
         if t != '__init__.py' and t.endswith('.py')]

for task_file in TASKS:
    task_name = task_file.split('.py')[0]
    task_class = name_to_task_class(task_name)
    register(
        id='%s-state-v0' % task_name,
        entry_point='amsolver.gym:VLMBenchEnv',
        kwargs={
            'task_class': task_class,
            'observation_mode': 'state'
        }
    )
    register(
        id='%s-vision-v0' % task_name,
        entry_point='amsolver.gym:VLMBenchEnv',
        kwargs={
            'task_class': task_class,
            'observation_mode': 'vision'
        }
    )
