import codecs
import os.path

from setuptools import setup


# Version meaning (X.Y.Z)
# X: Major version (e.g. vastly different scene, platform, etc)
# Y: Minor version (e.g. new tasks, major changes to existing tasks, etc)
# Z: Patch version (e.g. small changes to tasks, bug fixes, etc)


def read(rel_path):
    here = os.path.abspath(os.path.dirname(__file__))
    with codecs.open(os.path.join(here, rel_path), 'r') as fp:
        return fp.read()


def get_version(rel_path):
    for line in read(rel_path).splitlines():
        if line.startswith('__version__'):
            delim = '"' if '"' in line else "'"
            return line.split(delim)[1]
    else:
        raise RuntimeError("Unable to find version string.")


setup(name='amsolver',
      version=get_version("amsolver/__init__.py"),
      description='AMSolver',
      packages=[
            'amsolver',
            'amsolver.backend',
            # 'amsolver.tasks',
            # 'amsolver.task_ttms',
            'amsolver.robot_ttms',
            'amsolver.sim2real',
            # 'amsolver.assets',
            'amsolver.gym'
      ],
      package_data={'': ['*.ttm', '*.obj', '**/**/*.ttm', '**/**/*.obj'],
                    'amsolver': ['task_design.ttt']},
      )
