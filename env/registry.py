from typing import Type

def _import_android_gym() -> Type:
    from env.androidgym.android_env import AndroidGym
    return AndroidGym


def _import_os_env() -> Type:
    from  env.osgym.os_env import OSGym
    return OSGym


def _import_mc_env() -> Type:
    from env.mc.mc_env import  MCGym
    return MCGym

def _import_mc_gpu_env() -> Type:
    from env.mc_gpu_gym.mc_gpu_env import MCGPUGym
    return MCGPUGym

def _import_emb_env() -> Type:
    from env.embodiedgym.embodied_env import  EmbodiedAlfredGym
    return EmbodiedAlfredGym

def _import_dab_env() -> Type:
    from env.dabstep.dabstep_env import  DABStepEnv
    return DABStepEnv

def _import_dw_env():
    from env.dwgym.dw_env import DiscoveryWorldEnv
    return DiscoveryWorldEnv

def _import_geo3k_vl_test_env() -> Type:
    from env.geo3k_vl_test.geo3k_vl_test_env import Geo3kVLTestEnv
    return Geo3kVLTestEnv

def _import_qa_gym() -> Type:
    from env.qagym.qa_env import QAGym
    return QAGym


def _import_deepeyes_env() -> Type:
    from env.deepeyes.deepeyes_env import DeepEyesEnv
    return DeepEyesEnv


def _import_robotrustbench_env() -> Type:
    from env.robotrustbench.robotrustbench_env import RoboTrustBenchEnv
    return RoboTrustBenchEnv


def _import_math500_text_env() -> Type:
    from env.math500_text.math500_text_env import Math500TextEnv
    return Math500TextEnv
