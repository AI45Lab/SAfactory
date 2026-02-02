# # Copyright (c) 2020 All Rights Reserved
# # Author: William H. Guss, Brandon Houghton
import rich
import traceback
import csv
from copy import deepcopy
import json
import logging
import os
import struct
from .malmo import InstanceManager, MinecraftInstance, launch_queue_logger_thread, malmo_version
import uuid
import coloredlogs
import gym
import socket
import time, collections
from lxml import etree
from . import comms
import xmltodict
from concurrent.futures import ThreadPoolExecutor
import cv2
from PIL import ImageGrab
import numpy as np
import subprocess
import pathlib
from Xlib import display as Xdisplay
import socket, errno, time

from typing import Any, Callable, Dict, List, Optional, Tuple

MAX_RESETTING_ENV_COUNT = max(4, os.cpu_count() // 12) if os.cpu_count() is not None else 20

NS = "{http://ProjectMalmo.microsoft.com}"
STEP_OPTIONS = 0

MAX_WAIT = 600 * 5  # Time to wait before raising an exception (high value because some operations we wait on are very slow)
# SOCKTIME = 60.0 * 4 * 5  # After this much time a socket exception will be thrown.
SOCKTIME = 60. * 2
TICK_LENGTH = 0.05

logger = logging.getLogger(__name__)

class ReconnectingSocket:
    """
    对真正的 tcp socket 做一层包装：
    1. send() 时发现断线就自动重连并重发（最多重试 N 次）
    2. 外部代码照旧只调 send()，无需改业务逻辑
    """
    def __init__(self, ip, port, max_retry=3):
        self._ip, self._port = ip, port
        self._sock = None
        self.max_retry = max_retry
        self._ensure_connected()

    def _ensure_connected(self):
        """保证此时 self._sock 是活连接"""
        if self._sock is not None:
            try:   # 先简单探活
                self._sock.send(b'')
                return
            except socket.error:
                pass   # 死了，下面重新连
        # 真正重连
        for attempt in range(self.max_retry):
            try:
                self._sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                logger.warning(f"[ReconnectingSocket] port {self._port}")
                logger.warning(f"[ReconnectingSocket] ip {self._ip}")
                self._sock.connect((self._ip, self._port))
                return
            except socket.error as e:
                logger.warning(f"[ReconnectingSocket] attempt {attempt+1} failed: {e}")
                time.sleep(0.5 * (attempt + 1))
        # 实在连不上，让上层感知失败
        raise ConnectionRefusedError("ReconnectingSocket: all retries exhausted")

    def send(self, data: bytes):
        """
        发送数据，遇到断线自动重连并重发一次；
        如果仍旧失败就抛异常，让上层决定是跳过还是终止
        """
        for attempt in range(2):          # 最多“重连→重发”一次
            try:
                self._sock.sendall(data)
                return
            except (BrokenPipeError, ConnectionResetError, socket.error) as e:
                logger.warning(f"[ReconnectingSocket] send failed ({e}), try reconnect …")
                self._ensure_connected()
        # 重连后依旧失败，抛出去
        raise BrokenPipeError("ReconnectingSocket: send failed after reconnect")

    def close(self):
        if self._sock:
            try:
                self._sock.close()

            except Exception:
                pass
            self._sock = None

class _InfoReader:
    def __init__(self, path: str):
        self.path = path
        self._proc = subprocess.Popen(
                ['tail', '-n', '0', '-F', path],   # -n 0 表示只读新行
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                bufsize=1, universal_newlines=True)
        self._buffer = []

    def latest(self, timeout=0.5):
        import select
        if select.select([self._proc.stdout], [], [], timeout)[0]:
            line = self._proc.stdout.readline().rstrip('\n')
            if line:                       # 可能空行
                return json.loads(line)
        return {}

    def close(self):
        self._proc.terminate()
        self._proc.wait()


class XvfbRecorder:
    """
    只管录制，不管 Xvfb 生死
    display: 字符串，例如 ":99"
    """
    def __init__(self, display: str, fps=120, outfile="out.mp4"):
        self.display = display
        self.fps = fps
        self.outfile = pathlib.Path(outfile)
        self._ffmpeg = None
        # 自动解析分辨率（失败可手工传）
        try:
            w, h = self._get_resolution()
            self.size = f"{w}x{h}"
        except Exception:
            self.size = "1920x1200"   # 保底
        
        self.fps = 15
        self.size = "1920x1200"   # 分辨率比 "1024x768" 小

    # ---- 手工开关 ----
    def start(self):
        cmd = [
            "ffmpeg", "-y",
            "-f", "x11grab",
            "-probesize", "32M",
            "-flush_packets", "0",
            "-r", str(self.fps),
            "-s", self.size,
            "-i", ":"+str(self.display),
            "-c:v", "libx264",
            "-preset", "ultrafast",
            "-crf", "23",
            str(self.outfile)
        ]
        self._ffmpeg = subprocess.Popen(cmd)


    def stop(self):
        if self._ffmpeg is None:
            print("videoooooooooooooooooooooooooooooooo errorrrrrrrrrrrrrrrrrrrrrrrrrrr")
            return
        video_path = str(self.outfile.resolve())
        print(f"videoooooooooooooooooooooooooooooooo saved_to={video_path}")
        self._ffmpeg.terminate()
        self._ffmpeg.wait()
        self._ffmpeg = None

    # ---- 可选：自动读分辨率 ----
    def _get_resolution(self):
        d = Xdisplay.Display(":"+str(self.display))
        root = d.screen().root
        geom = root.get_geometry()
        return geom.width, geom.height




def multiagent_identity(x):
    return x
class MultiAgentEnv(gym.Env):
    """
    The MineRLEnv class, a gym environment which implements stepping, and resetting, for the MineRL
    simulator from an environment specification.


    THIS CLASS SHOULD NOT BE INSTANTIATED DIRECTLY
    USE ENV SPEC.

        Example:
            To actually create a MineRLEnv. Use any one of the package MineRL environments (Todo: Link.)
            literal blocks::

                import minestudio.simulator.minerl
                import gym

                env = gym.make('MineRLTreechop-v0') # Makes a minerl environment.

                # Use env like any other OpenAI gym environment.
                # ...

                # Alternatively:
                env = gym.make('MineRLTreechop-v0') # makes a default treechop environment (with treechop-specific action and observation spaces)
    
    """

    metadata = {'render.modes': ['human']}

    def __init__(self,
                 task, 
                 scene, 
                 working_dir,
                 output_dir, 
                 display_port,
                 xvfb,
                 instances: Optional[List[MinecraftInstance]] = None,
                 is_fault_tolerant: bool = True,
                 verbose: bool = False,
                 _xml_mutator_to_be_deprecated: Optional[Callable] = None,
                 refresh_instances_every: Optional[int] = None,
                 ):
        """
        Constructor of MineRLEnv.
        
        :param env_spec: The environment specification object.
        :param instances: A list of prelaunched Minecraft instances..
        :param is_fault_tolerant: If the instance is fault tolerant.
        :param verbose: If the MineRL env is verbose.
        :param _xml_mutator_to_be_deprecated: A function which mutates the mission XML when called.
        :param refresh_instances_every: As a band-aid to memory leaks, completely kill and rebuild the instances every
           N setups.
        """
        self.instances = instances if instances is not None else []  # type: List[MinecraftInstance]

        # TO DEPRECATE (FOR ENV_SPECS)

        self._xml_mutator_to_be_deprecated = _xml_mutator_to_be_deprecated or multiagent_identity
        self._refresh_inst_every = refresh_instances_every
        self._inst_setup_cntr = 0
        self.render_open = False

        # We use the env_spec's initial observation and action space
        # to satify the gym API

        self._init_seeding()
        self._init_viewer()
        self._init_interactive()
        self._init_fault_tolerance(is_fault_tolerant)
        self._init_logging(verbose)
        self.rec = []
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.ip = '127.0.0.1'
        self.port = 12345
        self.working_dir=working_dir
        self.task = task
        self.output_dir=output_dir
        self.display_port = display_port
        self.xvfb = xvfb
        self.xvfb = True
        self.scene = scene

    def _init_viewer(self) -> None:
        self.viewers = {}
        self._last_ac = {}
        self._last_pov = {}
        self._last_obs = {}

    def _init_interactive(self) -> None:
        self._is_interacting = False
        self._is_real_time = False
        self._last_step_time = -1

    def _init_fault_tolerance(self, is_fault_tolerant: bool) -> None:
        self._is_fault_tolerant = is_fault_tolerant
        self._last_obs = {}
        self._already_closed = False

    def _init_logging(self, verbose: bool) -> None:
        if verbose:
            coloredlogs.install(level=logging.DEBUG)

    def _init_seeding(self) -> None:
        self._seed = None

    ########### CONFIGURATION METHODS ########

    def seed(self, seed=None, seed_spaces=True):
        """Seeds the environment!

        This also seeds the aciton_space and observation_space sampling.

        Note:
        THIS MUST BE CALLED BEFORE :code:`env.reset()`
        
        Args:
            seed (long, optional):  Defaults to 42.
            seed_spaces (bool, option): If the observation space and action space shoud be seeded. Defaults to True.
        """
        assert isinstance(seed, int) or seed is None, "Seed must be an int!"
        self._seed = seed
        if seed_spaces:
            self.observation_space.seed(self._seed)
            self.action_space.seed(self._seed)

    def make_interactive(self, port, max_players=10, realtime=True):
        """
        Enables human interaction with the environment.

        To interact with the environment add `make_interactive` to your agent's evaluation code
        and then run the `minerl.interactor.`

        For example:

        .. code-block:: python

            env = gym.make('MineRL...')

            # set the environment to allow interactive connections on port 6666
            # and slow the tick speed to 6666.
            env.make_interactive(port=6666, realtime=True)

            # reset the env
            env.reset()

            # interact as normal.
            ...


        Then while the agent is running, you can start the interactor with the following command.

        .. code-block:: bash

            python3 -m minerl.interactor 6666 # replace with the port above.


        The interactor will disconnect when the mission resets, but you can connect again with the same command.
        If an interactor is already started, it won't need to be relaunched when running the commnad a second time.


        Args:
            port (int):  The interaction port
            realtime (bool, optional): If we should slow ticking to real time.. Defaults to True.
            max_players (int): The maximum number of players

        """
        self._is_interacting = True
        self._is_real_time = realtime
        self.interact_port = port
        self.max_players = max_players

        # TODO: TEST

    ########## STEP METHOD ###########

    def _process_observation(self, pov, info) -> Dict[str, Any]:
        """
        Process observation into the proper dict space.
        """

        # info['pov'] = pov


        # Process all of the observations using handlers.
        obs_dict = {}
        monitor_dict = {}
        obs_dict['pov'] = pov
        # for h in bottom_env_spec.observables:
        #     print(h.to_string())
            # obs_dict[h.to_string()] = h.from_hero(info)


        self._last_pov = obs_dict['pov']
        self._last_obs = obs_dict

        # Voxels
        if "voxels" in info:
            monitor_dict["voxels"] = info["voxels"]

        # Mobs
        if "mobs" in info:
            monitor_dict["mobs"] = info["mobs"]

        # Health and food
        if "health" in info:
            monitor_dict["health"] = info["health"]
        
        if "food_level" in info:
            monitor_dict["food_level"] = info["food_level"]

        obs_dict["player_pos"] = {
            "x": info['x'],
            "y": info['y'],
            "z": info['z'],
            "pitch": info["pitch"],
            "yaw": info["yaw"],
        }
        # obs_dict["is_gui_open"] = info["isGuiOpen"]
        obs_dict["is_gui_open"] = False

        return obs_dict, monitor_dict


    def send(self, cmd, duration=0.5):
        print(f"发送指令: {cmd}")
        self.socket.sendall((cmd + "\n").encode('utf-8'))
        time.sleep(duration)
    
    def step(self, actions) -> Tuple[
        Dict[str, Dict[str, Any]], Dict[str, float], bool, Dict[str, Dict[str, Any]]]:
        if not self.done:
            assert STEP_OPTIONS == 0 or STEP_OPTIONS == 2

            multi_obs = {}
            multi_reward = {}
            multi_monitor = {}

            if not self.has_finished:
                instance = self.instances[0]
                os.environ['DISPLAY'] = ':'+str(instance.xvfb_port)
                if actions['forward']!=0:
                    self.send("forward 0.2", 0.2)
                    action_record = 'forward'
                elif actions['back']!=0:
                    self.send("s 0.2", 0.2)
                    action_record = 's'
                elif actions['d']!=0:
                    self.send("d 0.2", 0.2)
                    action_record = 'd'
                elif actions['a']!=0:
                    self.send("a 0.2", 0.2)
                    action_record = 'a'
                elif actions['jump']!=0:
                    self.send("jump", 0.5)
                    action_record = 'jump'
                elif actions['sneak']!=0:
                    self.send("sneak", 0.5)
                    action_record = 'sneak'
                elif actions['sprint']!=0:
                    self.send("sneak", 0.5)
                    action_record = 'sprint'
                elif actions['inventory']!=0:
                    self.send("inventory", 2.0)
                    action_record = 'inventory'
                elif actions['attack']!=0:
                    self.send("attack", 0.3)
                    action_record = 'attack'
                elif actions['use']!=0:
                    self.send("use", 1.0)
                    action_record = 'use'
                elif actions['look_left']!=0 and actions['look_down']!=0:
                    self.send("turn -30 30 0.5", 0.5) 
                    action_record = 'look_left_down'
                elif actions['look_right']!=0 and actions['look_down']!=0:
                    self.send("turn 30 30 0.5", 0.5) 
                    action_record = 'look_right_down'
                elif actions['look_right']!=0 and actions['look_up']!=0:
                    self.send("turn 30 -30 0.5", 0.5)
                    action_record = 'look_right_up'
                elif actions['look_left']!=0 and actions['look_up']!=0:
                    self.send("turn -30 -30 0.5", 0.5)
                    action_record = 'look_left_up'
                elif actions['look_right']!=0:
                    self.send("turn 30 0 0.5", 0.5) 
                    action_record = 'look_right'
                elif actions['look_left']!=0:
                    self.send("turn -30 0 0.5", 0.5) 
                    action_record = 'look_left'
                elif actions['look_up']!=0:
                    self.send("turn 0 -30 0.5", 0.5) 
                    action_record = 'look_up'
                elif actions['look_down']!=0:
                    self.send("turn 0 30 0.5", 0.5) 
                    action_record = 'look_down'
                elif actions['wait']!=0:
                    self.send("wait", 0.5)
                    action_record = 'wait'
                elif actions['init']!=0:
                    self.send("wait", 0.5)
                    action_record = 'init'
                elif actions['operate']!="":
                    self.send(actions['operate'], 0.5)
                    self.send("use", 1.0)
                    action_record = 'operate'
                else:
                    self.send("wait", 0.5)
                    action_record = 'wrong'
                # except:
                #     print(f"errorrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrr")

                self.rollout_step+=1
                img = ImageGrab.grab()
                self.img_path = os.path.join(self.output_dir, f"mc_{self.task}_{self.train_step}_{self.rollout_step}_{action_record}.jpg")
                img.save(self.img_path, format='JPEG')

                # ↓↓↓↓↓ 原来 4 行 socket 通信删掉，换成抓屏 ↓↓↓↓↓
                # obs, (w, h) = np.array(img), img.size        # bytes
                workdir = instance.working_dir
                # try:
                info_path = os.path.join(instance.working_dir, "versions/1219/socketpuppet_data/recording.csv") 
                with open(info_path, newline='', encoding='utf-8') as f:
                    *_, last_row = csv.reader(f)
                info = {"x": float(last_row[1]), "y": float(last_row[2]), "z": float(last_row[3]), "yaw": float(last_row[4]), "pitch": float(last_row[5])}
                info_reward = info

                # 初始化 reward 和 done，避免 UnboundLocalError
                reward = 0
                done = 0

                if self.reward_fn['type'] == 'navigation' and actions['init'] == 0:
                    position = [info_reward['x'], info_reward['y'], info_reward['z']]
                    gt_position = self.reward_fn['gt_position']
                    if (position[0]-gt_position[0])**2+(position[1]-gt_position[1])**2+(position[2]-gt_position[2])**2<4:
                        reward = 1
                        done = 1
                    else:
                        reward = 0
                        done = 0
                elif self.reward_fn['type'] == 'operation' and actions['init'] == 0:
                    self.send(f"get_block {self.reward_fn['gt_position'][0]} {self.reward_fn['gt_position'][1]} {self.reward_fn['gt_position'][2]}")
                    statu = self.socket.recv(4096).decode('utf-8').strip()
                    print(statu)
                    if self.reward_fn['gt_obj_status'] in statu:
                        reward = 1
                        done=1
                    else:
                        reward = 0
                        done = 0
                if done:
                    logger.info("Agent has finished")

                self.has_finished = self.has_finished or done

                # Process the observation and done state.
                out_obs, monitor = self._process_observation(self.img_path, info)
            else:
                # IF THIS PARTICULAR AGENT IS DONE THEN:
                reward = 0.0
                out_obs = self._last_obs
                done = True
                monitor = {}

            # concatenate multi-agent obs, rew, done
            multi_obs = out_obs
            multi_reward = reward
            self.done =  done
            multi_monitor = monitor

        else:
            reward = 1.0
            out_obs = self._last_obs
            done = True
            monitor = {}
            # concatenate multi-agent obs, rew, done
            multi_obs = out_obs
            multi_reward = reward
            self.done = done
            multi_monitor = monitor
        return multi_obs, multi_reward, self.done, multi_monitor
    
    def command(self, command):
        # instance = self.instances[0]
        # os.environ['DISPLAY'] = ':'+str(instance.xvfb_port)
        # env = os.environ.copy()
        self.send(command)
        
    def noop_action(self):
        """Gets the no-op action for the environment.

        In addition one can get the no-op/default action directly from the action space.

            env.action_space.noop()


        Returns:
            Any: A no-op action in the env's space.
        """
        return self.action_space.no_op()


    ########### RENDERING METHODS #########

    def render(self, mode='human'):
        if mode == 'human':
            obs = self._last_obs
            pov = obs["pov"]
            cv2.imshow("MineRL Render", pov[:, :, ::-1])
            cv2.waitKey(1)

        return self._last_pov

    ########### RESET METHODS #########

    def reset(self, reward_fn) -> Any:
        """
        Reset the environment.

        Sets-up the Env from its specification (called everytime the env is reset.)

        Returns:
            The first observation of the environment. 
        """
        self.reward_fn = reward_fn
        request_uuid = str(uuid.uuid4())

        has_warned = False

        try:

            # Get a new episode UID and produce Mission XML's for the agents 
            # without the element for the slave -> master connection (for multiagent.)

            # Start missing instances, quit episodes, and make socket connections
            self._setup_instances()

            # Episodic state variables
            self.done = False
            self.has_finished = False
            with open(os.path.join(self.working_dir, "versions/1219/socketpuppet_data/port.txt"), 'r', encoding='utf-8') as f:
                port = int(f.read().strip())
            self.socket.connect((self.ip, port))
            self.train_step = 1
            self.rollout_step = 0

            return self._peek_obs()

        finally:
            print(11111111)

    def fast_reset(self, reward_fn) -> Any:
        """
        Reset the environment.

        Sets-up the Env from its specification (called everytime the env is reset.)

        Returns:
            The first observation of the environment. 
        """
        self.reward_fn = reward_fn
        request_uuid = str(uuid.uuid4())

        has_warned = False
        self.train_step+=1
        self.rollout_step = 0
        self.done = False
        self.has_finished = False
        self.task = reward_fn['task']

        multi_obs = {}
        for instance in self.instances:
            if self.xvfb:
                self.rec[-1].stop
                outfile = os.path.join(self.output_dir, f"mc_{self.task}_{self.train_step}.mp4")
                self.rec.append(XvfbRecorder(instance.xvfb_port, outfile=outfile))
                self.rec[-1].start()
            start_time = time.time()

            os.environ['DISPLAY'] = ':'+str(instance.xvfb_port)
            # self.rec[-1].start()
            img = ImageGrab.grab()
            self.img_path = os.path.join(self.output_dir, f"mc_{self.task}_{self.train_step}_{self.rollout_step}_fastreset.jpg")
            img.save(self.img_path, format='JPEG')

            workdir = instance.working_dir
            try:
                info_path = os.path.join(instance.working_dir, "versions/1219/socketpuppet_data/recording.csv") 
                with open(info_path, newline='', encoding='utf-8') as f:
                    *_, last_row = csv.reader(f)
                info = {"x": float(last_row[1]), "y": float(last_row[2]), "z": float(last_row[3]), "yaw": float(last_row[4]), "pitch": float(last_row[5])}
            except:
                info ={"x": 0, "y": 0, "z": 0, "yaw": 0, "pitch": 0}

            self.has_finished = self.has_finished

            multi_obs, _ = self._process_observation(self.img_path, info)

        return multi_obs

    def _setup_instances(self) -> None:
        """Sets up the instances for the environment 
        """
        num_instances_to_start = 1
        num_old_instances = len(self.instances)
        instance_futures = []
        if num_instances_to_start > 0:
            with ThreadPoolExecutor(max_workers=num_instances_to_start) as tpe:
                for _ in range(num_instances_to_start):
                    instance_futures.append(tpe.submit(self._get_new_instance))
            self.instances.extend([f.result() for f in instance_futures])
            self.instances = self.instances[:1]

        # Refresh old instances every N setups
        if self._refresh_inst_every is not None and self._inst_setup_cntr % self._refresh_inst_every == 0:
            for i in reversed(range(num_old_instances)):
                self.instances[i].kill()
                self.instances[i] = self._get_new_instance(instance_id=self.instances[i].instance_id)
        self._inst_setup_cntr += 1


    def _setup_slave_master_connection_info(self,
                                            slave_xml: etree.Element,
                                            mc_server_ip: str,
                                            mc_server_port: str):
        # note that this server port is different than above client port, and will be set later
        e = etree.Element(
            NS + "MinecraftServerConnection",
            attrib={"address": str(mc_server_ip), "port": str(mc_server_port)},
        )
        slave_xml.insert(2, e)

    def _peek_obs(self):
        multi_obs = {}
        if not self.done:
            logger.debug("Peeking the clients.")
            # <<< 唯一改动：把 peek_message 变成空操作，但保留变量 >>>
            peek_message = "<Peek/>"
            for instance in self.instances:
                if self.xvfb:
                    outfile = os.path.join(self.output_dir, f"mc_{self.task}_{self.train_step}.mp4")
                    self.rec.append(XvfbRecorder(instance.xvfb_port, outfile=outfile))
                    self.rec[-1].start()
                start_time = time.time()

                os.environ['DISPLAY'] = ':'+str(instance.xvfb_port)
                # self.rec[-1].start()
                img = ImageGrab.grab()
                self.img_path = os.path.join(self.output_dir, f"mc_{self.task}_{self.train_step}_{self.rollout_step}_init.jpg")
                img.save(self.img_path, format='JPEG')


                # ↓↓↓↓↓ 原来 4 行 socket 通信删掉，换成抓屏 ↓↓↓↓↓
                workdir = instance.working_dir
                try:
                    info_path = os.path.join(instance.working_dir, "versions/1219/socketpuppet_data/recording.csv") 
                    with open(info_path, newline='', encoding='utf-8') as f:
                        *_, last_row = csv.reader(f)
                    info = {"x": float(last_row[1]), "y": float(last_row[2]), "z": float(last_row[3]), "yaw": float(last_row[4]), "pitch": float(last_row[5])}
                except:
                    info ={"x": 0, "y": 0, "z": 0, "yaw": 0, "pitch": 0}

                self.has_finished = self.has_finished

                multi_obs, _ = self._process_observation(self.img_path, info)
        return multi_obs

    #############  CLOSE METHOD ###############
    def close(self):
        """gym api close"""
        logger.debug("Closing MineRL env...")

        if self.render_open:
            cv2.destroyWindow("MineRL Render")

        if self._already_closed:
            return

        for instance in self.instances:
            if self.xvfb:
                self.rec[-1].stop()

            if instance.running:
                instance.kill()

        self._already_closed = True
    

    def fast_close(self):
        """gym api close"""
        logger.debug("Fast Closing MineRL env...")

        # for instance in self.instances:
            # self.rec[-1].stop()

    def is_closed(self):
        return self._already_closed

    ############# AUX HELPER METHODS ###########

    # TODO - make a custom Logger with this integrated (See LogHelper.java)
    def _logger_warning(self, message, *args, once=False, **kwargs):
        if once:
            # make sure we have our silenced logs table
            if not hasattr(self, "silenced_logs"):
                self.silenced_logs = set()

            # hash the stack trace
            import hashlib
            import traceback
            stack = traceback.extract_stack()
            locator = f"{stack[-2].filename}:{stack[-2].lineno}"
            key = hashlib.md5(locator.encode('utf-8')).hexdigest()

            # check if stack trace is silenced
            if key in self.silenced_logs:
                return
            self.silenced_logs.add(key)

        logger.warning(message, *args, **kwargs)

    ############# INSTANCE HELPER METHODS ##################
    # TO_MOVE == These methods should really be part of a MinecraftInstance API
    # and not apart of the env which bridges tasks & instances!
    ########################################################
    def _TO_MOVE_clean_connection(self, instance: MinecraftInstance) -> None:
        """
        Cleans the conenction with a given instance.
        """
        try:
            if instance.client_socket:
                # Try to disconnect gracefully.
                try:
                    comms.send_message(instance.client_socket, "<Disconnect/>".encode())
                except:
                    pass
                instance.client_socket.shutdown(socket.SHUT_RDWR)
                instance.client_socket.close()
        except (BrokenPipeError, OSError, socket.error):
            # There is no connection left!
            pass

            instance.client_socket = None

    def _TO_MOVE_handle_frozen_minecraft(self, instance):
        if instance.had_to_clean:
            # Connect to a new instance!!
            logger.error(
                "Connection with Minecraft client {} cleaned "
                "more than once; restarting.".format(instance))

            instance.kill()
            instance = self._get_new_instance(instance_id=self.instance.instance_id)
        else:
            instance.had_to_clean = True


    def _TO_MOVE_quit_current_episode(self, instance: MinecraftInstance) -> None:
        has_quit = False

        logger.info("Attempting to quit: {instance}".format(instance=instance))
        # while not has_quit:
        comms.send_message(instance.client_socket, "<Quit/>".encode())
        reply = comms.recv_message(instance.client_socket)
        ok, = struct.unpack('!I', reply)
        has_quit = not (ok == 0)
        # TODO: Get this to work properly

        # time.sleep(0.1) 

    def _TO_MOVE_find_ip_and_port(self, instance: MinecraftInstance, token: str) -> Tuple[str, str]:
        # calling Find on the master client to get the server port
        sock = instance.client_socket

        # try until you get something valid
        port = 0
        tries = 0
        start_time = time.time()

        logger.info("Attempting to find_ip: {instance}".format(instance=instance))
        while port == 0 and time.time() - start_time <= MAX_WAIT:
            comms.send_message(
                sock, ("<Find>" + token + "</Find>").encode()
            )
            reply = comms.recv_message(sock)
            port, = struct.unpack('!I', reply)
            tries += 1
            time.sleep(0.1)
        if port == 0:
            raise Exception("Failed to find master server port!")
        self.integratedServerPort = port  # should/can this even be cached?
        logger.warning("MineRL agent is public, connect on port {} with Minecraft 1.11".format(port))

        # go ahead and set port for all non-controller clients
        return instance.host, str(port)

    @staticmethod
    def _TO_MOVE_hello(sock):
        comms.send_message(sock, ("<MalmoEnv" + malmo_version + "/>").encode())

    def _get_new_instance(self, instance_id=None):
        """
        Gets a new instance and sets up a logger if need be. 
        """

        instance = InstanceManager.get_instance(os.getpid(), instance_id=instance_id, display_port=self.display_port)

        if InstanceManager.is_remote():
            launch_queue_logger_thread(instance, self.is_closed)

        instance.launch(replaceable=self._is_fault_tolerant, working_dir=self.working_dir)

        # Add  a cleaning flag to the instance
        instance.had_to_clean = False
        return instance

    def _get_token(self, role, ep_uid: str):
        return ep_uid + ":" + str(role) + ":" + str(0)  # resets

    def _clean_connection(self):
        pass
