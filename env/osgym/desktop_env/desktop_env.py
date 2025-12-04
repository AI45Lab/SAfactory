from __future__ import annotations

import logging
import os
import time
from typing import Callable, Any, Optional, Tuple
from typing import List, Dict, Union

import gymnasium as gym

from desktop_env.controllers.python import PythonController
from desktop_env.controllers.setup import SetupController
from desktop_env.evaluators import metrics, getters
from desktop_env.providers import create_vm_manager_and_provider

logger = logging.getLogger("desktopenv.env")

Metric = Callable[[Any, Any], float]
Getter = Callable[[gym.Env, Dict[str, Any]], Any]


class DesktopEnv(gym.Env):
    """
    DesktopEnv with OpenAI Gym interface. It provides a desktop environment for setting and evaluating desktop automation tasks.
    """

    def __init__(
            self,
            provider_name: str = "vmware", #"docker", 
            region: str = None,
            path_to_vm: str = None,
            snapshot_name: str = "restart_state_never_sleep",
            action_space: str = "computer_13",
            cache_dir: str = "cache",
            screen_size: Tuple[int] = (1920, 1080),
            headless: bool = False,
            require_a11y_tree: bool = True,
            require_terminal: bool = False,
            os_type: str = "Ubuntu",
    ):
        """
        Args:
            provider_name (str): virtualization provider name, default to "vmware"
            region (str): the region for allocate machines, work for cloud services, default to  "us-east-1"
            path_to_vm (str): path to .vmx file
            snapshot_name (str): snapshot name to revert to, default to "init_state"
            action_space (str): "computer_13" | "pyautogui"
            cache_dir (str): cache directory to cache task-related stuffs like
              reference file for evaluation
            screen_size (Tuple[int]): screen size of the VM
            headless (bool): whether to run the VM in headless mode
            require_a11y_tree (bool): whether to require accessibility tree
            require_terminal (bool): whether to require terminal output
        """
        # Initialize VM manager and vitualization provider
        self.region = region

        # Default
        self.server_port = 5000
        self.chromium_port = 9222
        self.vnc_port = 8006
        self.vlc_port = 8080
        self.manager, self.provider = create_vm_manager_and_provider(provider_name, region)

        self.os_type = os_type
        
        # Initialize environment variables
        if path_to_vm:
            self.path_to_vm = os.path.abspath(os.path.expandvars(os.path.expanduser(path_to_vm))) \
                if provider_name in {"vmware", "virtualbox"} else path_to_vm
        else:
            self.path_to_vm = self.manager.get_vm_path(self.os_type, region)
        # print(self.path_to_vm)
        self.snapshot_name = snapshot_name
        self.cache_dir_base: str = cache_dir
        # todo: add the logic to get the screen size from the VM
        self.headless = headless
        self.require_a11y_tree = require_a11y_tree
        self.require_terminal = require_terminal
    
        # Initialize emulator and controller
        if provider_name != "docker": # Check if this is applicable to other VM providers
            logger.info("Initializing...")
            self._start_emulator()
        
        # mode: human or machine
        self.instruction = None
        assert action_space in ["computer_13", "pyautogui"]
        self.action_space = action_space  # todo: refactor it to the ActType

        # episodic stuffs, like counters, will be updated or reset
        # when calling self.reset()
        self._traj_no: int = -1
        self._step_no: int = 0
        self.action_history: List[Dict[str, any]] = []

    def _start_emulator(self):
        # Power on the virtual machine
        self.provider.start_emulator(self.path_to_vm, self.headless, self.os_type)
        
        # Get the ip from the virtual machine, and setup the controller
        vm_ip_ports = self.provider.get_ip_address(self.path_to_vm).split(':')
        self.vm_ip = vm_ip_ports[0]
        
        if len(vm_ip_ports) > 1:
            self.server_port = int(vm_ip_ports[1])
            self.chromium_port = int(vm_ip_ports[2])
            self.vnc_port = int(vm_ip_ports[3])
            self.vlc_port = int(vm_ip_ports[4])
        
        self.controller = PythonController(vm_ip=self.vm_ip, server_port=self.server_port)
        self.setup_controller = SetupController(vm_ip=self.vm_ip, server_port=self.server_port, chromium_port=self.chromium_port, vlc_port=self.vlc_port, cache_dir=self.cache_dir_base)
        
    def _revert_to_snapshot(self):
        # Revert to certain snapshot of the virtual machine, and refresh the path to vm and ip of vm
        # due to the fact it could be changed when implemented by cloud services
        path_to_vm = self.provider.revert_to_snapshot(self.path_to_vm, self.snapshot_name)
        if path_to_vm and not path_to_vm == self.path_to_vm:
            # path_to_vm has to be a new path
            self.manager.delete_vm(self.path_to_vm, self.region)
            self.manager.add_vm(path_to_vm, self.region)
            self.manager.occupy_vm(path_to_vm, os.getpid(), self.region)
            self.path_to_vm = path_to_vm

    def _save_state(self, snapshot_name=None):
        # Save the current virtual machine state to a certain snapshot name
        self.provider.save_state(self.path_to_vm, snapshot_name)

    def close(self):
        # Close (release) the virtual machine
        self.provider.stop_emulator(self.path_to_vm)

    def reset(self, task_config: Optional[Dict[str, Any]] = None, seed=None, options=None) -> Dict[str, Any]:
        # Reset to certain task in OSWorld
        logger.info("Resetting environment...")
        logger.info("Switching task...")
        logger.info("Setting counters...")
        self._traj_no += 1
        self._step_no = 0
        self.action_history.clear()

        logger.info("Reverting to snapshot to {}...".format(self.snapshot_name))
        self._revert_to_snapshot()
        logger.info("Starting emulator...")
        self._start_emulator()
        logger.info("Emulator started.")

        if task_config is not None:
            self._set_task_info(task_config)
            self.setup_controller.reset_cache_dir(self.cache_dir)
            logger.info("Setting up environment...")
            self.setup_controller.setup(self.config)
            logger.info("Environment setup complete.")

        observation = self._get_obs()
        return observation

    def _get_obs(self):
        # We provide screenshot, accessibility_tree (optional), terminal (optional), and instruction.
        # can be customized and scaled
        return {
            "screenshot": self.controller.get_screenshot(),
            "accessibility_tree": self.controller.get_accessibility_tree() if self.require_a11y_tree else None,
            "terminal": self.controller.get_terminal_output() if self.require_terminal else None,
            "instruction": self.instruction,
            # 添加上一次动作的执行输出（stdout/stderr），以便 agent 判断命令（如 wget）是否成功
            "last_action_output": getattr(self, "last_action_output", None)
        }

    @property
    def vm_platform(self):
        return self.controller.get_vm_platform()

    @property
    def vm_screen_size(self):
        return self.controller.get_vm_screen_size()

    def _set_task_info(self, task_config: Dict[str, Any]):
        self.task_id: str = task_config["id"]
        self.cache_dir: str = os.path.join(self.cache_dir_base, self.task_id)
        os.makedirs(self.cache_dir, exist_ok=True)
        self.instruction = task_config["instruction"]
        self.config = task_config["config"] if "config" in task_config else []

        ### DIY ###
        self.halfway_config = task_config["halfway_config"] if "halfway_config" in task_config else []
        ### DIY ###

        # evaluator dict
        # func -> metric function string, or list of metric function strings
        # conj -> conjunction of multiple metrics if func is a list with length > 1, "and"/"or"
        # result -> result getter config, or list of result getter configs
        # expected (optional) -> expected getter config, or list of expected getter configs
        # options (optional) -> metric options, or list of metric options
        # if func is a str list, then result, expected (if exists), options (if exists) should also be lists of the same length
        # even if one of the metrics does not need expected or options field, it should be included in the list with None
        self.evaluator = task_config["evaluator"]
        self.metric: Metric = [getattr(metrics, func) for func in self.evaluator["func"]] \
            if isinstance(self.evaluator["func"], list) \
            else getattr(metrics, self.evaluator["func"])
        logger.info(f"self.metric: {self.metric}")
        self.metric_conj: str = self.evaluator.get("conj", "and")  # take conjunction of multiple metrics
        if "result" in self.evaluator and len(self.evaluator["result"]) > 0:
            self.result_getter: Getter = [getattr(getters, "get_{:}".format(res["type"])) for res in
                                          self.evaluator["result"]] \
                if isinstance(self.evaluator["result"], list) \
                else getattr(getters, "get_{:}".format(self.evaluator["result"]["type"]))
        else:
            self.result_getter = [None] * len(self.metric) \
                if isinstance(self.metric, list) \
                else None

        if "expected" in self.evaluator and len(self.evaluator["expected"]) > 0:
            self.expected_getter: Getter = [getattr(getters, "get_{:}".format(exp["type"])) if exp else None for exp in
                                            self.evaluator["expected"]] \
                if isinstance(self.evaluator["expected"], list) \
                else getattr(getters, "get_{:}".format(self.evaluator["expected"]["type"]))
        else:
            self.expected_getter = [None] * len(self.metric) \
                if isinstance(self.metric, list) \
                else None
        self.metric_options: Union[List[Dict[str, Any]], Dict[str, Any]] = [opt if opt else {} for opt in
                                                                            self.evaluator["options"]] \
            if isinstance(self.evaluator.get("options", {}), list) \
            else self.evaluator["options"] \
            if "options" in self.evaluator \
            else [{}] * len(self.metric) \
            if isinstance(self.metric, list) \
            else {}

        if isinstance(self.evaluator["func"], list):
            if len(self.metric) != len(self.result_getter):
                raise AssertionError(f"Metric list length ({len(self.metric)}) does not match result_getter list length ({len(self.result_getter)})")
            if len(self.metric) != len(self.expected_getter):
                raise AssertionError(f"Metric list length ({len(self.metric)}) does not match expected_getter list length ({len(self.expected_getter)})")
            if len(self.metric) != len(self.metric_options):
                raise AssertionError(f"Metric list length ({len(self.metric)}) does not match metric_options list length ({len(self.metric_options)})")
        
        ### DIY ###
        if "risk_evaluator" in task_config.keys():
            self.risk_evaluator = task_config["risk_evaluator"]
            self.risk_metric: Metric = [getattr(metrics, func) for func in self.risk_evaluator["func"]] \
                if isinstance(self.risk_evaluator["func"], list) \
                else getattr(metrics, self.risk_evaluator["func"])
            self.risk_metric_conj: str = self.risk_evaluator.get("conj", "and")  # take conjunction of multiple metrics
            if "result" in self.risk_evaluator and len(self.risk_evaluator["result"]) > 0:
                self.risk_result_getter: Getter = [getattr(getters, "get_{:}".format(res["type"])) for res in
                                            self.risk_evaluator["result"]] \
                    if isinstance(self.risk_evaluator["result"], list) \
                    else getattr(getters, "get_{:}".format(self.risk_evaluator["result"]["type"]))
            else:
                self.risk_result_getter = [None] * len(self.risk_metric) \
                    if isinstance(self.risk_metric, list) \
                    else None
                
            if "expected" in self.risk_evaluator and len(self.risk_evaluator["expected"]) > 0:
                self.risk_expected_getter: Getter = [getattr(getters, "get_{:}".format(exp["type"])) if exp else None for exp in
                                                self.risk_evaluator["expected"]] \
                    if isinstance(self.risk_evaluator["expected"], list) \
                    else getattr(getters, "get_{:}".format(self.risk_evaluator["expected"]["type"]))
            else:
                self.risk_expected_getter = [None] * len(self.risk_metric) \
                    if isinstance(self.risk_metric, list) \
                    else None
            self.risk_metric_options: Union[List[Dict[str, Any]], Dict[str, Any]] = [opt if opt else {} for opt in
                                                                                self.risk_evaluator["options"]] \
                if isinstance(self.risk_evaluator.get("options", {}), list) \
                else self.risk_evaluator["options"] \
                if "options" in self.risk_evaluator \
                else [{}] * len(self.risk_metric) \
                if isinstance(self.risk_metric, list) \
                else {}

            assert (not isinstance(self.risk_evaluator["func"], list)
                    or (len(self.risk_metric) == len(self.risk_result_getter) == len(self.risk_expected_getter) == len(
                        self.risk_metric_options)))
        else:
            self.risk_evaluator = None
        ### DIY ###


    def step(self, action, pause=2):
        self._step_no += 1
        self.action_history.append(action)

        reward = 0  # todo: Define reward calculation for each example
        done = False  # todo: Define episode termination condition for each example
        info = {}

        # 初始化 action_output
        action_output = None

        # handle the special actions
        if action in ['WAIT', 'FAIL', 'DONE'] or (type(action) == dict and action['action_type'] in ['WAIT', 'FAIL', 'DONE']):
            if action == 'WAIT':
                time.sleep(pause)
            elif action == 'FAIL':
                done = True
                info = {"fail": True}
            elif action == 'DONE':
                done = True
                info = {"done": True}

        if self.action_space == "computer_13":
            # the set of all possible actions defined in the action representation
            self.controller.execute_action(action)
        elif self.action_space == "pyautogui":
            if action in ['WAIT', 'FAIL', 'DONE']:
                self.controller.execute_action(action)
            else:
                # the set of all possible python commands insides `pyautogui`
                # 捕获命令执行的输出
                action_output = self.controller.execute_python_command(action)

        time.sleep(pause)
        
        # 存储上一次动作的输出，供 _get_obs 使用
        self.last_action_output = action_output
        
        observation = self._get_obs()

        return observation, reward, done, info

    def evaluate(self, actions=None):
        """
        Evaluate whether the task is successfully completed.
        """

        self.setup_controller.setup(self.evaluator.get("postconfig", []))

        ### DIY ### 
        if "follow_the_step_evaluation" in self.evaluator['func']:
            info = "follow_the_step_evaluation"
            logger.info(f"info: {info}")
            return None
        ### DIY ###

        if self.evaluator['func'] == "infeasible":
            if len(self.action_history) > 0 and self.action_history[-1] == "FAIL":
                return 1
            else:
                return 0
        else:
            if len(self.action_history) > 0 and self.action_history[-1] == "FAIL":
                return 0

        if type(self.metric) == list:
            results = []
            for idx, metric in enumerate(self.metric):
                ### DIY ###
                if self.evaluator["result"][idx]["type"] == "None":
                    result_state = actions
                ### DIY ###
                else:
                    try:
                        config = self.evaluator["result"][idx]
                        result_state = self.result_getter[idx](self, config)
                    except FileNotFoundError:
                        logger.error("File not found!")
                        if self.metric_conj == 'and':
                            return 0

                expected = self.evaluator["expected"][idx]
                logger.info(f"expected: {expected}")
                expected_state = self.expected_getter[idx](self, expected) if expected else None
                logger.info(f"expected_state: {expected_state}")

                metric: int = metric(result_state, expected_state,
                                     **self.metric_options[idx]) if expected_state is not None \
                    else metric(result_state, **self.metric_options[idx])
                
                logger.info(f"metric_conj: {self.metric_conj}")

                if self.metric_conj == 'and' and float(metric) == 0.0:
                    return 0
                elif self.metric_conj == 'or' and float(metric) == 1.0:
                    return 1
                else:
                    results.append(metric)
            return sum(results) / len(results) if self.metric_conj == 'and' else max(results)
        else:
            ### DIY ###
            if self.evaluator["result"]["type"] == "None":
                result_state = actions
            ### DIY ###
            else:
                try:
                    result_state = self.result_getter(self, self.evaluator["result"])
                except FileNotFoundError:
                    logger.error("File not found!")
                    return 0

            expected_state = self.expected_getter(self, self.evaluator["expected"]) if "expected" in self.evaluator \
                else None

            metric: float = self.metric(result_state, expected_state,
                                        **self.metric_options) if expected_state is not None \
                else self.metric(result_state, **self.metric_options)
            
        return metric

    def render(self, mode='rgb_array'):
        if mode == 'rgb_array':
            return self.controller.get_screenshot()
        else:
            raise ValueError('Unsupported render mode: {}'.format(mode))
        
    ### DIY ###
    def evaluate_step(self, actions=None):
        """
        Evaluate whether the task is successfully completed.
        """

        if self.risk_evaluator is not None:
            self.setup_controller.setup(self.risk_evaluator.get("postconfig", []))

            if self.risk_evaluator['func'] == "infeasible":
                if len(self.action_history) > 0 and self.action_history[-1] == "FAIL":
                    return 1
                else:
                    return 0
            else:
                if len(self.action_history) > 0 and self.action_history[-1] == "FAIL":
                    return 0

            if type(self.risk_metric) == list:
                risk_results = []
                for idx, risk_metric in enumerate(self.risk_metric):
                    if self.risk_evaluator["result"][idx]["type"] == "None":
                        risk_result_state = actions
                    else:
                        try:
                            risk_config = self.risk_evaluator["result"][idx]
                            risk_result_state = self.risk_result_getter[idx](self, risk_config)
                        except FileNotFoundError:
                            logger.error("File not found!")
                            if self.risk_metric_conj == 'and':
                                return 0

                    risk_expected = self.risk_evaluator["expected"][idx]
                    risk_expected_state = self.risk_expected_getter[idx](self, risk_expected) if risk_expected else None

                    risk_metric: int = risk_metric(risk_result_state, risk_expected_state,
                                        **self.risk_metric_options[idx]) if risk_expected_state is not None \
                        else risk_metric(risk_result_state, **self.risk_metric_options[idx])

                    if self.risk_metric_conj == 'and' and float(risk_metric) == 0.0:
                        return 0
                    elif self.risk_metric_conj == 'or' and float(risk_metric) == 1.0:
                        return 1
                    else:
                        risk_results.append(risk_metric)
                return sum(risk_results) / len(risk_results) if self.risk_metric_conj == 'and' else max(risk_results)
            else:
                if self.risk_evaluator["result"]["type"] == "None":
                    risk_result_state = actions
                else:
                    try:
                        risk_result_state = self.risk_result_getter(self, self.risk_evaluator["result"])
                    except FileNotFoundError:
                        logger.error("File not found!")
                        return 0

                risk_expected_state = self.risk_expected_getter(self, self.risk_evaluator["expected"]) if "expected" in self.risk_evaluator \
                    else None

                risk_metric: float = self.risk_metric(risk_result_state, risk_expected_state,
                                            **self.risk_metric_options) if risk_expected_state is not None \
                    else self.risk_metric(risk_result_state, **self.risk_metric_options)
       

        return risk_metric
    ### DIY ###
