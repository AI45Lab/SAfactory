from typing import Optional, Dict, Any, Tuple, Callable, List
from openai.types.chat import ChatCompletionMessageParam
from core.types.base import ResetOutput, StepOutput, RenderOutput, PromptOutput, TextContent, OpenAIMessage, MessageContent
from core.env.base_env import BaseEnv
from core.env.env_register import register_env
import json
import os
import sys
import re
from enum import Enum
import gymnasium as gym
import base64
import difflib
try:
    import matplotlib
    matplotlib.use('Agg')  # 使用非交互式后端
    import matplotlib.pyplot as plt
    import numpy as np
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    plt = None
    np = None


class ActionType(Enum):
    """支持的动作：add/commit/test"""
    ADD = 0
    COMMIT = 1
    RUN_TEST = 2


class Task:
    """轻量任务描述：目标文件与需求"""
    def __init__(
        self,
        id: str,
        description: str,
        target_files: List[str],
        requirements: Optional[List[str]] = None,
        base_reward: float = 100.0,
    ) -> None:
        self.id = id
        self.description = description
        self.target_files = target_files
        self.requirements = requirements or []
        self.base_reward = base_reward


class CoreGitEnvLite(gym.Env):
    """极简 Git 风格环境：内存存储文件、配合 LLM 交互"""
    
    metadata = {"render_modes": ["ansi"]}

    def __init__(
        self,
        env_id: str,
        repo_name: str,
        llm_function: Optional[Callable[[str], str]] = None,
        max_steps: int = 100,
        verification_path: Optional[str] = None,
    ) -> None:
        super().__init__()
        self.env_id = env_id
        self.repo_name = repo_name
        self.llm_function = llm_function or (lambda prompt: "")
        self.max_steps = max_steps
        self.verification_path = verification_path or "project_verification.json"

        # 仓库状态（内存）
        self.files: Dict[str, str] = {}
        self.staged: Dict[str, str] = {}
        self.commits: List[Dict[str, Any]] = []
        self.current_task: Optional[Task] = None
        self.step_count = 0

        # 观测：仓库摘要 + 任务 JSON
        self.observation_space = gym.spaces.Dict({
            "repo": gym.spaces.Text(max_length=1024*1024),
            "task": gym.spaces.Text(max_length=10000),
        })
        # 动作：(action_id, file_index)
        self.action_space = gym.spaces.MultiDiscrete([3, 256])

    def parse_llm_response(self, response_text: str) -> Tuple[int, int]:
        """解析LLM响应，提取动作和文件索引"""
        response_text = str(response_text).strip()
        action = ActionType.ADD.value  # 默认动作
        file_index = 0
        
        # 提取ACTION
        action_match = re.search(r"ACTION\s*[:\-]\s*([012])", response_text, re.IGNORECASE)
        if action_match:
            action = int(action_match.group(1))
        else:
            # 容错：查找数字0, 1, 2
            nums = re.findall(r"(?<!\d)([012])(?!\d)", response_text)
            if nums:
                action = int(nums[0])
        
        # 提取FILE_INDEX
        index_match = re.search(r"FILE_INDEX\s*[:\-]\s*(\d+)", response_text, re.IGNORECASE)
        if index_match:
            file_index = int(index_match.group(1))
        else:
            # 容错：查找数字
            all_nums = re.findall(r"\d+", response_text)
            if len(all_nums) >= 2:
                file_index = int(all_nums[1])  # 第二个数字作为file_index
        
        return action, file_index

    def get_task_prompt_text(self) -> str:
        """返回任务 JSON 字符串"""
        if not self.current_task:
            return ""
        details = {
            "description": self.current_task.description,
            "files": self.current_task.target_files,
            "requirements": self.current_task.requirements,
        }
        return json.dumps(details, ensure_ascii=False, indent=2)

    def set_task(self, task: Task) -> None:
        """设置当前任务"""
        self.current_task = task

    def _evaluate_code_quality(self, code: str, filename: str = "<string>") -> Dict[str, float]:
        """评估代码质量，返回四维评分"""
        scores = {
            "syntax_score": 0.0,  # 语法正确性 (30分)
            "structure_score": 0.0,  # 代码结构 (10分)
            "functionality_score": 0.0,  # 功能正确性 (40分) - 在ADD阶段暂时为0
            "test_score": 0.0  # 测试用例通过 (20分) - 在ADD阶段暂时为0
        }
        
        # 1. 语法正确性检查 (30分)
        try:
            compile(code, filename, 'exec')
            scores["syntax_score"] = 30.0
        except SyntaxError:
            scores["syntax_score"] = 0.0
        except Exception:
            scores["syntax_score"] = 0.0
        
        # 2. 代码结构检查 (10分)
        structure_points = 0.0
        
        # 检查是否有函数定义
        if 'def ' in code:
            structure_points += 3.0
            # 检查是否有类定义
            if 'class ' in code:
                structure_points += 2.0
            
            # 检查是否有文档字符串
            if '"""' in code or "'''" in code:
                structure_points += 2.0
            
            # 检查代码行数是否合理
            lines = len(code.splitlines())
            if 10 <= lines <= 200:
                structure_points += 2.0
            elif 5 <= lines < 10 or 200 < lines <= 500:
                structure_points += 1.0
        
        scores["structure_score"] = min(structure_points, 10.0)
        
        return scores

    def _action_add(self, file_index: int, code: Optional[str] = None) -> Dict[str, Any]:
        """生成代码并暂存"""
        if not self.current_task:
            return {"message": "no task set"}
        if file_index >= len(self.current_task.target_files):
            return {"message": "file index out of range"}

        target_file = self.current_task.target_files[file_index]
        
        # 如果提供了代码，直接使用；否则尝试从llm_function生成
        if code:
            # 代码已经由agent提供，直接使用
            generated_code = code
            print(f"DEBUG: 使用从agent响应中提取的代码，长度: {len(generated_code)}")
        elif self.llm_function and callable(self.llm_function):
            # 如果没有提供代码，尝试使用llm_function生成（向后兼容）
            task_details = self.get_task_prompt_text()
            task_json = json.loads(task_details) if task_details else {}
            
            # 构建详细的prompt（保留原有逻辑）
            file_requirements = []
            file_signature = None
            file_description = ""
            
            gitgym_instance = getattr(self, '_gitgym_instance', None)
            if gitgym_instance and hasattr(gitgym_instance, 'task_config'):
                task_config = gitgym_instance.task_config
                if task_config and isinstance(task_config, dict):
                    files_info = task_config.get("files", [])
                    if isinstance(files_info, list):
                        for file_info in files_info:
                            if isinstance(file_info, dict) and file_info.get("name") == target_file:
                                file_requirements = file_info.get("requirements", [])
                                file_signature = file_info.get("function_signature", "")
                                file_description = file_info.get("description", "")
                                break
            
            if not file_requirements:
                file_requirements = task_json.get("requirements", [])
                if file_requirements:
                    relevant_reqs = [req for req in file_requirements if target_file in str(req)]
                    if relevant_reqs:
                        file_requirements = relevant_reqs
            
            prompt = (
                "You are an expert Python engineer. Your task is to generate COMPLETE, EXECUTABLE Python code.\n\n"
                f"TARGET FILE: {target_file}\n\n"
            )
            
            if file_description:
                prompt += f"FILE DESCRIPTION:\n{file_description}\n\n"
            elif task_json.get('description'):
                prompt += f"TASK DESCRIPTION:\n{task_json.get('description')}\n\n"
            
            if file_signature:
                prompt += f"REQUIRED FUNCTION SIGNATURE:\n{file_signature}\n\n"
            
            if file_requirements:
                prompt += "REQUIREMENTS:\n"
                for i, req in enumerate(file_requirements, 1):
                    req_text = req if isinstance(req, str) else str(req)
                    prompt += f"{i}. {req_text}\n"
                prompt += "\n"
            
            prompt += (
                "CRITICAL INSTRUCTIONS:\n"
                "1. Output ONLY Python code - NO markdown, NO explanations, NO comments (except docstrings inside code)\n"
                "2. The code MUST be complete and executable\n"
                "3. Include ALL necessary function definitions\n"
                "4. The code must be ready to run - no placeholders, no TODOs\n"
                "5. If the task requires a function, implement the function with proper signature\n"
                "6. Start directly with the code (no ```python, no markdown blocks)\n"
                "7. DO NOT include any explanation text before or after the code\n"
                "8. DO NOT include example usage or test code\n\n"
                "Now generate the COMPLETE Python code for the task. Start directly with the code:\n"
            )
            
            generated_code = self.llm_function(prompt)
            print(f"DEBUG: 使用llm_function生成的代码，长度: {len(generated_code) if generated_code else 0}")
        else:
            return {"message": "no code provided and no llm_function available"}
        
        # 使用提供的代码，只做最基本的清理
        code = generated_code
        
        # 保存原始响应用于调试
        original_code = code
        print(f"\n{'='*80}")
        print(f"DEBUG: 代码生成 - 文件: {target_file}")
        print(f"原始代码长度: {len(original_code)}")
        print(f"原始代码前500字符:\n{original_code[:500]}")
        print(f"{'='*80}\n")
        
        # 只做最基本的清理：移除首尾空白和markdown标记
        code = code.strip()
        
        # 移除markdown代码块标记（只移除最外层的）
        if code.startswith('```python'):
            code = code[9:].strip()
        elif code.startswith('```'):
            code = code[3:].strip()
        if code.endswith('```'):
            code = code[:-3].strip()
        
        # 如果代码被截断（从原始长度大幅减少），使用原始代码
        if len(code) < len(original_code) * 0.5:
            print(f"警告: 代码可能被错误截断，使用原始代码")
            code = original_code.strip()
            # 只移除markdown标记
            if code.startswith('```python'):
                code = code[9:].strip()
            elif code.startswith('```'):
                code = code[3:].strip()
            if code.endswith('```'):
                code = code[:-3].strip()
        
        print(f"DEBUG: 最终保存到staged的代码长度: {len(code)}")
        print(f"DEBUG: 最终保存到staged的代码内容:\n{code[:500]}")
        print(f"{'='*80}\n")
        
        self.staged[target_file] = code
        
        # 评估代码质量
        quality_scores = self._evaluate_code_quality(code, target_file)
        
        return {
            "message": f"staged {target_file}",
            "lines": len(code.splitlines()),
            "quality_scores": quality_scores  # 包含四维评分
        }

    def _action_commit(self) -> Dict[str, Any]:
        """提交暂存到仓库"""
        for path, content in self.staged.items():
            self.files[path] = content
        self.staged.clear()
        commit = {"id": len(self.commits) + 1, "files": list(self.files.keys())}
        self.commits.append(commit)
        return {"message": f"commit {commit['id']}", "commit": commit}

    def _action_run_test(self) -> Tuple[float, Dict[str, Any]]:
        """基于 verification 执行用例，失败则重写相关文件并重试"""
        verification = self._read_verification()
        tests = verification.get("tests", []) if isinstance(verification, dict) else []
        retries = 2  # 最大重试轮次
        last_eval: Dict[str, Any] = {"cases": [], "total": 0, "passed": 0, "failed": 0, "score": 0}

        for attempt in range(retries + 1):
            details: Dict[str, Any] = {"cases": []}
            restored: Dict[str, Any] = {}
            try:
                restored = self._register_inmemory_modules()

                total = 0
                passed = 0
                failed_idxs: List[int] = []
                for i, item in enumerate(tests, 1):
                    code = item.get("code") if isinstance(item, dict) else item
                    description = item.get("description", f"Test case {i}") if isinstance(item, dict) else f"Test case {i}"
                    category = item.get("category", "Unknown") if isinstance(item, dict) else "Unknown"
                    if not isinstance(code, str):
                        code = str(code) if code is not None else ""
                    total += 1
                    try:
                        local_ns: Dict[str, Any] = {}
                        exec(code, {}, local_ns)
                        passed += 1
                        details["cases"].append({
                            "index": i,
                            "status": "passed",
                            "description": description,
                            "category": category,
                            "test_code": code[:200]  # 保存测试代码
                        })
                    except AssertionError as e:
                        # 尝试提取实际值和期望值
                        error_msg = str(e)
                        actual_value = None
                        expected_value = None
                        
                        # 尝试从测试代码中解析assert语句
                        # 格式通常是: "assert function_call(...) == expected_value"
                        import re
                        # 匹配 assert 语句，提取函数调用和期望值
                        assert_pattern = r'assert\s+(.+?)\s+==\s+(.+)'
                        match = re.search(assert_pattern, code)
                        if match:
                            function_call_expr = match.group(1).strip()
                            expected_expr = match.group(2).strip()
                            
                            # 尝试执行函数调用获取实际值
                            try:
                                # 先执行import语句，确保函数可用
                                import_lines = []
                                exec_lines = []
                                for line in code.split('\n'):
                                    if line.strip().startswith('from ') or line.strip().startswith('import '):
                                        import_lines.append(line)
                                    elif line.strip().startswith('assert'):
                                        exec_lines.append(line)
                                
                                # 执行import语句
                                import_code = '\n'.join(import_lines)
                                if import_code:
                                    exec(import_code, {}, local_ns)
                                
                                # 执行函数调用获取实际值
                                actual_value = eval(function_call_expr, {}, local_ns)
                            except Exception as eval_err:
                                # 如果eval失败，尝试更简单的方法
                                pass
                            
                            # 尝试解析期望值
                            try:
                                expected_value = eval(expected_expr)
                            except:
                                # 如果无法eval，可能是字符串或其他，直接使用
                                expected_value = expected_expr
                        
                        details["cases"].append({
                            "index": i,
                            "status": "failed",
                            "error": error_msg,
                            "description": description,
                            "category": category,
                            "actual_value": actual_value,
                            "expected_value": expected_value,
                            "test_code": code[:200]  # 保存测试代码的前200字符
                        })
                        failed_idxs.append(i)
                    except Exception as e:
                        # 其他类型的错误（如ImportError, SyntaxError等）
                        error_msg = str(e)
                        details["cases"].append({
                            "index": i,
                            "status": "failed",
                            "error": error_msg,
                            "description": description,
                            "category": category,
                            "test_code": code[:200]  # 保存测试代码的前200字符
                        })
                        failed_idxs.append(i)

                score = round((passed / total) * 100.0, 1) if total > 0 else 0.0
                details.update({"total": total, "passed": passed, "failed": total - passed, "score": score})
                last_eval = details

                # 全部通过则结束
                if details["failed"] == 0:
                    return score, {"message": f"score: {score}", "evaluation": details}

                # 最后一轮不再修复
                if attempt == retries:
                    return score, {"message": f"score: {score}", "evaluation": details}

            finally:
                # 恢复 sys.modules
                for mod_name, old_val in restored.items():
                    if old_val is None:
                        if mod_name in sys.modules:
                            del sys.modules[mod_name]
                    else:
                        sys.modules[mod_name] = old_val

            # 基于失败用例推断需要重写的文件并重生代码
            targets = self._infer_targets_from_tests(tests, failed_idxs)
            if not targets:
                continue
            self._regenerate_targets(targets)

        # 返回最后一次评估
        score = float(last_eval.get("score", 0))
        return score, {"message": f"score: {score}", "evaluation": last_eval}

    def _build_observation(self) -> Dict[str, Any]:
        """构造观测：仓库摘要 + 任务 JSON"""
        repo_summary = {
            "files": list(self.files.keys()),
            "staged": list(self.staged.keys()),
            "commits": len(self.commits),
            "steps": self.step_count,
        }
        return {
            "repo": json.dumps(repo_summary, ensure_ascii=False),
            "task": self.get_task_prompt_text(),
        }

    def _read_verification(self) -> Dict[str, Any]:
        """读取 verification JSON"""
        path = self.verification_path
        try:
            # 如果是相对路径且文件不存在，尝试相对于代码文件查找
            if not os.path.isabs(path) and not os.path.exists(path):
                _current_dir = os.path.dirname(os.path.abspath(__file__))
                alt_path = os.path.join(_current_dir, path)
                if os.path.exists(alt_path):
                    path = alt_path
            
            if os.path.exists(path):
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                return data if isinstance(data, dict) else {"tests": []}
        except Exception:
            return {"tests": []}
        return {"tests": []}

    def _register_inmemory_modules(self) -> Dict[str, Any]:
        """把内存文件注册为模块，便于 'from xxx import ...'"""
        import types
        restored: Dict[str, Any] = {}
        for filename, content in self.files.items():
            if not filename.endswith('.py'):
                continue
            mod_name = filename[:-3].replace('/', '.').replace('\\', '.')
            old = sys.modules.get(mod_name)
            restored[mod_name] = old if mod_name in sys.modules else None
            module = types.ModuleType(mod_name)
            module.__dict__["__file__"] = filename
            try:
                code_obj = compile(content, filename, 'exec')
                exec(code_obj, module.__dict__)
                sys.modules[mod_name] = module
            except Exception as e:
                # 保留一个空模块，避免导入失败导致所有用例中断
                sys.modules[mod_name] = module
        return restored

    def _infer_targets_from_tests(self, tests: List[Any], failed_idxs: List[int]) -> List[str]:
        """从失败用例推断目标文件"""
        targets: set = set()
        for idx in failed_idxs:
            item = tests[idx-1]
            code = item.get("code") if isinstance(item, dict) else item
            if not isinstance(code, str):
                continue
            for pat in [r"from\s+([a-zA-Z_][\w]*)\s+import", r"import\s+([a-zA-Z_][\w]*)"]:
                m = re.search(pat, code)
                if m:
                    targets.add(f"{m.group(1)}.py")
        # 兜底：若为空，尝试常见文件名
        if not targets:
            for name in ["calculator.py", "scientific.py", "main.py"]:
                if name in self.files:
                    targets.add(name)
        return list(targets)

    def _regenerate_targets(self, target_files: List[str]) -> None:
        """重生目标文件并提交"""
        for tf in target_files:
            prompt = (
                "只输出可运行 Python 代码（不要 Markdown）。\n"
                f"实现文件: {tf}\n"
                f"任务上下文: {self.get_task_prompt_text()}\n"
            )
            try:
                code = self.llm_function(prompt)
                if isinstance(code, str) and code.strip():
                    self.files[tf] = code
            except Exception:
                continue
        # 记录一次提交
        commit = {"id": len(self.commits) + 1, "files": list(self.files.keys()), "auto_fix": True}
        self.commits.append(commit)


@register_env("core_git_env")
class GitGym(BaseEnv):
    """Git环境封装，遵循统一的BaseEnv接口"""
    
    def __init__(
        self,
        repo_name: str = "project_repo",
        llm_function: Optional[Callable[[str], str]] = None,
        max_steps: int = 100,
        task_config_path: Optional[str] = None,
        task_config: Optional[Dict[str, Any]] = None,
        verification_path: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        
        # 从kwargs中移除这些参数，避免传递给CoreGitEnvLite时重复
        kwargs.pop('env_id', None)
        kwargs.pop('env_name', None)
        kwargs.pop('dataset', None)
        
        self.repo_name = repo_name
        
        # 获取当前文件所在目录，用于查找默认配置文件
        _current_dir = os.path.dirname(os.path.abspath(__file__))
        
        # 设置默认路径：优先使用传入的路径，否则使用相对于代码文件的路径
        if verification_path:
            # 如果路径是相对路径且文件不存在，尝试相对于代码文件查找
            if not os.path.isabs(verification_path) and not os.path.exists(verification_path):
                alt_path = os.path.join(_current_dir, verification_path)
                if os.path.exists(alt_path):
                    self.verification_path = alt_path
                else:
                    self.verification_path = verification_path
            else:
                self.verification_path = verification_path
        else:
            # 默认查找当前目录下的project_verification.json
            default_verification = os.path.join(_current_dir, "project_verification.json")
            # 如果当前目录没有，尝试项目根目录
            if not os.path.exists(default_verification):
                default_verification = "project_verification.json"  # 回退到当前工作目录
            self.verification_path = default_verification

        # 加载任务配置
        if task_config_path:
            # 如果路径是相对路径且文件不存在，尝试相对于代码文件查找
            if not os.path.isabs(task_config_path) and not os.path.exists(task_config_path):
                _current_dir = os.path.dirname(os.path.abspath(__file__))
                alt_path = os.path.join(_current_dir, task_config_path)
                if os.path.exists(alt_path):
                    task_config_path = alt_path
            task_config = self._load_task_config_from_file(task_config_path)
        
        # 创建核心环境实例
        # 使用传入的 env_id（如果有）或默认值，但确保不会从 kwargs 中重复传入
        core_env_id = self.env_id if self.env_id else "git_env"
        # 确保 kwargs 中不包含 env_id，避免重复参数
        core_kwargs = {k: v for k, v in kwargs.items() if k != 'env_id'}
        
        self.core_env = CoreGitEnvLite(
            env_id=core_env_id,
            repo_name=repo_name,
            llm_function=llm_function or (lambda prompt: ""),
            max_steps=max_steps,
            verification_path=self.verification_path,
            **core_kwargs,
        )
        
        # 让CoreGitEnvLite能够访问GitGym的task_config（用于构建更好的prompt）
        self.core_env._gitgym_instance = self

        self.action_space = self.core_env.action_space
        self.observation_space = gym.spaces.Dict({
            "repo": gym.spaces.Text(max_length=1024*1024),
            "task": gym.spaces.Text(max_length=10000),
        })

        # 如果提供了任务配置，创建并设置任务
        if task_config:
            task = self._create_task_from_config(task_config)
            self.core_env.set_task(task)
            # 保存原始任务配置（包含examples、edge_cases等详细信息）
            self.task_config = task_config

        # 初始化状态变量
        self._truncated = False
        self.step_count = 0
        self.last_action_result: Optional[Dict[str, Any]] = None  # 保存最近一次操作的结果
        self.last_test_result: Optional[Dict[str, Any]] = None  # 保存最近一次测试的结果
        self.task_config: Optional[Dict[str, Any]] = None  # 保存原始任务配置（包含详细示例）
        self.action_history: List[Dict[str, Any]] = []  # 保存操作历史，用于可视化
        self.total_reward: float = 0.0  # 累计reward
        self.best_test_score: float = 0.0  # 最佳测试分数
        self.file_progress: Dict[str, bool] = {}  # 文件完成状态
        self.file_history: List[Dict[str, str]] = []  # 文件内容历史，用于代码对比
        # 保存前一次的代码和测试结果，用于改进代码生成
        self.previous_code: Dict[str, str] = {}  # 每个文件的前一次生成的代码
        self.previous_test_result: Optional[Dict[str, Any]] = None  # 前一次的测试结果
        self._sync_state()

    def reset(self, seed: Optional[int] = None) -> ResetOutput:
        """重置环境到初始状态"""
        self.core_env.staged.clear()
        self.core_env.step_count = 0
        self.step_count = 0
        self._truncated = False
        self.last_action_result = None
        self.last_test_result = None
        self.action_history = []
        self.total_reward = 0.0
        self.best_test_score = 0.0
        self.file_progress = {}
        self.file_history = []  # 重置文件历史
        self.previous_code = {}  # 重置前一次代码
        self.previous_test_result = None  # 重置前一次测试结果
        if self.current_task:
            for file in self.current_task.target_files:
                self.file_progress[file] = False
        self._sync_state()
        # 记录初始文件快照（空状态）
        self._record_file_snapshot()
        
        observation = self._get_observation()
        info = self._get_info()
        
        return ResetOutput(observation=observation, info=info)

    def step(self, action: str) -> StepOutput:
        """执行一步环境交互（支持LLM文本响应）"""
        super().step(action=action)
        self.messages.append(
            {"role": "assistant", "content": action}
        )
        
        # 解析LLM响应，提取动作和文件索引
        act, idx = self.core_env.parse_llm_response(action)
        
        # 如果是ADD动作，尝试从action中提取代码
        extracted_code = None
        if act == ActionType.ADD.value:
            extracted_code = self._extract_code_from_response(action)
        
        # 在执行动作前，先同步状态以确保记录的prev状态是最新的
        self._sync_state()
        
        self.core_env.step_count += 1
        self.step_count = self.core_env.step_count

        info: Dict[str, Any] = {}
        reward = 0.0
        terminated = False
        truncated = False
        
        # 记录操作前的状态（此时状态已同步，确保是最新的）
        prev_files = set(self.files.keys())
        prev_staged = set(self.staged_files.keys())
        prev_test_score = self.best_test_score

        if act == ActionType.ADD.value:
            # 传递提取的代码给_action_add
            info = self.core_env._action_add(idx, code=extracted_code)
            # 成功生成代码给予奖励，基于四维评估体系
            if "message" in info and "staged" in info.get("message", ""):
                target_file = self.current_task.target_files[idx] if (self.current_task and idx < len(self.current_task.target_files)) else None
                
                # 同步状态以获取最新代码
                self._sync_state()
                
                # 保存生成的代码到本地文件（用于调试和查看）
                if target_file and target_file in self.staged_files:
                    current_code = self.staged_files[target_file]
                    self._save_generated_code_to_file(target_file, current_code)
                    # 保存当前代码为"前一次代码"（用于下次生成时参考）
                    # 注意：这里保存的是当前代码，下次生成时会作为"前一次代码"使用
                    # 但我们需要在生成新代码之前保存旧代码，所以这个逻辑会在_build_text_prompt中处理
                
                # 基于四维评估体系计算reward
                # 1. 语法正确性 (30分) -> reward权重
                # 2. 代码结构 (10分) -> reward权重
                # 3. 功能正确性 (40分) -> 在RUN_TEST时评估
                # 4. 测试用例通过 (20分) -> 在RUN_TEST时评估
                
                quality_scores = info.get("quality_scores", {})
                syntax_score = quality_scores.get("syntax_score", 0.0)  # 0-30
                structure_score = quality_scores.get("structure_score", 0.0)  # 0-10
                
                # 将分数转换为reward（总分40分转换为reward）
                # 语法正确性权重: 30/40 * reward_multiplier
                # 代码结构权重: 10/40 * reward_multiplier
                reward_multiplier = 0.1  # 基础倍数，使reward在合理范围
                
                syntax_reward = (syntax_score / 40.0) * 30.0 * reward_multiplier  # 最高 7.5
                structure_reward = (structure_score / 40.0) * 10.0 * reward_multiplier  # 最高 2.5
                
                base_reward = syntax_reward + structure_reward
                
                # 如果这是新文件的首次生成，给予额外奖励
                if target_file and target_file not in prev_files:
                    base_reward += 0.2  # 新文件奖励
                    if target_file in self.file_progress:
                        self.file_progress[target_file] = True
                
                reward = base_reward
            else:
                reward = 0.0
            
            # 在生成新代码之前，保存当前代码为"前一次代码"（如果存在）
            # 注意：这里在ADD操作之后执行，所以保存的是生成新代码之前的旧代码
            if target_file:
                # 先检查staged中是否有旧代码（在生成新代码之前）
                # 但由于我们在ADD之后才执行，此时staged已经是新代码了
                # 所以我们需要在ADD之前保存，这个逻辑在_build_text_prompt中处理
                # 这里只做备份，确保不会丢失
                if target_file in self.files and target_file not in self.previous_code:
                    # 如果files中有代码且还没有保存为previous_code，保存它
                    self.previous_code[target_file] = self.files[target_file]
            
            self.last_action_result = {"action": "ADD", "file_index": idx, **info}
            self.last_test_result = None  # ADD操作不产生测试结果
        elif act == ActionType.COMMIT.value:
            info = self.core_env._action_commit()
            # 成功提交给予中等奖励，考虑提交的文件数量
            if "message" in info and "commit" in info.get("message", ""):
                base_reward = 0.5
                
                # 如果提交了新文件（从staged变成files），给予额外奖励
                newly_committed = len(prev_staged) - len(prev_files & prev_staged)
                if newly_committed > 0:
                    base_reward += newly_committed * 0.2  # 每个新文件+0.2
                
                reward = base_reward
            self.last_action_result = {"action": "COMMIT", **info}
            self.last_test_result = None  # COMMIT操作不产生测试结果
        elif act == ActionType.RUN_TEST.value:
            test_score, info = self.core_env._action_run_test()
            
            # 基于四维评估体系计算reward
            # 在RUN_TEST时评估功能正确性和测试用例通过率
            
            # 计算功能正确性分数 (40分)
            # 通过测试用例的通过率来评估功能正确性
            functionality_score = 0.0
            test_passed_score = 0.0
            
            if "evaluation" in info:
                eval_data = info["evaluation"]
                passed = eval_data.get("passed", 0)
                total = eval_data.get("total", 0)
                
                if total > 0:
                    # 3. 功能正确性 (40分): 基于测试用例的执行结果
                    # 如果能执行且通过测试，说明功能正确
                    pass_rate = passed / total
                    functionality_score = pass_rate * 40.0
                    
                    # 4. 测试用例通过 (20分): 测试用例通过率
                    test_passed_score = pass_rate * 20.0
            
            # 将所有分数转换为reward（总分100分）
            # 语法和结构在ADD时已评估，这里主要评估功能和测试
            reward_multiplier = 0.15  # 测试阶段的reward倍数
            
            functionality_reward = (functionality_score / 100.0) * 40.0 * reward_multiplier  # 最高 6.0
            test_reward = (test_passed_score / 100.0) * 20.0 * reward_multiplier  # 最高 3.0
            
            base_reward = functionality_reward + test_reward
            
            # 测试改进奖励：如果分数提升了，给予额外奖励
            improvement_bonus = 0.0
            if test_score > self.best_test_score:
                improvement = test_score - self.best_test_score
                improvement_bonus = improvement / 50.0  # 每10分改进 = 0.2 reward
                self.best_test_score = test_score
            
            # 如果全部通过测试，给予额外大奖励
            completion_bonus = 0.0
            if "evaluation" in info:
                eval_data = info["evaluation"]
                passed = eval_data.get("passed", 0)
                total = eval_data.get("total", 0)
                if total > 0 and passed == total:
                    # 全部通过，给予完成奖励
                    completion_bonus = 10.0  # 完成奖励
                    
                    # 额外考虑：如果早期完成（在1/3步数内），给予效率奖励
                    progress_ratio = self.step_count / self.core_env.max_steps
                    if progress_ratio < 0.33:
                        completion_bonus += 5.0  # 效率奖励
                    elif progress_ratio < 0.66:
                        completion_bonus += 2.0
            
            reward = base_reward + improvement_bonus + completion_bonus
            
            self.last_action_result = {"action": "RUN_TEST", **info}
            # 保存测试结果供下次prompt使用
            if "evaluation" in info:
                # 保存为last_test_result（当前测试结果）
                self.last_test_result = info["evaluation"]
                # 保存为previous_test_result（用于下次代码生成时参考）
                # 只有当有新的测试结果时才更新（避免覆盖）
                if self.previous_test_result is None or test_score > self.previous_test_result.get("score", 0):
                    self.previous_test_result = info["evaluation"].copy()
        else:
            info = {"message": f"unknown action: {act}"}
            self.last_action_result = {"action": "UNKNOWN", **info}
            reward = -0.1  # 未知操作给予小惩罚

        # 更新累计reward
        self.total_reward += reward
        
        # 在执行动作后、记录历史前，先同步状态确保files_after记录的是最新状态
        self._sync_state()
        
        # 记录操作历史（此时状态已同步，files_after是最新的）
        action_record = {
            "step": self.step_count,
            "action": ["ADD", "COMMIT", "RUN_TEST"][act] if act < 3 else "UNKNOWN",
            "reward": reward,
            "total_reward": self.total_reward,
            "files_before": list(prev_files),
            "staged_before": list(prev_staged),
            "files_after": list(self.files.keys()),  # 已同步后的最新状态
            "staged_after": list(self.staged_files.keys()),  # 已同步后的最新状态
            "info": info.copy()
        }
        self.action_history.append(action_record)
        
        # 记录文件快照（在动作执行后）
        self._record_file_snapshot()

        if self.step_count >= self.core_env.max_steps:
            truncated = True
            self.done = True
        
        self._truncated = truncated
        # 最终同步状态，确保返回时状态是最新的
        self._sync_state()
        
        observation = self._get_observation()
        updated_info = self._get_info()
        updated_info.update(info)
        updated_info["reward"] = reward  # 在info中也保存reward便于查看
        updated_info["total_reward"] = self.total_reward
        updated_info["best_test_score"] = self.best_test_score
        
        return StepOutput(
            observation=observation,
            reward=reward,
            terminated=terminated,
            truncated=self._truncated,
            info=updated_info
        )

    def get_task_prompt(self) -> List[ChatCompletionMessageParam]:
        """获取任务提示信息- 每次调用都会获取最新的环境状态"""
        # 确保状态是最新的
        self._sync_state()
        
        # 获取最新的观察和任务信息
        repo_obs = self.core_env._build_observation()
        task_prompt_text = self.core_env.get_task_prompt_text()
        
        # 构建包含最新状态的文本提示
        prompt_text = self._build_text_prompt(repo_obs, task_prompt_text)
        
        # 构建system消息
        system_prompt = "You are a Python developer working in a Git-like environment. Your task is to complete software development tasks by generating code, committing changes, and running tests."
        
        self.messages = [
            {"role": "system", "content": system_prompt}
        ]
        
        self.messages.append(
            {
                "role": "user",
                "content": prompt_text
            }
        )
        
        return self.messages

    def render(self) -> RenderOutput:
        """渲染仓库快照并返回RenderOutput（纯文本格式）- 包含每一步的状态变化"""
        self._sync_state()
        
        # 计算文件完成进度
        completed_files = []
        pending_files = []
        if self.current_task:
            for file in self.current_task.target_files:
                if file in self.files:
                    completed_files.append(file)
                else:
                    pending_files.append(file)
        
        # 构建详细的快照信息
        snapshot = {
            "step": self.step_count,
            "max_steps": self.core_env.max_steps,
            "progress": f"{self.step_count}/{self.core_env.max_steps} ({self.step_count/self.core_env.max_steps*100:.1f}%)",
            "repo_name": self.repo_name,
            "timestamp": self.step_count,  # 使用step_count作为时间戳
            
            # 当前状态
            "current_state": {
                "files_in_repo": list(self.files.keys()),
                "files_count": len(self.files),
                "staged_files": list(self.staged_files.keys()),
                "staged_count": len(self.staged_files),
                "commits": len(self.commits),
            },
            
            # 任务进度
            "task_progress": {
                "completed_files": completed_files,
                "pending_files": pending_files,
                "completion_rate": len(completed_files) / len(self.current_task.target_files) * 100 if (self.current_task and self.current_task.target_files) else 0.0
            },
            
            # 最近一次操作
            "last_action": self.last_action_result if self.last_action_result else None,
            
            # 最近一次测试结果
            "last_test": {
                "score": self.best_test_score,
                "details": self.last_test_result if self.last_test_result else None
            } if self.last_test_result else None,
            
            # Reward信息
            "rewards": {
                "step_reward": self.action_history[-1]["reward"] if self.action_history else 0.0,
                "total_reward": self.total_reward,
                "best_test_score": self.best_test_score,
            },
            
            # 操作历史（最近5步）
            "recent_history": self.action_history[-5:] if len(self.action_history) > 5 else self.action_history,
            
            # 提交详情（最近3次）
            "recent_commits": self.commits[-3:] if len(self.commits) > 3 else self.commits,
            
            # 文件内容摘要（最近修改的文件）
            "file_summaries": {},
            
            # 代码变更对比（自上次步骤）
            "code_changes": {}
        }
        
        # 获取自上次步骤以来的代码变更
        file_changes = self._get_file_changes_since_last_step()
        if file_changes:
            # 格式化代码变更信息，便于可视化
            formatted_changes = {}
            for file_path, diff_info in file_changes.items():
                formatted_changes[file_path] = {
                    "change_type": diff_info.get("change_type", "modified"),
                    "status": diff_info.get("status", "committed"),  # staged 或 committed
                    "statistics": diff_info.get("statistics", {}),
                    "diff_text": diff_info.get("diff_text", ""),  # 完整的diff文本
                    "diff_preview": self._format_diff_preview(diff_info),  # 格式化的预览
                    "modified_blocks": diff_info.get("modified_blocks", [])
                }
            snapshot["code_changes"] = formatted_changes
        
        # 添加最近操作涉及的文件内容摘要
        if self.action_history:
            last_action = self.action_history[-1]
            changed_files = set(last_action.get("files_after", [])) - set(last_action.get("files_before", []))
            changed_files.update(set(last_action.get("staged_after", [])) - set(last_action.get("staged_before", [])))
            
            for file_path in list(changed_files)[:3]:  # 最多显示3个文件
                if file_path in self.files:
                    content = self.files[file_path]
                    lines = content.split('\n')
                    summary = {
                        "line_count": len(lines),
                        "preview": '\n'.join(lines[:5]),  # 前5行预览
                    }
                    snapshot["file_summaries"][file_path] = summary
                elif file_path in self.staged_files:
                    content = self.staged_files[file_path]
                    lines = content.split('\n')
                    summary = {
                        "line_count": len(lines),
                        "preview": '\n'.join(lines[:5]),
                        "status": "staged"  # 标记为暂存状态
                    }
                    snapshot["file_summaries"][file_path] = summary
        
        # 生成基于reward的图表
        image_base64 = self._generate_reward_chart()
        
        # 返回包含base64图片的RenderOutput
        return RenderOutput(
            step=self.step_count,
            image_base64=image_base64
        )
    
    def _generate_reward_chart(self) -> str:
        """生成包含reward和代码diff的图表（两个子图）"""
        if not HAS_MATPLOTLIB or plt is None:
            # 如果没有matplotlib，创建一个简单的占位图片
            return self._create_placeholder_image()
        
        try:
            # 创建包含两个子图的图表
            fig = plt.figure(figsize=(16, 10))
            gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
            
            # 准备reward数据
            steps = [record["step"] for record in self.action_history]
            rewards = [record["reward"] for record in self.action_history]
            cumulative_rewards = [record["total_reward"] for record in self.action_history]
            
            # 左列：Reward图表（占据左侧两行）
            if not steps:
                # 如果没有历史数据，显示初始状态
                ax_reward = fig.add_subplot(gs[:, 0])
                ax_reward.text(0.5, 0.5, f'Step: {self.step_count}\nTotal Reward: {self.total_reward:.2f}\nBest Test Score: {self.best_test_score:.1f}%', 
                       ha='center', va='center', fontsize=14, 
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
                ax_reward.set_xlim(0, 1)
                ax_reward.set_ylim(0, 1)
                ax_reward.axis('off')
                ax_reward.set_title('Reward History', fontsize=14, fontweight='bold')
            else:
                # 上图：每步reward
                ax1 = fig.add_subplot(gs[0, 0])
                ax1.plot(steps, rewards, marker='o', linestyle='-', color='blue', markersize=4)
                ax1.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
                ax1.set_xlabel('Step', fontsize=10)
                ax1.set_ylabel('Step Reward', fontsize=10)
                ax1.set_title(f'Step Rewards (Step {self.step_count}/{self.core_env.max_steps})', fontsize=11)
                ax1.grid(True, alpha=0.3)
                
                # 下图：累计reward
                ax2 = fig.add_subplot(gs[1, 0])
                ax2.plot(steps, cumulative_rewards, marker='s', linestyle='-', color='green', markersize=4)
                ax2.set_xlabel('Step', fontsize=10)
                ax2.set_ylabel('Cumulative Reward', fontsize=10)
                ax2.set_title(f'Cumulative: {self.total_reward:.2f} | Best Score: {self.best_test_score:.1f}%', fontsize=11)
                ax2.grid(True, alpha=0.3)
            
            # 右列：代码Diff可视化（占据右侧两行）
            ax_diff = fig.add_subplot(gs[:, 1])
            self._render_code_diff_on_axis(ax_diff)
            
            # 添加总标题
            info_text = f"Files: {len(self.files)} | Staged: {len(self.staged_files)} | Commits: {len(self.commits)}"
            fig.suptitle(f'GitGym Environment - {self.repo_name}\n{info_text}', fontsize=14, fontweight='bold')
            
            # 转换为base64
            import io
            buffer = io.BytesIO()
            plt.savefig(buffer, format='PNG', dpi=100, bbox_inches='tight')
            buffer.seek(0)
            img_bytes = buffer.getvalue()
            plt.close(fig)
            
            img_base64 = base64.b64encode(img_bytes).decode('utf-8')
            return img_base64
            
        except Exception as e:
            # 如果生成图表失败，返回占位图片
            print(f"警告: 生成图表失败: {e}，使用占位图片")
            if plt:
                plt.close('all')
            return self._create_placeholder_image()
    
    def _render_code_diff_on_axis(self, ax) -> None:
        """在matplotlib轴上渲染代码diff（彩色显示）"""
        # 获取最新的代码变更
        file_changes = self._get_file_changes_since_last_step()
        
        # 确定要显示的文件（优先显示有变更的文件，否则显示当前文件）
        display_file = None
        old_content = ""
        new_content = ""
        
        if file_changes:
            # 有变更：显示第一个变更的文件
            display_file = list(file_changes.keys())[0]
            diff_info = file_changes[display_file]
            # 从历史中获取前后内容
            if len(self.file_history) >= 2:
                prev_snapshot = self.file_history[-2]
                curr_snapshot = self.file_history[-1]
                if isinstance(prev_snapshot, dict) and isinstance(curr_snapshot, dict):
                    prev_files = prev_snapshot.get("files", {})
                    curr_files = curr_snapshot.get("files", {})
                    prev_staged = prev_snapshot.get("staged", {})
                    curr_staged = curr_snapshot.get("staged", {})
                    
                    # 优先从staged获取（因为staged是最近的变化）
                    if isinstance(prev_staged, dict) and isinstance(curr_staged, dict):
                        if display_file in prev_staged and display_file in curr_staged:
                            old_content = prev_staged.get(display_file, "") if isinstance(prev_staged, dict) else ""
                            new_content = curr_staged.get(display_file, "") if isinstance(curr_staged, dict) else ""
                    if not new_content and isinstance(prev_files, dict) and isinstance(curr_files, dict):
                        if display_file in prev_files and display_file in curr_files:
                            old_content = prev_files.get(display_file, "") if isinstance(prev_files, dict) else ""
                            new_content = curr_files.get(display_file, "") if isinstance(curr_files, dict) else ""
        else:
            # 没有变更：显示当前文件内容（如果有文件）
            if self.files:
                display_file = list(self.files.keys())[0]
                new_content = self.files[display_file]
                old_content = ""  # 显示为新建
            elif self.staged_files:
                display_file = list(self.staged_files.keys())[0]
                new_content = self.staged_files[display_file]
                old_content = ""  # 显示为新建
        
        if not display_file:
            # 完全没有文件，显示提示
            ax.text(0.5, 0.5, 'No files yet.\nWaiting for code generation...', 
                   ha='center', va='center', fontsize=12,
                   bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.5))
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.axis('off')
            ax.set_title('Code Changes (Diff)', fontsize=14, fontweight='bold')
            return
        
        # 生成diff（使用n=0最小上下文，只显示改变的行）
        if file_changes and display_file in file_changes:
            diff_info = file_changes[display_file]
            raw_diff_lines = diff_info.get("diff_lines", [])
        else:
            # 没有变更，但显示文件内容（作为对比）
            raw_diff_lines = list(difflib.unified_diff(
                old_content.splitlines(keepends=True) if old_content else [],
                new_content.splitlines(keepends=True) if new_content else [],
                fromfile=f"{display_file} (old)" if old_content else f"{display_file} (empty)",
                tofile=f"{display_file} (current)",
                lineterm='',
                n=0  # 不使用上下文，只显示改变的行
            ))
        
        # 过滤diff_lines：只保留真正改变的行（+或-）和必要的元数据行
        diff_lines = []
        for line in raw_diff_lines:
            stripped = line.strip()
            # 保留文件头（---和+++）
            if line.startswith('---') or line.startswith('+++'):
                diff_lines.append(line)
            # 保留@@行（但可以简化显示）
            elif line.startswith('@@'):
                # 只在有实际改变时显示@@行，作为分隔符
                diff_lines.append(line)
            # 只保留真正改变的行（+或-）
            elif line.startswith('+') or line.startswith('-'):
                diff_lines.append(line)
            # 忽略所有其他行（上下文行，即没有+或-的行）
        
        if not diff_lines and new_content:
            # 如果没有diff但有新内容，显示新文件的内容（标记为全部新增）
            new_lines = new_content.splitlines(keepends=True)
            diff_lines = [f"--- {display_file} (empty)\n", f"+++ {display_file} (new)\n"]
            diff_lines.append(f"@@ -0,0 +1,{len(new_lines)} @@\n")
            # 确保每行都以+开头
            for line in new_lines:
                if line.endswith('\n'):
                    diff_lines.append(f"+{line.rstrip()}\n")
                else:
                    diff_lines.append(f"+{line}\n")
        
        if not diff_lines:
            # 如果还是没有diff lines，显示文件基本信息
            ax.text(0.5, 0.5, f'File: {display_file}\nNo changes to display.', 
                   ha='center', va='center', fontsize=12,
                   bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.axis('off')
            ax.set_title('Code Changes (Diff)', fontsize=14, fontweight='bold')
            return
        
        # 准备显示diff文本
        y_position = 0.95
        line_height = 0.03
        max_lines = min(30, len(diff_lines))  # 最多显示30行
        
        # 设置标题
        ax.text(0.5, 0.98, f'Code Diff: {display_file}', 
               ha='center', va='top', fontsize=11, fontweight='bold',
               transform=ax.transAxes)
        
        # 显示统计信息（只统计真正的改变行）
        added_count = sum(1 for line in diff_lines if line.startswith('+') and not line.startswith('+++'))
        removed_count = sum(1 for line in diff_lines if line.startswith('-') and not line.startswith('---'))
        
        if file_changes and display_file in file_changes:
            stats = file_changes[display_file].get("statistics", {})
            # 使用实际统计，如果没有则使用计算的
            added_count = stats.get('added_lines', added_count)
            removed_count = stats.get('removed_lines', removed_count)
        
        stats_text = f"+{added_count} -{removed_count}"
        ax.text(0.02, 0.95, stats_text, 
               ha='left', va='top', fontsize=9,
               bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.3),
               transform=ax.transAxes)
        
        # 渲染diff行（使用颜色，高亮显示变化）
        for i, line in enumerate(diff_lines[:max_lines]):
            if y_position < 0.05:
                break
                
            # 根据行类型选择颜色和背景
            line_stripped = line.rstrip('\n\r')
            if line.startswith('+++') or line.startswith('---'):
                color = 'gray'
                fontweight = 'normal'
                bg_color = None
            elif line.startswith('@@'):
                color = 'blue'
                fontweight = 'normal'  # @@行不需要加粗
                bg_color = 'lightblue'
            elif line.startswith('+'):
                color = 'green'
                fontweight = 'bold'  # 新增行加粗高亮
                bg_color = 'lightgreen'
            elif line.startswith('-'):
                color = 'red'
                fontweight = 'bold'  # 删除行加粗高亮
                bg_color = 'mistyrose'
            else:
                # 不应该有其他类型的行（已过滤）
                continue
            
            # 截断过长的行
            display_line = line_stripped
            if len(display_line) > 70:
                display_line = display_line[:67] + '...'
            
            # 如果有背景色，添加背景框
            if bg_color:
                ax.text(0.02, y_position, display_line,
                       ha='left', va='top', fontsize=7,
                       color=color, fontweight=fontweight,
                       family='monospace',
                       bbox=dict(boxstyle='round,pad=0.2', facecolor=bg_color, alpha=0.3, edgecolor=color, linewidth=0.5),
                       transform=ax.transAxes)
            else:
                ax.text(0.02, y_position, display_line,
                       ha='left', va='top', fontsize=7,
                       color=color, fontweight=fontweight,
                       family='monospace',
                       transform=ax.transAxes)
            y_position -= line_height
        
        if len(diff_lines) > max_lines:
            ax.text(0.5, y_position, f'... ({len(diff_lines) - max_lines} more lines)',
                   ha='center', va='top', fontsize=8, style='italic',
                   transform=ax.transAxes)
        
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
        ax.set_title('Code Changes (Diff)', fontsize=14, fontweight='bold', pad=20)
    
    def _save_generated_code_to_file(self, file_path: str, code: str) -> None:
        """保存生成的代码到本地文件（用于调试和查看）"""
        try:
            # 创建保存目录（使用env_id作为子目录，如果有的话）
            base_dir = os.path.join(os.getcwd(), "env/gitgym/generated_code")
            if self.env_id:
                save_dir = os.path.join(base_dir, f"env_{self.env_id}")
            else:
                save_dir = os.path.join(base_dir, self.repo_name)
            os.makedirs(save_dir, exist_ok=True)
            
            # 保存文件
            file_name = file_path.replace('/', '_').replace('\\', '_')
            timestamp = self.step_count
            save_path = os.path.join(save_dir, f"{file_name}_step{timestamp:04d}.py")
            
            with open(save_path, 'w', encoding='utf-8') as f:
                f.write(f"# Generated at step {timestamp}\n")
                f.write(f"# File: {file_path}\n")
                f.write(f"# Repo: {self.repo_name}\n")
                f.write("# " + "="*70 + "\n\n")
                f.write(code)
            
            # 同时保存一个latest版本（最新的）
            latest_path = os.path.join(save_dir, f"{file_name}_latest.py")
            with open(latest_path, 'w', encoding='utf-8') as f:
                f.write(f"# Latest version at step {timestamp}\n")
                f.write(f"# File: {file_path}\n")
                f.write("# " + "="*70 + "\n\n")
                f.write(code)
        except Exception as e:
            print(f"保存代码文件失败: {e}")
    
    def _create_placeholder_image(self) -> str:
        """创建一个简单的占位图片（1x1像素的白色PNG）"""
        # 创建一个最小的PNG图片（1x1白色像素）
        minimal_png = base64.b64decode(
            'iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=='
        )
        return base64.b64encode(minimal_png).decode('utf-8')

    def close(self) -> None:
        """关闭环境，释放资源"""
        super().close()

    def set_task(self, task: Task) -> None:
        """设置当前任务"""
        self.core_env.set_task(task)
        self._sync_state()

    def set_llm_function(self, llm_function: Callable[[str], str]) -> None:
        """设置LLM函数（用于生成代码）"""
        self.core_env.llm_function = llm_function
    
    def _extract_code_from_response(self, response: str) -> Optional[str]:
        """从agent响应中提取代码（如果包含代码块）"""
        if not response:
            return None
        
        print(f"DEBUG: 尝试从响应中提取代码，响应长度: {len(response)}")
        print(f"DEBUG: 响应前500字符:\n{response[:500]}")
        
        # 方法1: 尝试提取 CODE: 标记后的代码（最优先）
        # 改进正则：匹配CODE:后的所有内容，直到遇到下一个ACTION/FILE_INDEX/EXPLANATION标记或响应结束
        # 使用非贪婪匹配，但确保能匹配到完整的代码块
        code_section_pattern = r'CODE\s*:\s*\n(.*?)(?=\n\s*(?:ACTION|FILE_INDEX|EXPLANATION)\s*:|$)'
        code_match = re.search(code_section_pattern, response, re.DOTALL | re.IGNORECASE)
        if code_match:
            extracted = code_match.group(1).strip()
            # 移除可能的markdown标记
            if extracted.startswith('```python'):
                extracted = extracted[9:].strip()
            elif extracted.startswith('```'):
                extracted = extracted[3:].strip()
            if extracted.endswith('```'):
                extracted = extracted[:-3].strip()
            
            # 检查代码是否完整（至少应该有一个完整的函数定义）
            if len(extracted) > 10:
                # 检查代码是否被截断（以不完整的语句结尾）
                lines = extracted.split('\n')
                last_line = lines[-1].strip() if lines else ""
                # 如果最后一行是不完整的（比如"for i in r"），可能是被截断了
                if last_line and not last_line.endswith((':', ')', ']', '}', '"', "'")) and len(last_line) < 20:
                    # 可能是截断，尝试从响应末尾获取更多内容
                    code_start = response.find('CODE:')
                    if code_start >= 0:
                        # 从CODE:开始到响应结束的所有内容
                        potential_code = response[code_start + 5:].strip()
                        # 移除开头的换行
                        if potential_code.startswith('\n'):
                            potential_code = potential_code[1:]
                        # 移除markdown标记
                        if potential_code.startswith('```python'):
                            potential_code = potential_code[9:].strip()
                        elif potential_code.startswith('```'):
                            potential_code = potential_code[3:].strip()
                        if potential_code.endswith('```'):
                            potential_code = potential_code[:-3].strip()
                        # 如果新提取的内容更长，使用它
                        if len(potential_code) > len(extracted):
                            extracted = potential_code
                            print(f"DEBUG: 检测到代码可能被截断，使用完整提取，长度: {len(extracted)}")
                
                print(f"DEBUG: 从 CODE: 标记后提取代码，长度: {len(extracted)}")
                print(f"DEBUG: 代码最后50字符: {repr(extracted[-50:])}")
                return extracted
        
        # 方法2: 尝试提取markdown代码块
        code_block_pattern = r'```(?:python)?\s*\n(.*?)\n```'
        matches = re.findall(code_block_pattern, response, re.DOTALL)
        if matches:
            extracted = matches[0].strip()
            if len(extracted) > 10:
                print(f"DEBUG: 从markdown代码块中提取代码，长度: {len(extracted)}")
                return extracted
        
        # 方法3: 尝试提取代码块（没有标记的）
        # 查找def或class开头的代码段
        first_def = response.find('def ')
        first_class = response.find('class ')
        
        if first_def >= 0 or first_class >= 0:
            start_idx = min([i for i in [first_def, first_class] if i >= 0] or [len(response)])
            
            # 检查前面是否有 "CODE:" 标记，如果没有，可能是解释文本
            before_code = response[:start_idx].strip()
            if 'CODE:' in before_code.upper() or len(before_code) < 50:
                # 提取从第一个def/class到响应结束的代码
                code = response[start_idx:].strip()
                
                # 移除可能的后续解释文本
                lines = code.split('\n')
                code_lines = []
                for line in lines:
                    stripped = line.strip()
                    # 如果遇到明显的解释性文本，停止
                    if stripped.lower().startswith(('note:', 'explanation:', 'this code', 'the code', 'example:', 'usage:')):
                        break
                    # 如果遇到空行后跟大段文本（可能是解释），停止
                    if code_lines and code_lines[-1].strip() == '' and len(stripped) > 50 and not any(kw in stripped for kw in ['def ', 'class ', 'import ', 'from ', '=', 'return ']):
                        break
                    code_lines.append(line)
                
                result = '\n'.join(code_lines).strip()
                if len(result) > 10:  # 确保有实际内容
                    print(f"DEBUG: 从响应中提取代码（def/class开头），长度: {len(result)}")
                    return result
        
        print(f"DEBUG: 未能从响应中提取代码")
        return None

    def _get_observation(self) -> Dict[str, Any]:
        """获取当前观测"""
        return self.core_env._build_observation()

    def _get_info(self) -> Dict[str, Any]:
        """获取当前信息字典"""
        return {
            "step_count": self.step_count,
            "max_steps": self.core_env.max_steps,
            "truncated": self._truncated,
            "commits_count": len(self.commits),
            "files_count": len(self.files),
            "staged_count": len(self.staged_files),
            "has_task": self.current_task is not None
        }

    def _sync_state(self) -> None:
        """同步核心状态到外部可见状态"""
        self.commits = self.core_env.commits
        self.files = self.core_env.files
        self.staged_files = getattr(self.core_env, "staged", {})
        self.current_task = self.core_env.current_task
        self.step_count = self.core_env.step_count

    def _record_file_snapshot(self) -> None:
        """记录当前步骤的文件内容快照"""
        snapshot: Dict[str, Any] = {
            "step": self.step_count,
            "files": {k: v for k, v in self.files.items()},  # 深拷贝当前文件内容
            "staged": {k: v for k, v in self.staged_files.items()}  # 深拷贝暂存区内容
        }
        self.file_history.append(snapshot)

    def _generate_file_diff(
        self, 
        file_path: str, 
        old_content: str, 
        new_content: str,
        context_lines: int = 3
    ) -> Dict[str, Any]:
        """生成文件差异对比（求同存异：先找共同行，再算新增和删除）
        
        使用SequenceMatcher找到最长公共子序列（LCS），然后基于此确定新增和删除的行
        
        Args:
            file_path: 文件路径
            old_content: 旧内容
            new_content: 新内容
            context_lines: 上下文行数（已废弃，保留兼容性）
            
        Returns:
            包含差异信息的字典
        """
        # 使用splitlines()不带keepends，确保行匹配正确
        old_lines = old_content.splitlines() if old_content else []
        new_lines = new_content.splitlines() if new_content else []
        
        # 调试信息
        print(f"\nDEBUG: _generate_file_diff for {file_path}")
        print(f"  Old lines count: {len(old_lines)}, New lines count: {len(new_lines)}")
        
        # 使用SequenceMatcher找到最长公共子序列（LCS）
        # 使用autojunk=False确保精确匹配
        matcher = difflib.SequenceMatcher(None, old_lines, new_lines, autojunk=False)
        
        # 获取匹配块（这些是共同的行）
        matching_blocks = matcher.get_matching_blocks()
        print(f"  Matching blocks (LCS): {len(matching_blocks)}")
        
        # 构建映射：哪些行是匹配的（共同的）
        old_matched = set()
        new_matched = set()
        for i, j, n in matching_blocks:
            # i是old_lines中的起始位置，j是new_lines中的起始位置，n是匹配的长度
            for k in range(n):
                old_matched.add(i + k)
                new_matched.add(j + k)
        
        # 基于匹配的行，确定新增和删除的行
        # 删除的行：old_lines中存在但不在匹配中的行
        # 新增的行：new_lines中存在但不在匹配中的行
        diff = []
        diff.append(f"--- {file_path} (old)\n")
        diff.append(f"+++ {file_path} (new)\n")
        
        # 统计变更信息
        added_lines = 0
        removed_lines = 0
        modified_blocks = []
        
        # 使用opcodes来按顺序输出diff，但基于LCS（共同行）来识别
        opcodes = matcher.get_opcodes()
        
        for tag, i1, i2, j1, j2 in opcodes:
            if tag == 'equal':
                # 共同的行（LCS），不显示在diff中
                continue
            elif tag == 'delete':
                # 删除的行（old中有但不在LCS中）
                removed_block_lines = []
                for line in old_lines[i1:i2]:
                    diff.append(f"-{line}\n")
                    removed_lines += 1
                    removed_block_lines.append(line)
                
                if removed_block_lines:
                    modified_blocks.append({
                        "type": "removed",
                        "lines": removed_block_lines
                    })
            elif tag == 'insert':
                # 新增的行（new中有但不在LCS中）
                added_block_lines = []
                for line in new_lines[j1:j2]:
                    diff.append(f"+{line}\n")
                    added_lines += 1
                    added_block_lines.append(line)
                
                if added_block_lines:
                    modified_blocks.append({
                        "type": "added",
                        "lines": added_block_lines
                    })
            elif tag == 'replace':
                # 替换：先删除旧行，再添加新行
                # 删除旧行（不在LCS中）
                removed_block_lines = []
                for line in old_lines[i1:i2]:
                    diff.append(f"-{line}\n")
                    removed_lines += 1
                    removed_block_lines.append(line)
                
                if removed_block_lines:
                    modified_blocks.append({
                        "type": "removed",
                        "lines": removed_block_lines
                    })
                
                # 添加新行（不在LCS中）
                added_block_lines = []
                for line in new_lines[j1:j2]:
                    diff.append(f"+{line}\n")
                    added_lines += 1
                    added_block_lines.append(line)
                
                if added_block_lines:
                    modified_blocks.append({
                        "type": "added",
                        "lines": added_block_lines
                    })
        
        print(f"  Final diff: +{added_lines} -{removed_lines} lines")
        print(f"  Total diff lines: {len(diff)}")
        
        return {
            "file_path": file_path,
            "diff_text": ''.join(diff),  # 完整的diff文本
            "diff_lines": diff,  # diff行列表（只包含真正改变的行）
            "statistics": {
                "added_lines": added_lines,
                "removed_lines": removed_lines,
                "net_change": added_lines - removed_lines,
                "modified_blocks_count": len(modified_blocks)
            },
            "modified_blocks": modified_blocks,
            "change_type": self._determine_change_type(old_content, new_content)
        }
    
    def _determine_change_type(self, old_content: str, new_content: str) -> str:
        """确定变更类型"""
        if not old_content:
            return "created"
        elif not new_content:
            return "deleted"
        elif old_content != new_content:
            return "modified"
        else:
            return "unchanged"
    
    def _get_file_changes_since_last_step(self) -> Dict[str, Dict[str, Any]]:
        """获取自上次步骤以来的文件变更"""
        if len(self.file_history) < 2:
            return {}
        
        prev_snapshot = self.file_history[-2]
        curr_snapshot = self.file_history[-1]
        
        changes = {}
        
        # 确保快照是字典类型
        if not isinstance(prev_snapshot, dict) or not isinstance(curr_snapshot, dict):
            return {}
        
        prev_files = prev_snapshot.get("files", {})
        curr_files = curr_snapshot.get("files", {})
        prev_staged = prev_snapshot.get("staged", {})
        curr_staged = curr_snapshot.get("staged", {})
        
        # 检查仓库文件的变化
        if isinstance(prev_files, dict) and isinstance(curr_files, dict):
            all_files = set(prev_files.keys()) | set(curr_files.keys())
            
            for file_path in all_files:
                old_content = prev_files.get(file_path, "") if isinstance(prev_files, dict) else ""
                new_content = curr_files.get(file_path, "") if isinstance(curr_files, dict) else ""
                
                if old_content != new_content:
                    changes[file_path] = self._generate_file_diff(file_path, old_content, new_content)
                    changes[file_path]["status"] = "committed"
        
        # 检查暂存区文件的变化
        if isinstance(prev_staged, dict) and isinstance(curr_staged, dict):
            all_staged = set(prev_staged.keys()) | set(curr_staged.keys())
            
            for file_path in all_staged:
                old_content = prev_staged.get(file_path, "") if isinstance(prev_staged, dict) else ""
                new_content = curr_staged.get(file_path, "") if isinstance(curr_staged, dict) else ""
                
                # 如果文件在暂存区，标记为staged
                if old_content != new_content:
                    print(f"DEBUG: Staged file change detected: {file_path}")
                    print(f"  Old content length: {len(old_content)}, New content length: {len(new_content)}")
                    diff_info = self._generate_file_diff(file_path, old_content, new_content)
                    diff_info["status"] = "staged"
                    changes[file_path] = diff_info
        
        return changes
    
    def _format_diff_preview(self, diff_info: Dict[str, Any], max_lines: int = 20) -> str:
        """格式化差异预览，用于显示"""
        diff_lines = diff_info.get("diff_lines", [])
        if not diff_lines:
            return ""
        
        # 限制显示行数
        preview_lines = diff_lines[:max_lines]
        preview_text = ''.join(preview_lines)
        
        if len(diff_lines) > max_lines:
            preview_text += f"\n... ({len(diff_lines) - max_lines} more lines)"
        
        return preview_text
    
    def _get_file_changes_between_steps(self, step_from: int, step_to: int) -> Dict[str, Dict[str, Any]]:
        """获取两个步骤之间的文件变更（用于历史对比）"""
        if step_from < 0 or step_to >= len(self.file_history) or step_from >= step_to:
            return {}
        
        from_snapshot = self.file_history[step_from]
        to_snapshot = self.file_history[step_to]
        
        changes = {}
        
        # 确保快照是字典类型
        if not isinstance(from_snapshot, dict) or not isinstance(to_snapshot, dict):
            return {}
        
        from_files = from_snapshot.get("files", {})
        to_files = to_snapshot.get("files", {})
        
        # 检查所有文件的变化
        if isinstance(from_files, dict) and isinstance(to_files, dict):
            all_files = set(from_files.keys()) | set(to_files.keys())
            
            for file_path in all_files:
                old_content = from_files.get(file_path, "") if isinstance(from_files, dict) else ""
                new_content = to_files.get(file_path, "") if isinstance(to_files, dict) else ""
                
                if old_content != new_content:
                    changes[file_path] = self._generate_file_diff(file_path, old_content, new_content)
                    changes[file_path]["status"] = "committed"
        
        return changes

    def _load_task_config_from_file(self, config_path: str) -> Dict[str, Any]:
        """从JSON文件加载任务配置"""
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"任务配置文件不存在：{config_path}")
        
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        
        return config

    def _create_task_from_config(self, config: Dict[str, Any]) -> Task:
        """从配置字典创建Task对象"""
        # 支持两种配置格式：
        # 1. 单个任务配置：{"id": "...", "description": "...", "target_files": [...], "requirements": [...]}
        # 2. 项目配置：{"project_name": "...", "files": [{"name": "...", "description": "...", ...}]}
        
        if "id" in config:
            # 单个任务配置
            return Task(
                id=config.get("id", "task_1"),
                description=config.get("description", ""),
                target_files=config.get("target_files", []),
                requirements=config.get("requirements", []),
                base_reward=config.get("base_reward", 100.0)
            )
        elif "files" in config:
            # 项目配置格式：从project_requirements.json加载
            target_files = [f["name"] for f in config.get("files", [])]
            all_requirements = []
            for f in config.get("files", []):
                file_reqs = f.get("requirements", [])
                for req in file_reqs:
                    all_requirements.append(f"{f['name']}: {req}")
            
            return Task(
                id=config.get("project_name", "project_task"),
                description=config.get("description", "完成项目开发任务"),
                target_files=target_files,
                requirements=all_requirements,
                base_reward=100.0
            )
        else:
            raise ValueError(f"不支持的任务配置格式：{config}")

    def _build_text_prompt(self, repo_obs: Dict[str, Any], task_prompt: str) -> str:
        """构建包含完整信息的文本提示（供LLM使用）- 包含最新的环境状态"""
        repo_summary = json.loads(repo_obs["repo"]) if isinstance(repo_obs["repo"], str) else repo_obs["repo"]
        
        prompt_lines = [
            "You are a Python developer working in a Git-like environment.",
            f"Repository: {self.repo_name}",
            f"Current Step: {self.step_count} / {self.core_env.max_steps}",
            "",
            "=== Current Repository Status ===",
            f"Files in repository: {len(repo_summary.get('files', []))}",
            f"Staged files: {len(repo_summary.get('staged', []))}",
            f"Commits: {repo_summary.get('commits', 0)}",
            "",
        ]
        
        # 添加已存在文件的详细信息
        existing_files = repo_summary.get('files', [])
        if existing_files:
            prompt_lines.append("=== Files in Repository ===")
            for file_path in existing_files:
                if file_path in self.files:
                    content = self.files[file_path]
                    lines = content.split('\n')
                    preview = '\n'.join(lines[:10])  # 显示前10行
                    if len(lines) > 10:
                        preview += f"\n... (total {len(lines)} lines)"
                    prompt_lines.append(f"File: {file_path}")
                    prompt_lines.append(f"Content preview:")
                    prompt_lines.append(preview)
                    prompt_lines.append("")
        
        # 添加暂存区文件的详细信息
        staged_files = repo_summary.get('staged', [])
        if staged_files:
            prompt_lines.append("=== Staged Files (not yet committed) ===")
            for file_path in staged_files:
                if file_path in self.staged_files:
                    content = self.staged_files[file_path]
                    lines = content.split('\n')
                    preview = '\n'.join(lines[:10])  # 显示前10行
                    if len(lines) > 10:
                        preview += f"\n... (total {len(lines)} lines)"
                    prompt_lines.append(f"File: {file_path}")
                    prompt_lines.append(f"Content preview:")
                    prompt_lines.append(preview)
                    prompt_lines.append("")
        else:
            prompt_lines.append("=== Staged Files ===")
            prompt_lines.append("No files currently staged.")
            prompt_lines.append("")
        
        # 添加最近一次操作的结果
        if self.last_action_result:
            prompt_lines.append("=== Last Action Result ===")
            action = self.last_action_result.get("action", "UNKNOWN")
            message = self.last_action_result.get("message", "")
            prompt_lines.append(f"Action: {action}")
            prompt_lines.append(f"Result: {message}")
            if "lines" in self.last_action_result:
                prompt_lines.append(f"Generated lines: {self.last_action_result['lines']}")
            prompt_lines.append("")
        
        # 添加最近一次测试的结果（如果有）
        if self.last_test_result:
            prompt_lines.append("=== Last Test Results ===")
            total = self.last_test_result.get("total", 0)
            passed = self.last_test_result.get("passed", 0)
            failed = self.last_test_result.get("failed", 0)
            score = self.last_test_result.get("score", 0.0)
            prompt_lines.append(f"Test Score: {score}% ({passed}/{total} passed, {failed} failed)")
            
            # 显示失败的测试用例详情（包含实际值和期望值）
            cases = self.last_test_result.get("cases", [])
            failed_cases = [c for c in cases if c.get("status") == "failed"]
            if failed_cases:
                prompt_lines.append("")
                prompt_lines.append("=== Failed Test Cases (需要修复) ===")
                # 显示所有失败的用例（最多10个）
                for case in failed_cases[:10]:
                    idx = case.get("index", "?")
                    description = case.get("description", f"Test case {idx}")
                    category = case.get("category", "Unknown")
                    error = case.get("error", "Unknown error")
                    test_code = case.get("test_code", "")
                    
                    prompt_lines.append(f"")
                    prompt_lines.append(f"[Test {idx}] {description} ({category})")
                    
                    # 显示测试代码
                    if test_code:
                        prompt_lines.append(f"  Test code: {test_code}")
                    
                    # 显示实际值和期望值（如果有）
                    actual_value = case.get("actual_value")
                    expected_value = case.get("expected_value")
                    
                    if actual_value is not None and expected_value is not None:
                        prompt_lines.append(f"  Expected output: {expected_value}")
                        prompt_lines.append(f"  Actual output: {actual_value}")
                        prompt_lines.append(f"  Error: The actual output does not match the expected output.")
                    else:
                        # 无法提取值，显示错误信息
                        prompt_lines.append(f"  Error: {error}")
                    
                    prompt_lines.append("")
                
                if len(failed_cases) > 10:
                    prompt_lines.append(f"  ... and {len(failed_cases) - 10} more failed test cases")
                    prompt_lines.append("")
                
                prompt_lines.append("IMPORTANT: Please analyze the failed test cases above and fix your code accordingly.")
                prompt_lines.append("Focus on:")
                prompt_lines.append("  1. Understanding why the actual output differs from the expected output")
                prompt_lines.append("  2. Checking edge cases and boundary conditions")
                prompt_lines.append("  3. Verifying the logic of your algorithm")
                prompt_lines.append("")
            else:
                prompt_lines.append("All tests passed! ✓")
            prompt_lines.append("")
        
        # 如果任务配置中包含项目题目信息，优先显示
        project_info = None
        if self.task_config:
            project_info = self.task_config.get("project_problem")
        
        if project_info:
            prompt_lines.append("=== Project Problem Info ===")
            prompt_lines.append(f"Problem ID: {project_info.get('problem_id', 'N/A')}")
            prompt_lines.append(f"Title: {project_info.get('title', 'N/A')}")
            prompt_lines.append(f"Difficulty: {project_info.get('difficulty', 'N/A')}")
            prompt_lines.append(f"Time Limit: {project_info.get('time_limit', 'N/A')}")
            prompt_lines.append(f"Memory Limit: {project_info.get('memory_limit', 'N/A')}")
            prompt_lines.append("")
            
            # 显示问题描述
            if self.task_config and "problem_statement" in self.task_config:
                ps = self.task_config["problem_statement"]
                prompt_lines.append("=== Problem Statement ===")
                prompt_lines.append(ps.get("description", ""))
                prompt_lines.append("")
                prompt_lines.append("Input Format:")
                prompt_lines.append(ps.get("input_format", ""))
                prompt_lines.append("")
                prompt_lines.append("Output Format:")
                prompt_lines.append(ps.get("output_format", ""))
                prompt_lines.append("")
                
                # 注意：Sample Input/Output 不在这里显示，而是从测试结果中提取
            
            # 显示算法提示（重要：这些hints可以帮助你理解问题和实现思路）
            if self.task_config and "algorithm_hints" in self.task_config:
                hints = self.task_config["algorithm_hints"]
                if hints:
                    prompt_lines.append("=== Algorithm Hints (重要提示) ===")
                    for i, hint in enumerate(hints, 1):
                        # 如果hint本身已经包含编号，直接显示；否则添加编号
                        if hint.strip().startswith(str(i)) or hint.strip().startswith(f"{i}."):
                            prompt_lines.append(f"  {hint}")
                        else:
                            prompt_lines.append(f"  {i}. {hint}")
                    prompt_lines.append("")
            
            # 显示调试提示（如果有）
            if self.task_config and "common_debug_tips" in self.task_config:
                tips = self.task_config["common_debug_tips"]
                if tips:
                    prompt_lines.append("=== Debug Tips (调试提示) ===")
                    for i, tip in enumerate(tips, 1):
                        prompt_lines.append(f"  {i}. {tip}")
                prompt_lines.append("")
            
            # 显示约束条件
            if self.task_config and "constraints" in self.task_config:
                constraints = self.task_config["constraints"]
                prompt_lines.append("=== Constraints ===")
                prompt_lines.append(f"Time Complexity: {constraints.get('time_complexity', 'N/A')}")
                prompt_lines.append(f"Space Complexity: {constraints.get('space_complexity', 'N/A')}")
                if "edge_cases" in constraints:
                    prompt_lines.append("Edge Cases to Consider:")
                    for ec in constraints["edge_cases"]:
                        prompt_lines.append(f"  - {ec}")
                prompt_lines.append("")
        
        # 添加任务信息
        prompt_lines.append("=== Task ===")
        prompt_lines.append(task_prompt)
        prompt_lines.append("")
        
        # 如果任务配置中包含详细示例和边界情况，添加到prompt中
        if self.task_config and "files" in self.task_config:
            files_config = self.task_config.get("files", [])
            for file_config in files_config:
                file_name = file_config.get("name", "")
                
                # 添加使用示例
                examples = file_config.get("examples", [])
                if examples:
                    prompt_lines.append(f"=== Usage Examples for {file_name} ===")
                    for i, example in enumerate(examples, 1):
                        desc = example.get("description", f"Example {i}")
                        code = example.get("code", "")
                        prompt_lines.append(f"\n{desc}:")
                        prompt_lines.append(code)
                    prompt_lines.append("")
                
                # 添加边界情况
                edge_cases = file_config.get("edge_cases", [])
                if edge_cases:
                    prompt_lines.append(f"=== Edge Cases for {file_name} ===")
                    for i, edge_case in enumerate(edge_cases, 1):
                        case = edge_case.get("case", "")
                        desc = edge_case.get("description", "")
                        expected = edge_case.get("expected_behavior", "")
                        example_code = edge_case.get("example", "")
                        prompt_lines.append(f"\n{i}. {case}:")
                        prompt_lines.append(f"   Description: {desc}")
                        if expected:
                            prompt_lines.append(f"   Expected: {expected}")
                        if example_code:
                            prompt_lines.append(f"   Example: {example_code}")
                    prompt_lines.append("")
                
                # 添加常见问题和解决方案
                common_issues = file_config.get("common_issues", [])
                if common_issues:
                    prompt_lines.append(f"=== Common Issues and Solutions for {file_name} ===")
                    for i, issue in enumerate(common_issues, 1):
                        issue_name = issue.get("issue", "")
                        issue_desc = issue.get("description", "")
                        solution = issue.get("solution", "")
                        prompt_lines.append(f"\n{i}. {issue_name}:")
                        prompt_lines.append(f"   Problem: {issue_desc}")
                        if solution:
                            prompt_lines.append(f"   Solution: {solution}")
                    prompt_lines.append("")
                
                # 添加测试场景说明
                test_scenarios = file_config.get("test_scenarios", [])
                if test_scenarios:
                    prompt_lines.append(f"=== Test Scenarios for {file_name} ===")
                    for scenario in test_scenarios:
                        scenario_name = scenario.get("scenario", "")
                        scenario_desc = scenario.get("description", "")
                        tests = scenario.get("tests", [])
                        prompt_lines.append(f"\n{scenario_name}: {scenario_desc}")
                        if tests:
                            for test in tests:
                                prompt_lines.append(f"  - {test}")
                    prompt_lines.append("")
        
        # 添加前一次代码和测试结果（用于改进代码生成）
        # 检查是否有前一次的代码需要改进
        if self.current_task:
            target_files = self.current_task.target_files
            for target_file in target_files:
                # 如果文件在staged或files中，说明已经有代码了
                current_code = None
                if target_file in self.staged_files:
                    current_code = self.staged_files[target_file]
                elif target_file in self.files:
                    current_code = self.files[target_file]
                
                # 如果有当前代码，保存为"前一次代码"（用于下次生成时参考）
                if current_code and target_file not in self.previous_code:
                    self.previous_code[target_file] = current_code
                
                # 如果有前一次代码和测试结果，添加到prompt中
                if target_file in self.previous_code and self.previous_test_result:
                    prompt_lines.append(f"=== Previous Code for {target_file} (需要改进) ===")
                    prev_code = self.previous_code[target_file]
                    code_lines = prev_code.split('\n')
                    # 显示前一次代码（完整显示，不超过100行）
                    if len(code_lines) <= 100:
                        prompt_lines.append("Previous code:")
                        for line in code_lines:
                            prompt_lines.append(line)
                    else:
                        prompt_lines.append("Previous code (first 50 lines):")
                        for line in code_lines[:50]:
                            prompt_lines.append(line)
                        prompt_lines.append(f"... (total {len(code_lines)} lines, showing first 50)")
                    prompt_lines.append("")
                    
                    # 显示前一次测试结果
                    prev_total = self.previous_test_result.get("total", 0)
                    prev_passed = self.previous_test_result.get("passed", 0)
                    prev_failed = self.previous_test_result.get("failed", 0)
                    prev_score = self.previous_test_result.get("score", 0.0)
                    prompt_lines.append(f"Previous test score: {prev_score}% ({prev_passed}/{prev_total} passed, {prev_failed} failed)")
                    
                    # 显示失败的测试用例（包含输入输出）
                    prev_cases = self.previous_test_result.get("cases", [])
                    prev_failed_cases = [c for c in prev_cases if c.get("status") == "failed"]
                    if prev_failed_cases:
                        prompt_lines.append("")
                        prompt_lines.append("Failed test cases from previous attempt:")
                        for case in prev_failed_cases[:5]:  # 最多显示5个失败用例
                            idx = case.get("index", "?")
                            description = case.get("description", f"Test case {idx}")
                            test_code = case.get("test_code", "")
                            actual_value = case.get("actual_value")
                            expected_value = case.get("expected_value")
                            
                            prompt_lines.append(f"  [{idx}] {description}")
                            if test_code:
                                prompt_lines.append(f"    Input: {test_code[:150]}")
                            if actual_value is not None and expected_value is not None:
                                prompt_lines.append(f"    Expected output: {expected_value}")
                                prompt_lines.append(f"    Actual output: {actual_value}")
                            prompt_lines.append("")
                    
                    # 显示通过的测试用例（作为参考）
                    prev_passed_cases = [c for c in prev_cases if c.get("status") == "passed"]
                    if prev_passed_cases:
                        prompt_lines.append(f"Passed test cases ({len(prev_passed_cases)}):")
                        for case in prev_passed_cases[:3]:  # 最多显示3个通过的用例作为参考
                            idx = case.get("index", "?")
                            description = case.get("description", f"Test case {idx}")
                            test_code = case.get("test_code", "")
                            prompt_lines.append(f"  [{idx}] {description}")
                            if test_code:
                                prompt_lines.append(f"    Input: {test_code[:100]}")
                        prompt_lines.append("")
                    
                    prompt_lines.append("IMPORTANT: You should IMPROVE the previous code based on the failed test cases above.")
                    prompt_lines.append("Do NOT rewrite from scratch. Instead:")
                    prompt_lines.append("  1. Keep the working parts of the previous code")
                    prompt_lines.append("  2. Fix the bugs that caused test failures")
                    prompt_lines.append("  3. Ensure all previously passed tests still pass")
                    prompt_lines.append("  4. Make minimal changes to fix the issues")
                    prompt_lines.append("")
        
        # 添加可用操作说明
        prompt_lines.append("=== Available Actions ===")
        prompt_lines.append("You can perform the following actions:")
        prompt_lines.append("1. ADD (action=0, file_index): Generate and stage code for a target file")
        prompt_lines.append("2. COMMIT (action=1, file_index=0): Commit staged files to repository")
        prompt_lines.append("3. RUN_TEST (action=2, file_index=0): Run tests to verify your code")
        prompt_lines.append("")
        
        # 添加目标文件列表（如果还没有全部完成）
        if self.current_task:
            all_files_done = all(f in self.files for f in self.current_task.target_files)
            if not all_files_done:
                prompt_lines.append("=== Target Files (to be implemented) ===")
                for i, target_file in enumerate(self.current_task.target_files):
                    status = "✓ Done" if target_file in self.files else "✗ Not done"
                    prompt_lines.append(f"{i}. {target_file} [{status}]")
                    
                    # 如果文件未完成，显示该文件的详细需求
                    if target_file not in self.files and self.task_config:
                        files_config = self.task_config.get("files", [])
                        for file_config in files_config:
                            if file_config.get("name") == target_file:
                                requirements = file_config.get("requirements", [])
                                if requirements:
                                    prompt_lines.append(f"   Requirements:")
                                    for req in requirements[:3]:  # 最多显示3个需求
                                        prompt_lines.append(f"     - {req}")
                                function_signature = file_config.get("function_signature", "")
                                if function_signature:
                                    prompt_lines.append(f"   Function signature: {function_signature}")
                                break
                prompt_lines.append("")
        
        # 添加输出格式说明
        prompt_lines.append("=== Output Format ===")
        prompt_lines.append("IMPORTANT: If you choose ACTION 0 (ADD), you MUST include the COMPLETE Python code in your response!")
        prompt_lines.append("")
        prompt_lines.append("Response format:")
        prompt_lines.append("ACTION: <0 or 1 or 2>")
        prompt_lines.append("FILE_INDEX: <file index for ADD action, 0 for others>")
        prompt_lines.append("EXPLANATION: <brief explanation of your action>")
        prompt_lines.append("")
        prompt_lines.append("CRITICAL: If ACTION is 0 (ADD), you MUST also include:")
        prompt_lines.append("CODE:")
        prompt_lines.append("<complete Python code here - NO markdown, NO explanations, just the code>")
        prompt_lines.append("")
        prompt_lines.append("Example for ADD action:")
        prompt_lines.append("ACTION: 0")
        prompt_lines.append("FILE_INDEX: 0")
        prompt_lines.append("EXPLANATION: Implementing fibonacci function")
        prompt_lines.append("CODE:")
        prompt_lines.append("def fibonacci(n):")
        prompt_lines.append("    if n <= 1:")
        prompt_lines.append("        return n")
        prompt_lines.append("    return fibonacci(n-1) + fibonacci(n-2)")
        prompt_lines.append("")
        prompt_lines.append("What action would you like to take?")
        
        return "\n".join(prompt_lines)
