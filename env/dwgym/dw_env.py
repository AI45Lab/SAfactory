import json
import os
import sys
import time
import base64
from typing import Any, Dict, List, Optional

_current_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.dirname(os.path.dirname(_current_dir))
_dw_outer = os.path.join(_current_dir, 'discoveryworld')

if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

if _current_dir not in sys.path:
    sys.path.insert(0, _current_dir)

if _dw_outer not in sys.path:
    sys.path.insert(0, _dw_outer)

import gymnasium as gym

from openai.types.chat import ChatCompletionMessageParam
from core.types.base import (
    ResetOutput, StepOutput, RenderOutput,
    PromptOutput, OpenAIMessage, MessageContent, TextContent, ImageContent
)
from core.env.base_env import BaseEnv
from core.env.env_register import register_env

try:
    from discoveryworld.DiscoveryWorldAPI import DiscoveryWorldAPI
    from discoveryworld.ScenarioMaker import (
        SCENARIO_NAMES, 
        SCENARIO_DIFFICULTY_OPTIONS,
        SCENARIO_INFOS
    )
except ImportError as e:
    print("=" * 70)
    print("ERROR: DiscoveryWorld import failed!")
    print("=" * 70)
    print("Please check:")
    print(f"  1. Project root directory: {_project_root}")
    print(f"  2. dwgym directory: {_current_dir}")
    print(f"  3. DiscoveryWorld path: {os.path.join(_current_dir, 'discoveryworld', 'discoveryworld')}")
    print()
    print("First 5 entries of current sys.path:")
    for i, p in enumerate(sys.path[:5], 1):
        print(f"  {i}. {p}")
    print("=" * 70)
    print(f"Error details: {str(e)}")
    print("=" * 70)
    raise e


@register_env("discoveryworld")
class DiscoveryWorldEnv(BaseEnv):
    
    def __init__(self, 
                 thread_id: int = 0,
                 scenario_name: Optional[str] = None,
                 difficulty: str = "Normal",
                 seed: int = 0,
                 max_steps: int = 300,
                 use_vision: bool = False,
                 capture_frames: bool = False,
                 narrate_actions: bool = True,
                 verbose: bool = True,
                 max_history_length: Optional[int] = None,
                 max_recent_actions: int = 5,
                 **kwargs):
        super().__init__(**kwargs)
        
        self.thread_id = thread_id
        self.scenario_name = scenario_name or SCENARIO_NAMES[0]
        self.difficulty = difficulty
        self.seed = seed
        self.max_steps = max_steps
        self.use_vision = use_vision
        self.capture_frames = capture_frames
        self.narrate_actions = narrate_actions
        self.verbose = verbose
        self.max_history_length = max_history_length
        self.max_recent_actions = max_recent_actions  
        
        # validate configuration
        if self.scenario_name not in SCENARIO_NAMES:
            raise ValueError(
                f"Unknown scenario: {self.scenario_name}.\n"
                f"Valid options: {SCENARIO_NAMES}"
            )
        
        if isinstance(SCENARIO_DIFFICULTY_OPTIONS, dict):
            valid_difficulties = list(SCENARIO_DIFFICULTY_OPTIONS.values())
        else:
            valid_difficulties = SCENARIO_DIFFICULTY_OPTIONS
        
        if self.difficulty not in valid_difficulties:
            raise ValueError(
                f"Unknown difficulty: {self.difficulty}.\n"
                f"Valid options: {valid_difficulties}"
            )
        
        # environment state
        self.api: Optional[DiscoveryWorldAPI] = None
        self.step_count = 0
        self.last_observation_dict = None
        self.last_action_result = None
        self.last_action_dict = None
        self.last_agent_response = None
        self._last_normalized_score = 0.0
        
        # conversation history
        self.conversation_history: List[ChatCompletionMessageParam] = []
        self.recent_actions: List[Dict[str, Any]] = []
    
        # video frame history
        if self.capture_frames:
            self.frame_history: List[str] = []
        else:
            self.frame_history = None
        
        # define actions & observation spaces
        self.action_space = gym.spaces.Text(max_length=2000)
        self.observation_space = gym.spaces.Text(max_length=100000)
        
        if self.verbose:
            print(f"[INFO] DiscoveryWorldEnv initialized: {self.scenario_name}/{self.difficulty}/seed{self.seed}")
    
    def reset(self, seed: Optional[int] = None, **kwargs) -> ResetOutput:
        if seed is not None:
            self.seed = seed
        
        scenario_name = kwargs.get('scenario_name', self.scenario_name)
        difficulty = kwargs.get('difficulty', self.difficulty)
        max_steps = kwargs.get('max_steps', self.max_steps)
        
        self.api = DiscoveryWorldAPI(threadID=self.thread_id)
        
        success = self.api.loadScenario(
            scenarioName=scenario_name,
            difficultyStr=difficulty,
            randomSeed=self.seed,
            numUserAgents=1
        )
        
        if not success:
            raise RuntimeError(
                f"Failed to load scenario '{scenario_name}' "
                f"with difficulty '{difficulty}'"
            )
        
        self.scenario_name = scenario_name
        self.difficulty = difficulty
        self.max_steps = max_steps
        self.step_count = 0
        self.last_action_result = None
        self.last_action_dict = None
        self._last_normalized_score = 0.0
        self.recent_actions = []
        
        self.conversation_history = []
        
        if self.capture_frames:
            self.frame_history = []
        
        self.last_observation_dict = self.api.getAgentObservation(agentIdx=0)
        
        system_message = {"role": "system", "content": self._build_system_prompt()}
        
        self.conversation_history.append(system_message)
        
        observation_text = self._format_observation(
            self.last_observation_dict,
            action_narration="Environment initialized. Ready to begin task."
        )
        
        frame_base64 = None
        if self.use_vision:
            frame_base64 = self._get_current_frame()
            if frame_base64:
                self.conversation_history.append(
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": observation_text
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{frame_base64}"
                                }
                            }
                        ]
                    }
                )
            else:
                self.conversation_history.append(
                    {"role": "user", "content":observation_text}
                )
        
        info = {
            "scenario": self.scenario_name,
            "difficulty": self.difficulty,
            "seed": self.seed,
            "step": self.step_count,
            "max_steps": self.max_steps,
            "available_scenarios": SCENARIO_NAMES,
            "available_difficulties": list(SCENARIO_DIFFICULTY_OPTIONS.values()) if isinstance(SCENARIO_DIFFICULTY_OPTIONS, dict) else SCENARIO_DIFFICULTY_OPTIONS,
            "scenario_infos": SCENARIO_INFOS,
            "has_vision": self.use_vision,
            "frame_count": len(self.frame_history) if self.capture_frames else 0,
            "conversation_length": len(self.conversation_history),
        }
        
        observation = {
            "text": observation_text,
            "image": frame_base64,
            "raw_obs": self.last_observation_dict,
        }
        
        return ResetOutput(observation=observation, info=info)
    
    def step(self, action: str) -> StepOutput:
        if self.api is None:
            raise RuntimeError("Environment not initialized. Call reset() first.")
        
        self.last_agent_response = action
        
        try:
            action_dict = json.loads(action)
            self.last_action_dict = action_dict
        except json.JSONDecodeError:
            lines = action.strip().split('\n')
            json_found = False
            
            for line in reversed(lines):
                line = line.strip()
                if line.startswith('{') and line.endswith('}'):
                    try:
                        action_dict = json.loads(line)
                        self.last_action_dict = action_dict
                        json_found = True
                        break
                    except json.JSONDecodeError:
                        continue
            
            if not json_found:
                error_obs = (
                    f"ERROR: Invalid action JSON\n"
                    f"Error: Could not find valid JSON in response\n"
                    f"Received: {action}\n\n"
                    f"Expected format: The last line should be a valid JSON object like:\n"
                    f'{{"action": "ACTION_NAME", "arg1": "value1", "arg2": "value2"}}'
                )
                return StepOutput(
                    observation={
                        "text": error_obs,
                        "image": None,
                        "raw_obs": None,
                    },
                    reward=0.0,
                    terminated=False,
                    truncated=False,
                    info={"error": "Invalid JSON format", "step": self.step_count}
                )
        
        try:
            assistant_message = {"role": "assistant", "content": action}
            self._add_to_conversation_history(assistant_message)
            
            # Compatibility handling: direction actions
            if action_dict.get("action") in ["MOVE_DIRECTION", "ROTATE_DIRECTION"]:
                arg1 = action_dict.get("arg1")
                if isinstance(arg1, str) and arg1.lower() in ["north", "south", "east", "west"]:
                    self.last_action_result = self.api.performAgentAction(
                        agentIdx=0, 
                        actionJSON=action_dict
                    )
                    
                    if not self.last_action_result.get("success", False):
                        errors = self.last_action_result.get("errors", [])
                        if any("expected type of arg1 is 'int'" in str(err) for err in errors):
                            direction_map = {"north": 0, "east": 1, "south": 2, "west": 3}
                            action_dict_int = action_dict.copy()
                            action_dict_int["arg1"] = direction_map.get(arg1.lower(), 0)
                            
                            if self.verbose:
                                print(f"[INFO] Retrying with integer direction: {action_dict_int}")
                            
                            self.last_action_result = self.api.performAgentAction(
                                agentIdx=0,
                                actionJSON=action_dict_int
                            )
                else:
                    self.last_action_result = self.api.performAgentAction(
                        agentIdx=0,
                        actionJSON=action_dict
                    )
            else:
                self.last_action_result = self.api.performAgentAction(
                    agentIdx=0, 
                    actionJSON=action_dict
                )
            
            self.api.tick()
            self.step_count += 1
            
            self.last_observation_dict = self.api.getAgentObservation(agentIdx=0)
            
            action_narration = self._generate_action_narration(
                action_dict, 
                self.last_action_result
            )
            action_summary = {
                "step": self.step_count,
                "action": action_dict.get("action"),
                "result_summary": self._summarize_action_result(action_dict, self.last_action_result),
                "success": self.last_action_result.get("success", True)
            }
            
            self.recent_actions.append(action_summary)
            
            # Keep only the most recent N steps
            if len(self.recent_actions) > self.max_recent_actions:
                self.recent_actions.pop(0)
            
            observation_text = self._format_observation(
                self.last_observation_dict,
                action_narration=action_narration
            )
            
            frame_base64 = None
            user_content = [{"type": "text", "text": observation_text}]

            if self.use_vision:
                frame_base64 = self._get_current_frame()
                if frame_base64:
                    user_content.append({
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{frame_base64}"}
                    })
                    if self.capture_frames:
                        self.frame_history.append(frame_base64)

            user_message = {"role": "user", "content": user_content}
            self._add_to_conversation_history(user_message)
            
            scorecard = self.api.getTaskScorecard()
            
            if isinstance(scorecard, dict):
                current_score = float(scorecard.get("scoreNormalized", 0.0))
                task_completed = scorecard.get("completed", False)
            elif isinstance(scorecard, list) and scorecard:
                first_card = scorecard[0]
                if isinstance(first_card, dict):
                    current_score = float(first_card.get("scoreNormalized", 0.0))
                    task_completed = first_card.get("completed", False)
                else:
                    current_score = 0.0
                    task_completed = False
                scorecard = first_card if isinstance(first_card, dict) else {}
            else:
                current_score = 0.0
                task_completed = False
                scorecard = {}
            
            reward = current_score - self._last_normalized_score
            self._last_normalized_score = current_score
            
            max_steps_reached = self.step_count >= self.max_steps
            
            terminated = task_completed
            truncated = max_steps_reached and not task_completed
            
            info = {
                "step": self.step_count,
                "max_steps": self.max_steps,
                "action": action_dict,
                "action_result": self.last_action_result,
                "action_narration": action_narration,
                "scorecard": scorecard,
                "completed": task_completed,
                "completed_successfully": scorecard.get("completedSuccessfully", False),
                "score": scorecard.get("score", 0),
                "score_normalized": current_score,
                "max_score": scorecard.get("maxScore", 0),
                "frame_count": len(self.frame_history) if self.capture_frames else 0,
                "conversation_length": len(self.conversation_history),
            }
            
            observation = {
                "text": observation_text,
                "image": frame_base64,
                "raw_obs": self.last_observation_dict,
            }
            
            return StepOutput(
                observation=observation,
                reward=reward,
                terminated=terminated,
                truncated=truncated,
                info=info
            )
            
        except AttributeError as e:
            import traceback
            error_details = traceback.format_exc()
            print(f"\n{'='*70}")
            print("AttributeError details:")
            print(f"{'='*70}")
            print(error_details)
            print(f"{'='*70}\n")
            
            return StepOutput(
                observation={
                    "text": f"ERROR in step: {str(e)}\nSee console for details.",
                    "image": None,
                    "raw_obs": None,
                },
                reward=0.0,
                terminated=False,
                truncated=False,
                info={"error": str(e), "step": self.step_count, "traceback": error_details}
            )
    
    def get_task_prompt(self) -> List[ChatCompletionMessageParam]:
        return self.conversation_history
    
    def render(self) -> RenderOutput:
        """Generate a combined frame: visual + text information."""
        if self.api is None or self.last_observation_dict is None:
            return RenderOutput(
                step=self.step_count,
                text_content="Environment not initialized. Call reset() first."
            )
        
        # return plain text if vision is disabled
        if not self.use_vision:
            return self._render_text_fallback()
        
        try:
            from PIL import Image, ImageDraw, ImageFont
            from io import BytesIO
            
            # get visual frame
            frame_base64 = self._get_current_frame()
            if not frame_base64:
                return self._render_text_fallback()
            
            frame_data = base64.b64decode(frame_base64)
            visual_img = Image.open(BytesIO(frame_data))
            
            # generate text info
            text_content = self._generate_render_text_compact()
            
            # load font
            font = self._load_font()
            
            # render text as image
            text_img = self._render_text_to_image(text_content, visual_img.width, font)
            
            # concatenate
            combined_height = visual_img.height + text_img.height
            combined_img = Image.new('RGB', (visual_img.width, combined_height), color='white')
            combined_img.paste(visual_img, (0, 0))
            combined_img.paste(text_img, (0, visual_img.height))
            
            # convert to bytes
            buffer = BytesIO()
            combined_img.save(buffer, format='PNG')
            image_bytes = buffer.getvalue()
            
            return RenderOutput(
                step=self.step_count,
                image_data=image_bytes,
                image_base64=base64.b64encode(image_bytes).decode('utf-8'),
                text_content=text_content
            )
            
        except ImportError:
            if self.verbose:
                print("[WARNING] PIL not available, falling back to text render")
            return self._render_text_fallback()
        except Exception as e:
            if self.verbose:
                print(f"[WARNING] Error generating combined frame: {e}")
            return self._render_text_fallback()

    def _render_text_fallback(self) -> RenderOutput:
        """Fallback: plain text (when PIL is unavailable or vision is disabled)."""
        render_parts = []
        ui = self.last_observation_dict.get("ui", {})
        
        # Task
        task_progress = ui.get("taskProgress", [])
        if task_progress:
            render_parts.append("=" * 70)
            render_parts.append("TASK")
            render_parts.append("=" * 70)
            first_task = task_progress[0] if task_progress else {}
            if isinstance(first_task, dict):
                render_parts.append(first_task.get("description", "No description"))
            else:
                render_parts.append(str(first_task))
            render_parts.append("")
        
        # Agent status
        agent_loc = ui.get("agentLocation", {})
        if agent_loc and isinstance(agent_loc, dict):
            render_parts.append("=" * 70)
            render_parts.append("AGENT STATUS")
            render_parts.append("=" * 70)
            
            if 'location' in agent_loc:
                location = agent_loc['location']
            elif 'x' in agent_loc and 'y' in agent_loc:
                location = f"({agent_loc['x']}, {agent_loc['y']})"
            else:
                location = "Unknown"
            
            if 'facing_direction' in agent_loc:
                facing = agent_loc['facing_direction']
            elif 'faceDirection' in agent_loc:
                facing = agent_loc['faceDirection']
            else:
                facing = "Unknown"
            
            render_parts.append(f"Location: {location}")
            render_parts.append(f"Facing: {facing}")
            render_parts.append(f"Step: {self.step_count}/{self.max_steps}")
            render_parts.append("")
        
        # Inventory
        inventory = ui.get("inventoryObjects", [])
        if inventory:
            render_parts.append("=" * 70)
            render_parts.append("INVENTORY")
            render_parts.append("=" * 70)
            for item in inventory:
                if isinstance(item, dict):
                    render_parts.append(f"  - {item.get('name', 'Unknown')}")
                else:
                    formatted = self._safe_format_object(item)
                    if formatted:
                        render_parts.append(formatted)
            render_parts.append("")
        else:
            render_parts.append("=" * 70)
            render_parts.append("INVENTORY")
            render_parts.append("=" * 70)
            render_parts.append("  (empty)")
            render_parts.append("")
        
        # Last action
        if self.last_action_dict and self.last_action_result:
            render_parts.append("=" * 70)
            render_parts.append("LAST ACTION")
            render_parts.append("=" * 70)
            
            if self.last_agent_response:
                render_parts.append("Agent Response:")
                render_parts.append(self.last_agent_response)
                render_parts.append("")
            
            narration = self._generate_action_narration(
                self.last_action_dict,
                self.last_action_result
            )
            render_parts.append("Result:")
            render_parts.append(narration)
            render_parts.append("")
        
        # Verbose debug
        if self.verbose and self.last_observation_dict:
            render_parts.append("=" * 70)
            render_parts.append("DEBUG: ACTUAL OBSERVATION SENT TO AGENT")
            render_parts.append("=" * 70)
            action_narration = self._generate_action_narration(
                self.last_action_dict or {},
                self.last_action_result or {}
            ) if self.last_action_dict else "Environment initialized"
            actual_obs = self._format_observation(
                self.last_observation_dict,
                action_narration=action_narration
            )
            render_parts.append(actual_obs)
            render_parts.append("")
        
        # Progress
        scorecard = self.api.getTaskScorecard()
        if isinstance(scorecard, list) and scorecard:
            scorecard = scorecard[0] if isinstance(scorecard[0], dict) else {}
        elif not isinstance(scorecard, dict):
            scorecard = {}
        
        render_parts.append("=" * 70)
        render_parts.append("PROGRESS")
        render_parts.append("=" * 70)
        render_parts.append(f"Score: {scorecard.get('score', 0)}/{scorecard.get('maxScore', 0)}")
        render_parts.append(f"Normalized: {scorecard.get('scoreNormalized', 0.0):.3f}")
        render_parts.append(f"Completed: {scorecard.get('completed', False)}")
        render_parts.append(f"Successfully: {scorecard.get('completedSuccessfully', False)}")
        
        return RenderOutput(
            step=self.step_count,
            text_content="\n".join(render_parts)
        )

    def _generate_render_text_compact(self) -> str:
        """Generate detailed text (for combined frame)."""
        lines = []
        ui = self.last_observation_dict.get("ui", {})
        
        # get progress info
        scorecard = self.api.getTaskScorecard()
        if isinstance(scorecard, list) and scorecard:
            scorecard = scorecard[0] if isinstance(scorecard[0], dict) else {}
        elif not isinstance(scorecard, dict):
            scorecard = {}
        
        score = scorecard.get('score', 0)
        max_score = scorecard.get('maxScore', 0)
        score_normalized = scorecard.get('scoreNormalized', 0.0)
        
        lines.append(f"STEP {self.step_count}/{self.max_steps} | Score: {score}/{max_score} ({score_normalized:.2f})")
        lines.append("=" * 80)
        
        # task description (auto-wrapped, no truncated)
        task_progress = ui.get("taskProgress", [])
        if task_progress:
            lines.append("TASK:")
            first_task = task_progress[0] if task_progress else {}
            if isinstance(first_task, dict):
                task_desc = first_task.get("description", "No description")
                # Auto-wrap at 100 characters per line
                wrapped_lines = self._wrap_text(task_desc, width=100, indent="  ")
                lines.extend(wrapped_lines)
            lines.append("")
        
        # recent history
        if self.recent_actions:
            lines.append("RECENT HISTORY:")
            for action_info in self.recent_actions[-3:]:
                symbol = "success" if action_info.get("success") else "fail"
                summary = action_info['result_summary']
                wrapped = self._wrap_text(summary, width=75, indent="     ")
                lines.append(f"  {action_info['step']:2d}. {wrapped[0]} [{symbol}]")
                if len(wrapped) > 1:
                    lines.extend(wrapped[1:])
            lines.append("")
        
        # current location & inventory
        agent_loc = ui.get("agentLocation", {})
        if agent_loc and isinstance(agent_loc, dict):
            if 'location' in agent_loc:
                location = agent_loc['location']
            elif 'x' in agent_loc and 'y' in agent_loc:
                location = f"({agent_loc['x']}, {agent_loc['y']})"
            else:
                location = "Unknown"
            
            if 'facing_direction' in agent_loc:
                facing = agent_loc['facing_direction']
            elif 'faceDirection' in agent_loc:
                facing = agent_loc['faceDirection']
            else:
                facing = "Unknown"
            
            lines.append(f"LOCATION: {location} | FACING: {facing}")
        
        inventory = ui.get("inventoryObjects", [])
        if inventory:
            inv_names = [item.get('name', 'Unknown')[:20] for item in inventory[:3] if isinstance(item, dict)]
            if len(inventory) > 3:
                inv_names.append(f"(+{len(inventory)-3} more)")
            lines.append(f"INVENTORY: {', '.join(inv_names)}")
        else:
            lines.append("INVENTORY: (empty)")
        lines.append("")
        
        # Agent reasoning (auto-wrapped)
        if self.last_agent_response:
            lines.append("AGENT REASONING:")
            response_lines = self.last_agent_response.strip().split('\n')
            thinking_lines = []
            for line in response_lines:
                line = line.strip()
                if line.startswith('{') and line.endswith('}'):
                    break
                if line and not line.startswith('```'):
                    thinking_lines.append(line)
            
            # show first 3 lines of reasoning, auto-wrapped
            for line in thinking_lines[:3]:
                wrapped = self._wrap_text(line, width=100, indent="  ")
                lines.extend(wrapped)
            
            if len(thinking_lines) > 3:
                lines.append(f"  ... ({len(thinking_lines)-3} more lines)")
            lines.append("")
        
        # last action(detailed)
        if self.last_action_dict and self.last_action_result:
            action_name = self.last_action_dict.get("action", "UNKNOWN")
            arg1 = self.last_action_dict.get("arg1")
            arg2 = self.last_action_dict.get("arg2")
            success = self.last_action_result.get("success", True)
            symbol = "success" if success else "fail"
            
            lines.append(f"LAST ACTION: {action_name} [{symbol}]")
            
            if arg1 is not None:
                lines.append(f"  arg1: {arg1}")
            if arg2 is not None:
                lines.append(f"  arg2: {arg2}")
            
            ui_msg = ui.get("lastActionMessage", "")
            if ui_msg:
                # display result message with wrapping
                lines.append("  Result:")
                wrapped = self._wrap_text(ui_msg, width=95, indent="    ")
                lines.extend(wrapped)
        
        return "\n".join(lines)

    def _wrap_text(self, text: str, width: int = 80, indent: str = "") -> List[str]:
        if not text:
            return [indent]
        
        words = text.split()
        lines = []
        current_line = indent
        
        for word in words:
            test_line = current_line + (" " if current_line != indent else "") + word
            if len(test_line) <= width:
                current_line = test_line
            else:
                # current line is full, start a new line
                if current_line.strip():
                    lines.append(current_line)
                current_line = indent + word
        
        # append the last line
        if current_line.strip():
            lines.append(current_line)
        
        return lines if lines else [indent]

    def _load_font(self, size: int = 14):
        """load font"""
        from PIL import ImageFont
        
        font_paths = [
            "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
            "/System/Library/Fonts/Monaco.ttf",
            "C:/Windows/Fonts/consola.ttf",
            "/usr/share/fonts/truetype/liberation/LiberationMono-Regular.ttf",
        ]
        
        for font_path in font_paths:
            try:
                return ImageFont.truetype(font_path, size)
            except:
                continue
        
        return ImageFont.load_default()

    def _render_text_to_image(self, text: str, width: int, font):
        """render text to image"""
        from PIL import Image, ImageDraw
        
        lines = text.split('\n')
        line_height = 16  # slightly compact
        padding = 12
        height = len(lines) * line_height + padding * 2
        
        img = Image.new('RGB', (width, height), color='#f8f8f8')
        draw = ImageDraw.Draw(img)
        draw.rectangle([0, 0, width-1, height-1], outline='#cccccc', width=2)
        
        y = padding
        for line in lines:
            # color coding
            if line.startswith("STEP") or line.startswith("TASK:"):
                color = '#0066cc'  
            elif line.startswith("LOCATION") or line.startswith("INVENTORY"):
                color = '#0066cc'  
            elif line.startswith("RECENT HISTORY") or line.startswith("AGENT REASONING") or line.startswith("LAST ACTION"):
                color = '#006600' 
            elif '[success]' in line:
                color = '#009900' 
            elif '[fail]' in line:
                color = '#cc0000' 
            elif '=' in line:
                color = '#999999' 
            elif line.startswith("  "):
                color = '#333333'
            else:
                color = '#000000'
            
            draw.text((padding, y), line, fill=color, font=font)
            y += line_height
        
        return img
    
    def close(self):
        if self.api is not None:
            try:
                if hasattr(self.api, "shutdown"):
                    self.api.shutdown()
                elif hasattr(self.api, "close"):
                    self.api.close()
            except Exception as e:
                if self.verbose:
                    print(f"[WARNING] Error during DW API cleanup: {e}")
            finally:
                self.api = None
        
        self.conversation_history.clear()
        
        if self.capture_frames:
            self.frame_history.clear()
        
        if self.verbose:
            print(f"[INFO] Environment closed (thread_id={self.thread_id})")
    
    def _add_to_conversation_history(self, message):
        self.conversation_history.append(message)
        
        if self.max_history_length and len(self.conversation_history) > self.max_history_length:
            system_msg = self.conversation_history[0]
            recent_msgs = self.conversation_history[-(self.max_history_length - 1):]
            self.conversation_history = [system_msg] + recent_msgs
            
            if self.verbose:
                print(f"[INFO] Conversation history truncated to {self.max_history_length} messages")
    
    def _get_current_frame(self) -> Optional[str]:
        if not self.use_vision or self.last_observation_dict is None:
            return None
        
        vision = self.last_observation_dict.get("vision", {})
        frame_with_prefix = vision.get("base64_no_grid")
        
        if not frame_with_prefix:
            return None
        
        if frame_with_prefix.startswith("data:image/png;base64,"):
            frame = frame_with_prefix.replace("data:image/png;base64,", "", 1)
        else:
            frame = frame_with_prefix
        
        return frame
    
    def _create_image_content(self, frame_base64: str) -> ImageContent:
        return ImageContent(
            type="image_url",
            image_url={"url": f"data:image/png;base64,{frame_base64}"}
        )
    
    def _summarize_action_result(self, action_dict: Dict[str, Any], action_result: Dict[str, Any]) -> str:
        """Generate a short summary of the action result (for history display)"""
        action_name = action_dict.get("action", "UNKNOWN")
        success = action_result.get("success", True)
        
        # get result message
        ui = self.last_observation_dict.get("ui", {})
        last_msg = ui.get("lastActionMessage", "")
        
        # generate summary based on action type
        if action_name == "MOVE_DIRECTION":
            direction = action_dict.get('arg1', 'unknown')
            return f"Moved {direction} {'success' if success else 'fail'}"
        
        elif action_name == "PICKUP":
            if success and last_msg:
                # extract item name
                if "picked up" in last_msg.lower():
                    return last_msg[:50] + "..." if len(last_msg) > 50 else last_msg
            return f"Pickup {'success' if success else 'fail'}"
        
        elif action_name == "DROP":
            if success and last_msg:
                return last_msg[:50] + "..." if len(last_msg) > 50 else last_msg
            return f"Drop {'success' if success else 'fail'}"
        
        elif action_name == "USE":
            if last_msg:                
                # general case
                if len(last_msg) > 80:
                    return f"{last_msg[:80]}..."
                return last_msg
            return f"Used tool {'success' if success else 'fail'}"
        
        elif action_name == "OPEN":
            return f"Opened container {'success' if success else 'fail'}"
        
        elif action_name == "READ":
            return f"Read document {'success' if success else 'fail'}"
        
        elif action_name == "TALK":
            return f"Talked to agent {'success' if success else 'fail'}"
        
        else:
            # general format
            return f"{action_name} {'success' if success else 'fail'}"
    
    def _generate_action_narration(self, 
                                   action_dict: Dict[str, Any],
                                   action_result: Dict[str, Any]) -> str:
        """Generate a human-readable narration of the action."""
        if not self.narrate_actions:
            return json.dumps(action_result)
        
        action_name = action_dict.get("action", "UNKNOWN")
        arg1 = action_dict.get("arg1")
        arg2 = action_dict.get("arg2")
        
        ui = self.last_observation_dict.get("ui", {})
        last_msg = ui.get("lastActionMessage", "")
        
        narration_parts = []
        
        action_descriptions = {
            "MOVE_DIRECTION": f"Agent moved {arg1}",
            "ROTATE_DIRECTION": f"Agent rotated to face {arg1}",
            "PICKUP": f"Agent attempted to pick up item [{arg1}]" if arg1 else "Agent attempted to pick up item",
            "DROP": f"Agent attempted to drop item [{arg1}]" if arg1 else "Agent attempted to drop item",
            "PUT": f"Agent attempted to put item [{arg1}] in container [{arg2}]" if arg1 and arg2 else "Agent attempted to put item in container",
            "USE": f"Agent used tool [{arg1}] on target [{arg2}]" if arg1 and arg2 else "Agent used tool on target",
            "OPEN": f"Agent opened container [{arg1}]" if arg1 else "Agent opened container",
            "CLOSE": f"Agent closed container [{arg1}]" if arg1 else "Agent closed container",
            "ACTIVATE": f"Agent activated device [{arg1}]" if arg1 else "Agent activated device",
            "DEACTIVATE": f"Agent deactivated device [{arg1}]" if arg1 else "Agent deactivated device",
            "TALK": f"Agent talked to character [{arg1}]" if arg1 else "Agent talked to another character",
            "READ": f"Agent read document [{arg1}]" if arg1 else "Agent read document",
            "EAT": f"Agent ate item [{arg1}]" if arg1 else "Agent ate item",
            "TELEPORT_TO_LOCATION": f"Agent teleported to {arg1}",
            "TELEPORT_TO_OBJECT": f"Agent teleported to object [{arg1}]" if arg1 else "Agent teleported to object",
            "DISCOVERY_FEED_GET_UPDATES": f"Agent checked Discovery Feed",
            "DISCOVERY_FEED_GET_POST_BY_ID": f"Agent read Discovery Feed post [{arg1}]" if arg1 else "Agent read Discovery Feed post",
        }
        
        if "chosen_dialog_option_int" in action_dict:
            option_num = action_dict["chosen_dialog_option_int"]
            narration_parts.append(f"Agent chose dialog option {option_num}")
        else:
            description = action_descriptions.get(
                action_name,
                f"Agent performed action: {action_name}"
            )
            narration_parts.append(description)
        
        if last_msg:
            narration_parts.append(f"Result: {last_msg}")
        
        success = action_result.get("success", True)
        if success:
            narration_parts.append("[SUCCESS]")
        else:
            narration_parts.append("[FAILED]")
            errors = action_result.get("errors", [])
            if errors:
                cleaned_errors = []
                for error in errors:
                    error_str = str(error)
                    
                    # handle "Could not find object" error
                    if "Could not find object with UUID" in error_str:
                        import re
                        uuid_match = re.search(r"UUID '(\d+)'", error_str)
                        arg_match = re.search(r"(arg\d+)", error_str)
                        
                        if uuid_match and arg_match:
                            uuid = uuid_match.group(1)
                            arg = arg_match.group(1)
                            
                            # provide a clearer hint
                            cleaned_errors.append(
                                f"{arg}: Object [{uuid}] is not accessible. "
                                f"Only objects in 'Accessible Objects' list can be used. "
                                f"If it's in 'Nearby Objects', move closer first."
                            )
                            continue
                    
                    # keep other errors as-is
                    cleaned_errors.append(error_str)
                
                narration_parts.append(f"Errors: {', '.join(cleaned_errors)}")
            else:
                narration_parts.append("Errors: Action failed with no error details")
        
        return "\n".join(narration_parts)
    
    def _build_system_prompt(self) -> str:
        task_description = "Complete a scientific discovery task"
        if self.last_observation_dict:
            ui = self.last_observation_dict.get("ui", {})
            task_progress = ui.get("taskProgress", [])
            if task_progress:
                first_task = task_progress[0] if task_progress else {}
                if isinstance(first_task, dict):
                    task_description = first_task.get("description", task_description)
        
        vision_note = "Visual observations (top-down view) are provided." if self.use_vision else ""
        
        return f"""You are a scientific research agent in DiscoveryWorld.

TASK: {task_description}

OBSERVATIONS:
You'll receive text descriptions of:
- Recent action history (last few steps you took)
- Your current location and facing direction
- Your inventory
- Accessible Objects (can interact immediately)
- Nearby Objects (need to move closer first)
- Task progress and score
{vision_note}

ACTIONS:
Movement:
- {{"action": "MOVE_DIRECTION", "arg1": "north"}}  (north/south/east/west)

Object interaction:
- {{"action": "PICKUP", "arg1": 12345}}  (UUID from Accessible Objects)
- {{"action": "DROP", "arg1": 12345}}
- {{"action": "USE", "arg1": tool_uuid, "arg2": target_uuid}}
- {{"action": "OPEN", "arg1": container_uuid}}

Information gathering:
- {{"action": "TALK", "arg1": agent_uuid}}
- {{"action": "READ", "arg1": document_uuid}}
- {{"action": "DISCOVERY_FEED_GET_UPDATES"}}

STRATEGY:
- Review your recent history to avoid repeating actions
- Explore systematically to map the environment
- Read documents and talk to NPCs for information
- Check Discovery Feed for research findings
- Experiment carefully and observe results

KEY RULES:
1. Use UUIDs from "Accessible Objects" for interactions
2. Move closer to "Nearby Objects" before interacting
3. If action fails, analyze why and try differently
4. In dialog mode, respond with: {{"chosen_dialog_option_int": <number>}}
5. Check your recent history to track your progress

OUTPUT FORMAT:
[Optional: your reasoning/thinking]
{{"action": "ACTION_NAME", "arg1": value}}

The JSON MUST be on the last line and properly formatted."""


    def _safe_format_object(self, obj: Any, include_desc: bool = False) -> Optional[str]:
        if obj is None:
            return None
        
        if isinstance(obj, dict):
            name = obj.get('name', 'Unknown')
            uuid = obj.get('uuid', 'no-uuid')
            result = f"  - {name} [{uuid}]"
            
            if include_desc:
                desc = obj.get('description', '')
                if desc:
                    result += f"\n    Description: {desc}"
            
            return result
        
        elif isinstance(obj, (list, tuple)):
            if not obj:
                return None
            for item in obj:
                formatted = self._safe_format_object(item, include_desc)
                if formatted:
                    return formatted
            return None
        
        else:
            return f"  - {obj}"
    
    def _format_observation(self, 
                          observation_dict: Dict[str, Any],
                          action_narration: Optional[str] = None) -> str:
        obs_parts = []
        ui = observation_dict.get("ui", {})
        
        # step counter
        obs_parts.append("=" * 70)
        obs_parts.append(f"STEP {self.step_count}/{self.max_steps}")
        obs_parts.append("=" * 70)
        obs_parts.append("")
        
        if self.recent_actions and len(self.recent_actions) > 0:
            obs_parts.append("RECENT HISTORY")
            obs_parts.append("-" * 70)
            # show all saved histroy（up to max_recent_actions steps）
            for action_info in self.recent_actions:
                obs_parts.append(f"  Step {action_info['step']}: {action_info['result_summary']}")
            obs_parts.append("")
        
        # task description
        task_progress = ui.get("taskProgress", [])
        if task_progress:
            obs_parts.append("TASK")
            obs_parts.append("-" * 70)
            first_task = task_progress[0] if task_progress else {}
            if isinstance(first_task, dict):
                obs_parts.append(first_task.get("description", "No description"))
                
                #  if API provides a progress description, display it
                if "taskProgressDescription" in first_task:
                    obs_parts.append(f"\nProgress: {first_task['taskProgressDescription']}")
                
                completed = first_task.get("completed", False)
            else:
                obs_parts.append(str(first_task))
                completed = False
            status = "[COMPLETED]" if completed else "[IN PROGRESS]"
            obs_parts.append(f"Status: {status}")
            obs_parts.append("")
        
        # action narration
        if action_narration:
            obs_parts.append("LAST ACTION")
            obs_parts.append("-" * 70)
            obs_parts.append(action_narration)
            obs_parts.append("")
        
        # dialog mode warning
        if self.api and self.api.isAgentInDialog(agentIdx=0):
            obs_parts.append("!" * 70)
            obs_parts.append("DIALOG MODE - YOU MUST CHOOSE A RESPONSE!")
            obs_parts.append("!" * 70)
            dialog_info = ui.get("dialog_box", {})
            obs_parts.append(json.dumps(dialog_info, indent=2))
            obs_parts.append("")
            obs_parts.append("RESPOND WITH: {\"chosen_dialog_option_int\": <number>}")
            obs_parts.append("!" * 70)
            obs_parts.append("")
        
        # current state
        obs_parts.append("CURRENT STATE")
        obs_parts.append("-" * 70)
        
        agent_loc = ui.get("agentLocation", {})
        
        if agent_loc and isinstance(agent_loc, dict):            
            if 'location' in agent_loc:
                location = agent_loc['location']
            elif 'x' in agent_loc and 'y' in agent_loc:
                location = f"({agent_loc['x']}, {agent_loc['y']})"
            else:
                location = "Unknown"
            
            if 'facing_direction' in agent_loc:
                facing = agent_loc['facing_direction']
            elif 'faceDirection' in agent_loc:
                facing = agent_loc['faceDirection']
            else:
                facing = "Unknown"
            
            obs_parts.append(f"Location: {location}")
            obs_parts.append(f"Facing: {facing}")
            
            can_move = agent_loc.get("directions_can_move_to") or agent_loc.get("directions_you_can_move", [])
            if can_move:
                obs_parts.append(f"Can move to: {', '.join(can_move)}")
            
            obs_parts.append("")
        elif agent_loc:
            obs_parts.append(f"Location: [Raw data] {str(agent_loc)[:100]}")
            obs_parts.append("")
        
        # inventory
        inventory = ui.get("inventoryObjects", [])
        if inventory:
            obs_parts.append("Inventory:")
            for item in inventory:
                formatted = self._safe_format_object(item)
                if formatted:
                    obs_parts.append(formatted)
            obs_parts.append("")
        else:
            obs_parts.append("Inventory: (empty)")
            obs_parts.append("")
        
        # interactable objects
        accessible = ui.get("accessibleEnvironmentObjects", [])
        
        if accessible:
            useless_objects = ['floor', 'wall', 'air', 'ground']
            useful_objects = []
            
            for obj in accessible:
                if isinstance(obj, dict):
                    obj_name = obj.get('name', '').lower()
                    if obj_name not in useless_objects:
                        useful_objects.append(obj)
                else:
                    useful_objects.append(obj)
            
            if useful_objects:
                obs_parts.append("Accessible Objects (can interact with these):")
                for obj in useful_objects:
                    formatted = self._safe_format_object(obj, include_desc=True)
                    if formatted:
                        obs_parts.append(formatted)
                obs_parts.append("")
        
        # nearby objetcs
        nearby = ui.get("nearbyObjects", {})
        
        if isinstance(nearby, dict):
            nearby_objects_list = nearby.get("objects", {})
            
            if isinstance(nearby_objects_list, dict) and any(nearby_objects_list.values()):
                obs_parts.append("Nearby Objects:")
                for direction, objects in sorted(nearby_objects_list.items()):
                    if objects and isinstance(objects, (list, tuple)) and direction.lower() != "same_location":
                        obs_parts.append(f"  {direction.upper()}:")
                        for obj in objects[:5]:
                            formatted = self._safe_format_object(obj, include_desc=False)
                            if formatted:
                                obs_parts.append("  " + formatted)
                obs_parts.append("")
            
            elif isinstance(nearby_objects_list, list) and nearby_objects_list:
                obs_parts.append("Nearby Objects:")
                for obj in nearby_objects_list[:10]:
                    formatted = self._safe_format_object(obj)
                    if formatted:
                        obs_parts.append(formatted)
                obs_parts.append("")
        
        # nearby Agent
        nearby_agents_dict = ui.get("nearbyAgents", {})
        
        if isinstance(nearby_agents_dict, dict):
            nearby_agents = nearby_agents_dict.get("list_of_agents", [])
        else:
            nearby_agents = nearby_agents_dict if isinstance(nearby_agents_dict, list) else []
        
        if nearby_agents:
            obs_parts.append("Nearby Agents:")
            for agent in nearby_agents:
                formatted = self._safe_format_object(agent)
                if formatted:
                    obs_parts.append(formatted)
            obs_parts.append("")
        
        # Discovery Feed
        discovery_feed = ui.get("discoveryFeed", {})
        
        if isinstance(discovery_feed, dict):
            recent_posts = discovery_feed.get("posts", [])
        else:
            recent_posts = []
        
        if recent_posts:
            obs_parts.append("Discovery Feed (Recent):")
            for post in recent_posts[:3]:
                if isinstance(post, dict):
                    author = post.get('author', 'Unknown')
                    content = post.get('content', '')[:50]
                    obs_parts.append(f"  - {author}: {content}...")
                else:
                    formatted = self._safe_format_object(post)
                    if formatted:
                        obs_parts.append(formatted)
            obs_parts.append("")
        
        return "\n".join(obs_parts)
    
    def get_scorecard(self) -> Dict[str, Any]:
        if self.api is None:
            return {}
        return self.api.getTaskScorecard()
    
    def get_conversation_history(self) -> List[ChatCompletionMessageParam]:
        return self.conversation_history.copy()
    
    def get_frame_history(self) -> Optional[List[str]]:
        if self.capture_frames:
            return self.frame_history.copy()
        return None
    
    def export_video(self, output_path: str, fps: int = 2):
        if not self.capture_frames:
            print("[ERROR] Video export requires capture_frames=True during initialization")
            return
        
        if not self.frame_history:
            print("[WARNING] No frames captured")
            return
        
        try:
            import cv2
            import numpy as np
            
            first_frame_data = base64.b64decode(self.frame_history[0])
            nparr = np.frombuffer(first_frame_data, np.uint8)
            first_frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            height, width = first_frame.shape[:2]
            
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            
            for frame_b64 in self.frame_history:
                frame_data = base64.b64decode(frame_b64)
                nparr = np.frombuffer(frame_data, np.uint8)
                frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                out.write(frame)
            
            out.release()
            
            duration = len(self.frame_history) / fps
            print(f"[SUCCESS] Video exported to: {output_path}")
            print(f"          Frames: {len(self.frame_history)}, FPS: {fps}, Duration: {duration:.1f}s")
            
        except ImportError:
            print("[ERROR] opencv-python is required for video export")
            print("        Install with: pip install opencv-python")
        except Exception as e:
            print(f"[ERROR] Error exporting video: {str(e)}")
    
    def export_logs(self, log_directory: str, log_suffix: str = "") -> None:
        if self.api is None:
            print("[WARNING] Environment not initialized, cannot export logs")
            return
        
        os.makedirs(log_directory, exist_ok=True)
        
        log_info = {
            "scenarioName": self.scenario_name,
            "difficulty": self.difficulty,
            "seed": self.seed,
            "numSteps": self.step_count,
            "threadId": self.thread_id,
            "dateStarted": time.strftime("%Y-%m-%d %H:%M:%S"),
            "verboseLogDirectory": log_directory,
            "verboseLogFilename": os.path.join(
                log_directory, 
                f"world-history-{log_suffix}.json"
            )
        }
        
        try:
            self.api.world.exportWorldHistoryJSON(
                log_info,
                log_info["verboseLogFilename"],
                None, None, None
            )
            print(f"[SUCCESS] Logs exported to: {log_directory}")
        except Exception as e:
            print(f"[ERROR] Error exporting logs: {str(e)}")
        
        if self.capture_frames and self.frame_history:
            frames_dir = os.path.join(log_directory, "frames")
            os.makedirs(frames_dir, exist_ok=True)
            
            for i, frame in enumerate(self.frame_history):
                frame_path = os.path.join(frames_dir, f"frame_{i:04d}.png")
                try:
                    with open(frame_path, "wb") as f:
                        f.write(base64.b64decode(frame))
                except Exception as e:
                    print(f"[ERROR] Error saving frame {i}: {str(e)}")
                    break
            
            print(f"[SUCCESS] {len(self.frame_history)} frames exported to: {frames_dir}")
    
    @staticmethod
    def get_available_scenarios() -> List[str]:
        return SCENARIO_NAMES.copy()
    
    @staticmethod
    def get_available_difficulties() -> List[str]:
        if isinstance(SCENARIO_DIFFICULTY_OPTIONS, dict):
            return list(SCENARIO_DIFFICULTY_OPTIONS.values())
        return SCENARIO_DIFFICULTY_OPTIONS.copy()
    
    @staticmethod
    def get_scenario_infos() -> Dict[str, Any]:
        return SCENARIO_INFOS.copy()