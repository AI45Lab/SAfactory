"""Action execution flow for OSGym."""

import logging
import time
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger("osgym")


class ActionFlow:
    def __init__(self, osgym):
        self.osgym = osgym

    def step(self, action: str):
        env = self.osgym
        if env.task_finished:
            duplicate_output = env.get_duplicate_terminal_output()
            if duplicate_output is not None:
                logger.info(
                    "Received duplicate step after task %s finished; returning cached terminal output.",
                    env.task_id,
                )
                return duplicate_output
            raise RuntimeError("Task already finished. Cannot step on a finished task.")

        env.current_step_in_task += 1

        parsed_actions, special_cmd = self._parse_agent_action(action)

        logger.debug(
            "Step %s/%s, Parsed Actions: %s",
            env.current_step_in_task,
            env.max_steps,
            parsed_actions,
        )

        repeat_result = env.repeated_action_detector.check(
            parsed_actions,
            special_cmd=special_cmd,
        )
        if repeat_result and repeat_result.repeated:
            env.prompt_session.add_assistant_message(action, parsed_actions)
            info = {
                "executed_actions": [],
                "truncated_reason": repeat_result.reason,
                "repeated_action": repeat_result.action,
                "repeat_count": repeat_result.repeat_count,
            }
            logger.info(
                "Detected repeated action %s for %s consecutive steps; truncating task.",
                repeat_result.action,
                repeat_result.repeat_count,
            )
            return env._finish_task(
                info=info,
                agent_signal="TRUNCATED",
                skip_evaluation=False,
            )

        env.prompt_session.add_assistant_message(action, parsed_actions)

        if special_cmd == "WAIT":
            return self._handle_wait_signal()

        if special_cmd in {"DONE", "FAIL"} and not parsed_actions:
            return self._finish_after_terminal_signal(
                special_cmd,
                executed_actions=[],
            )

        if env.eval_mode == "safety":
            env.evaluator.capture_pre_action_state()

        executed_actions, reward, done, info = self._execute_actions(parsed_actions)
        info = self._attach_step_risk(info, executed_actions)

        info.setdefault("executed_actions", executed_actions)

        if env.eval_mode == "safety" and bool(info.get("risk_result")):
            info["termination_reason"] = "risk_triggered"
            logger.info("Risk triggered for task %s; terminating safety episode.", env.task_id)
            return env._finish_task(info=info)

        if done:
            return env._finish_task(info=info)

        if special_cmd in {"DONE", "FAIL"}:
            return self._finish_after_terminal_signal(
                special_cmd,
                executed_actions,
                info,
            )

        if env.current_step_in_task >= env.max_steps:
            return self._finish_at_step_limit(
                info,
                executed_actions=executed_actions,
            )

        return env._step_output(reward=reward, info=info)

    def _finish_at_step_limit(
        self,
        info: Optional[Dict[str, Any]],
        executed_actions: Optional[List[Any]] = None,
    ):
        """Finish the horizon according to the active model protocol."""
        env = self.osgym
        finish_info = dict(info) if info else {}
        if env.eval_mode == "safety":
            finish_info["termination_reason"] = "safety_horizon_reached"
        else:
            finish_info["truncated_reason"] = "max_steps_reached"

        step_limit_signal = getattr(env.model_protocol, "step_limit_signal", "FAIL")
        if step_limit_signal:
            logger.info(
                "Reached max steps (%s); applying protocol terminal signal %s.",
                env.max_steps,
                step_limit_signal,
            )
            return self._finish_after_terminal_signal(
                step_limit_signal,
                list(executed_actions or []),
                finish_info,
            )

        logger.info(
            "Reached max steps (%s); ending at the horizon without a synthetic FAIL.",
            env.max_steps,
        )
        return env._finish_task(info=finish_info)

    def _parse_agent_action(self, action: str) -> Tuple[List[str], Optional[str]]:
        env = self.osgym
        parsed_actions = env.model_protocol.parse_actions(action) or ["WAIT"]
        return env.model_protocol.strip_special_command(parsed_actions)

    def _execute_actions(
        self,
        actions: List[str],
    ) -> Tuple[List[Any], float, bool, Dict[str, Any]]:
        env = self.osgym
        executed_actions: List[Any] = []
        info: Dict[str, Any] = {}
        reward = 0.0
        done = False

        for act in actions:
            _, reward, done, info = env.env.step(
                act,
                pause=env.sleep_after_execution,
                return_observation=False,
            )
            executed_actions.append(act)
            env._mark_observation_stale()
            if done:
                break

        return executed_actions, reward, done, info or {}

    def _finish_after_terminal_signal(
        self,
        signal: str,
        executed_actions: List[Any],
        info: Optional[Dict[str, Any]] = None,
    ):
        env = self.osgym
        env.env.step(signal, pause=env.sleep_after_execution, return_observation=False)
        env._mark_observation_stale()

        finish_info = dict(info) if info else {}
        finish_info.setdefault("executed_actions", executed_actions or [signal])
        task_score_override = 0.0 if signal == "FAIL" and env.eval_mode == "safety" else None
        return env._finish_task(
            info=finish_info,
            agent_signal=signal,
            task_score_override=task_score_override,
        )

    def _attach_step_risk(
        self,
        info: Optional[Dict[str, Any]],
        executed_actions: List[Any],
    ) -> Dict[str, Any]:
        env = self.osgym
        info = dict(info) if info else {}
        if env.eval_mode != "safety":
            return info

        risk_result = env.evaluator.evaluate_step_risk(
            executed_actions,
            env.task_id,
            env.obs_processor.get_attack_params(),
            risk_evaluator=env.task_config.get("risk_evaluator"),
        )
        if risk_result is not None:
            info["risk_result"] = risk_result
            env.risk_results.append(risk_result)
        return info

    def _handle_wait_signal(self):
        env = self.osgym
        if env.current_step_in_task >= env.max_steps:
            info = {"executed_actions": []}
            return self._finish_at_step_limit(
                info,
                executed_actions=[],
            )

        wait_time = env.sleep_after_execution if env.sleep_after_execution > 0 else 1.0
        logger.debug(f"WAIT signal received, waiting {wait_time}s...")
        time.sleep(wait_time)

        try:
            env._refresh_observation()
        except Exception as e:
            logger.warning(f"Failed to get new observation after WAIT: {e}")

        info = {"agent_signal": "WAIT", "executed_actions": []}
        return env._step_output(info=info)
