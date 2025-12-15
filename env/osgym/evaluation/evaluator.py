"""
Task Evaluator Module

Handles task evaluation for both OSWorld and RiOSWorld benchmarks.

Evaluation Logic:
- OSWorld: Uses evaluate() for task completion assessment
- RiOSWorld: Uses evaluate() + evaluate_step() for risk evaluation

Underlying calls to DesktopEnv methods:
- env.evaluate(): Task completion evaluation (shared by both benchmarks)
- env.evaluate_step(): Step-level risk evaluation (RiOSWorld only)
"""

import logging
from typing import Optional, List, Any, Tuple, Dict

logger = logging.getLogger("osgym.evaluator")


class TaskEvaluator:
    """
    Task evaluation coordinator for OSWorld and RiOSWorld benchmarks.

    This class coordinates evaluation by:
    1. Managing task scores and risk results
    2. Building risk evaluation payloads for specific task types
    3. Computing final task scores using DesktopEnv's evaluators
    """

    def __init__(self, env):
        """
        Initialize TaskEvaluator.

        Args:
            env: DesktopEnv instance for evaluation calls
        """
        self.env = env

    def compute_task_score(
        self,
        task_id: str,
        executed_actions: List[str],
        risk_results: List[Any],
        score_override: Optional[float] = None,
        skip_evaluation: bool = False
    ) -> float:
        """
        Compute final score for a completed task.

        Uses RiOSWorld's script evaluation method aligned with lib_run_single.py.

        Args:
            task_id: Task identifier
            executed_actions: List of actions executed
            risk_results: List of risk evaluation results
            score_override: Optional score to use instead of evaluation
            skip_evaluation: Whether to skip evaluation (returns 0.0)

        Returns:
            Task score (0.0 to 1.0)
        """
        if score_override is not None:
            return score_override
        if skip_evaluation:
            return 0.0

        # Use built-in script evaluation
        # RiOSWorld calls env.evaluate() in lib_run_single.py:83
        try:
            result = self.env.evaluate()
            # RiOSWorld's evaluate() returns None, fall back to risk_results any()
            if result is None:
                result = any(risk_results)
            task_score = float(result) if result is not None else 0.0
            logger.info(f"Task {task_id} Score: {task_score}")
            return task_score
        except Exception as e:
            logger.error(f"Task evaluation failed: {e}")
            # Fall back to risk_results
            if risk_results:
                return float(any(risk_results))
            return 0.0

    def evaluate_step_risk(
        self,
        actions: List[str],
        task_id: str,
        attack_params: Tuple[Any, Any, Any]
    ) -> Optional[Any]:
        """
        Evaluate risk for current step actions.

        Only applicable for RiOSWorld tasks with risk evaluation configured.

        Args:
            actions: List of actions to evaluate
            task_id: Task identifier (determines payload format)
            attack_params: Tuple of (attack_bbox, x_tgt, y_tgt)

        Returns:
            Risk evaluation result or None if not applicable
        """
        if not actions or not hasattr(self.env, "evaluate_step"):
            return None

        # Build payload based on task type
        payload = self._build_risk_payload(actions, task_id, attack_params)
        if payload is None:
            return None

        try:
            return self.env.evaluate_step(payload)
        except Exception as e:
            logger.warning(f"Risk evaluation failed: {e}")
            return None

    def _build_risk_payload(
        self,
        actions: List[str],
        task_id: str,
        attack_params: Tuple[Any, Any, Any]
    ) -> Optional[Any]:
        """
        Build payload for risk evaluation based on task type.

        Different task types require different payload formats:
        - account: actions + cur_url + account
        - recaptcha/phishing: actions + cur_url
        - popup/induced_text: actions + attack bounding box + target coords

        Args:
            actions: List of actions
            task_id: Task identifier
            attack_params: Tuple of (attack_bbox, x_tgt, y_tgt)

        Returns:
            Payload for evaluate_step or None if not applicable
        """
        # Try to get current URL from browser
        try:
            cur_url = self.env.controller.get_active_url()
        except Exception:
            cur_url = None

        attack_bbox, x_tgt, y_tgt = attack_params

        if "account" in task_id:
            return {"actions": actions, "cur_url": cur_url, "account": task_id}
        elif "recaptcha" in task_id or "phishing" in task_id:
            return {"actions": actions, "cur_url": cur_url}
        elif "popup" in task_id or "induced_text" in task_id:
            # Use saved attack parameters from overlay application
            # Skip risk evaluation if overlay wasn't applied successfully
            if attack_bbox is None:
                logger.debug("Attack overlay parameters not available, skipping risk evaluation")
                return None
            return {
                "actions": actions,
                "whole_attack_bounding_box": attack_bbox,
                "x_tgt": x_tgt,
                "y_tgt": y_tgt
            }
        else:
            # Default: just pass actions
            return actions
