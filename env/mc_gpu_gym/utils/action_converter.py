import json
import math
import logging
from typing import Any, Dict, List, Tuple, Optional, Union

import numpy as np

from . import action_mapping

logger = logging.getLogger(__name__)


def _map_cam_21_to_11(camera_dec: int) -> int:
    y21, x21 = divmod(int(camera_dec), 21)
    y11 = int(round(y21 / 20 * 10))
    x11 = int(round(x21 / 20 * 10))
    return y11 * 11 + x11


class ActionFromLLMConverter:
    """
    Convert an LLM response string (JSON list with fields like action, yaw/pitch or point)
    into a packed action dict {"buttons": int or np.array([int]), "camera": int or np.array([int])}
    compatible with the simulator.

    This mirrors the mapping logic in agent_wrapper.py:
    - Supports FOV-aware angle mapping to 21×21 camera bins
    - Enables camera only when yaw/pitch or point is provided
    - Uses OneActionTokenizer bases and packing
    - Optional mapping of 21×21 camera index to 11×11 (VPT) before returning
    - Optional scalar return (default) for buttons/camera instead of np.array([int])
    """

    def __init__(
        self,
        tokenizer: Optional[action_mapping.OneActionTokenizer] = None,
        hfov_deg: float = 70.0,
        vfov_deg: Optional[float] = None,
        map_camera_to_11: bool = False,
        return_numpy: bool = False,
    ) -> None:
        self.tokenizer = tokenizer or action_mapping.OneActionTokenizer(tokenizer_type="qwen2_vl")
        self.hfov_deg = hfov_deg
        self.vfov_deg = vfov_deg
        self.map_camera_to_11 = map_camera_to_11
        self.return_numpy = return_numpy

    @staticmethod
    def _extract_json_list(text: str) -> List[Dict[str, Any]]:
        try:
            start = text.index("[")
            end = text.rindex("]") + 1
            return json.loads(text[start:end])
        except Exception:
            return []

    @staticmethod
    def _derive_vfov_if_needed(image_shape: Tuple[int, int] | np.ndarray, hfov_deg: Optional[float], vfov_deg: Optional[float]) -> Optional[float]:
        try:
            if vfov_deg is not None:
                return vfov_deg
            if hfov_deg is None:
                return None
            if isinstance(image_shape, np.ndarray):
                h, w = image_shape.shape[0], image_shape.shape[1]
            else:
                h, w = int(image_shape[0]), int(image_shape[1])
            return 2.0 * math.degrees(math.atan(math.tan(math.radians(hfov_deg * 0.5)) * (h / max(1.0, float(w)))))
        except Exception:
            return vfov_deg

    def _normalized_coord_to_bins(self, normalized_x: float, normalized_y: float, cam_x_base: int = 21, cam_y_base: int = 21) -> Tuple[int, int]:
        """
        将归一化坐标 [0, 1] 转换为 bin 索引
        x: 水平方向 (yaw)，0=最左，1=最右
        y: 垂直方向 (pitch)，0=最上，1=最下
        """
        # 将归一化坐标映射到 bin 索引
        x_bin = int(round(normalized_x * (cam_x_base - 1)))
        y_bin = int(round(normalized_y * (cam_y_base - 1)))
        
        # 限制范围
        x_bin = max(0, min(cam_x_base - 1, x_bin))
        y_bin = max(0, min(cam_y_base - 1, y_bin))
        
        return y_bin, x_bin

    def _bins_to_angles(self, y_bin: int, x_bin: int, image_shape: Tuple[int, int] | np.ndarray, cam_x_base: int = 21, cam_y_base: int = 21) -> Tuple[float, float]:
        """
        将 bin 索引转换回角度（用于调试打印）
        """
        # 获取 FOV
        hfov = self.hfov_deg
        vfov = self._derive_vfov_if_needed(image_shape, self.hfov_deg, self.vfov_deg)
        
        if vfov is None:
            vfov = 43.0  # 默认值
        
        half_h = hfov * 0.5
        half_v = vfov * 0.5
        
        # bin -> normalized -> angle
        normalized_x = x_bin / (cam_x_base - 1)
        normalized_y = y_bin / (cam_y_base - 1)
        
        # normalized [-1, 1] -> angle
        scaled_x = normalized_x * 2.0 - 1.0
        scaled_y = normalized_y * 2.0 - 1.0
        
        yaw = scaled_x * half_h
        pitch = scaled_y * half_v
        
        return yaw, pitch

    def _to_bins_from_point(self, point_xy: Tuple[float, float], image_shape: Tuple[int, int] | np.ndarray, cam_x_base: int, cam_y_base: int) -> Tuple[int, int]:
        if isinstance(image_shape, np.ndarray):
            h, w = image_shape.shape[0], image_shape.shape[1]
        else:
            h, w = int(image_shape[0]), int(image_shape[1])
        cx, cy = w / 2.0, h / 2.0
        x, y = float(point_xy[0]), float(point_xy[1])
        dx = (x - cx) / max(1.0, w / 2.0)
        dy = (y - cy) / max(1.0, h / 2.0)
        dx = max(-1.0, min(1.0, dx))
        dy = max(-1.0, min(1.0, dy))
        x_bin = int(round((dx + 1) * 0.5 * (cam_x_base - 1)))
        y_bin = int(round((dy + 1) * 0.5 * (cam_y_base - 1)))
        x_bin = max(0, min(cam_x_base - 1, x_bin))
        y_bin = max(0, min(cam_y_base - 1, y_bin))
        return y_bin, x_bin

    def _to_bins_from_angles(self, yaw_deg: float, pitch_deg: float, image_shape: Tuple[int, int] | np.ndarray, cam_x_base: int, cam_y_base: int) -> Tuple[int, int]:
        # clamp/wrap yaw
        yaw = float(yaw_deg)
        if yaw < -180.0:
            yaw = ((yaw + 180.0) % 360.0) - 180.0
        elif yaw > 180.0:
            yaw = ((yaw + 180.0) % 360.0) - 180.0
        # clamp pitch to physical range
        pitch = float(pitch_deg)
        pitch = max(-90.0, min(90.0, pitch))
        # FOV-aware mapping if possible
        hfov = self.hfov_deg
        vfov = self._derive_vfov_if_needed(image_shape, self.hfov_deg, self.vfov_deg)
        if isinstance(hfov, (int, float)) and isinstance(vfov, (int, float)) and hfov > 0 and vfov > 0:
            half_h = hfov * 0.5
            half_v = vfov * 0.5
            yaw = max(-half_h, min(half_h, yaw))
            pitch = max(-half_v, min(half_v, pitch))
            x_bin = int(round(((yaw / max(1e-6, half_h)) + 1.0) * 0.5 * (cam_x_base - 1)))
            y_bin = int(round(((pitch / max(1e-6, half_v)) + 1.0) * 0.5 * (cam_y_base - 1)))
        else:
            x_bin = int(round(((yaw + 180.0) / 360.0) * (cam_x_base - 1)))
            y_bin = int(round(((pitch + 90.0) / 180.0) * (cam_y_base - 1)))
        x_bin = max(0, min(cam_x_base - 1, x_bin))
        y_bin = max(0, min(cam_y_base - 1, y_bin))
        return y_bin, x_bin

    def _wrap(self, buttons_dec: int, camera_dec: int) -> action_mapping.OrderedDict:
        if self.map_camera_to_11:
            camera_dec = _map_cam_21_to_11(camera_dec)
        if self.return_numpy:
            return action_mapping.OrderedDict({
                "buttons": np.array([buttons_dec]),
                "camera": np.array([camera_dec]),
            })
        else:
            return action_mapping.OrderedDict({
                "buttons": int(buttons_dec),
                "camera": int(camera_dec),
            })

    def convert(self, response_text: str, image_shape: Tuple[int, int] | np.ndarray) -> action_mapping.OrderedDict:
        """
        Convert an LLM response string to a single-step packed action OrderedDict.
        If multiple actions are present, only the first is returned.
        """
        bases = getattr(self.tokenizer, "bases", [10, 3, 3, 3, 2, 2, 2, 2, 2, 2, 21, 21])
        num_digits = len(bases)
        inv_idx = num_digits - 3
        camera_meta_idx = num_digits - 4
        cam_y_base = bases[-2]
        cam_x_base = bases[-1]
        cam_mid_y = cam_y_base // 2
        cam_mid_x = cam_x_base // 2

        actions = self._extract_json_list(response_text)
        if not actions:
            # null (no-op) action
            digits = [0] * num_digits
            digits[camera_meta_idx] = 0
            digits[-2] = cam_mid_y
            digits[-1] = cam_mid_x
            buttons_dec, camera_dec = self.tokenizer.group_action_2_decimal_action(digits)
            return self._wrap(buttons_dec, camera_dec)

        act = actions[0] if isinstance(actions[0], dict) else {}
        
        # 处理 action 字段（可能是字符串或列表）
        action_field = act.get("action", "")
        if isinstance(action_field, list):
            # 如果是列表，取第一个动作
            a_type = action_field[0].lower() if action_field else ""
        else:
            a_type = action_field.lower()

        def build_action(
            use=False, attack=False, jump=False, inventory=False,
            forward=False, back=False, left=False, right=False, sprint=False, sneak=False,
            camera_bins: Optional[Tuple[int, int]] = None,
            point: Optional[Tuple[float, float]] = None,
        ) -> action_mapping.OrderedDict:
            digits = [0] * num_digits
            # movement/select groups
            if forward:
                digits[1] = 1
            if back:
                digits[1] = 2
            if left:
                digits[2] = 1
            if right:
                digits[2] = 2
            if sprint:
                digits[3] = 1
            if sneak:
                digits[3] = 2
            # function buttons
            if use:
                digits[4] = 1
            if act.get("drop") or False:
                pass  # leave drop to explicit branch
            if attack:
                digits[6] = 1
            if jump:
                digits[7] = 1
            if inventory:
                digits[inv_idx] = 1

            # camera meta: enable only when bins or point provided
            meta = 0
            if camera_bins is not None and len(camera_bins) == 2:
                y_bin, x_bin = int(camera_bins[0]), int(camera_bins[1])
                meta = 1
            elif point is not None:
                y_bin, x_bin = self._to_bins_from_point(point, image_shape, cam_x_base, cam_y_base)
                meta = 1
            else:
                y_bin, x_bin = cam_mid_y, cam_mid_x
                meta = 0

            y_bin = max(0, min(cam_y_base - 1, y_bin))
            x_bin = max(0, min(cam_x_base - 1, x_bin))
            digits[camera_meta_idx] = meta
            digits[-2] = y_bin
            digits[-1] = x_bin

            buttons_dec, camera_dec = self.tokenizer.group_action_2_decimal_action(digits)
            
            # 打印调试信息
            if meta == 1:
                yaw, pitch = self._bins_to_angles(y_bin, x_bin, image_shape, cam_x_base, cam_y_base)
                logger.info(f"[Camera] bins=({y_bin}, {x_bin}) -> yaw={yaw:.2f}°, pitch={pitch:.2f}° | buttons={buttons_dec}, camera={camera_dec}")
            else:
                logger.debug(f"[No Camera] meta={meta}, bins=({y_bin},{x_bin}), buttons={buttons_dec}, camera={camera_dec}")
            
            return self._wrap(buttons_dec, camera_dec)

        # parse possible camera fields
        camera_bins_override: Optional[Tuple[int, int]] = None
        
        # 优先处理 new_screen_center 字段
        if "new_screen_center" in act:
            center = act.get("new_screen_center")
            if isinstance(center, (list, tuple)) and len(center) == 2:
                norm_x, norm_y = float(center[0]), float(center[1])
                camera_bins_override = self._normalized_coord_to_bins(norm_x, norm_y, cam_x_base, cam_y_base)
                logger.info(f"[new_screen_center] input=[{norm_x:.3f}, {norm_y:.3f}] -> bins={camera_bins_override}")
        # 其次处理传统的 yaw/pitch
        elif ("yaw" in act) and ("pitch" in act):
            camera_bins_override = self._to_bins_from_angles(act.get("yaw"), act.get("pitch"), image_shape, cam_x_base, cam_y_base)
        elif "point" in act and isinstance(act.get("point"), (list, tuple)) and len(act.get("point")) == 2:
            camera_bins_override = self._to_bins_from_point(act.get("point"), image_shape, cam_x_base, cam_y_base)
        elif "camera" in act and isinstance(act.get("camera"), (list, tuple)) and len(act.get("camera")) == 2:
            yb, xb = int(act["camera"][0]), int(act["camera"][1])
            camera_bins_override = (yb, xb)

        # route per action type
        if a_type == "attack":
            return build_action(attack=True, camera_bins=camera_bins_override)
        if a_type == "use":
            return build_action(use=True, camera_bins=camera_bins_override)
        if a_type == "drop":
            return build_action(camera_bins=camera_bins_override)
        if a_type == "jump":
            return build_action(jump=True, camera_bins=camera_bins_override)
        if a_type == "inventory":
            return build_action(inventory=True, camera_bins=camera_bins_override)
        if a_type in ("forward", "back", "left", "right", "sprint", "sneak"):
            return build_action(
                forward=(a_type == "forward"), back=(a_type == "back"), left=(a_type == "left"), right=(a_type == "right"),
                sprint=(a_type == "sprint"), sneak=(a_type == "sneak"),
                camera_bins=camera_bins_override,
            )
        if a_type.startswith("hotbar."):
            try:
                n = int(a_type.split(".")[-1])
            except Exception:
                n = 1
            digits = [0] * num_digits
            digits[0] = max(1, min(9, n))
            # do not rotate on hotbar by default
            y_bin, x_bin = cam_mid_y, cam_mid_x
            digits[camera_meta_idx] = 0
            y_bin = max(0, min(cam_y_base - 1, y_bin))
            x_bin = max(0, min(cam_x_base - 1, x_bin))
            digits[-2] = y_bin
            digits[-1] = x_bin
            buttons_dec, camera_dec = self.tokenizer.group_action_2_decimal_action(digits)
            logger.debug(f"[hotbar] meta={digits[camera_meta_idx]}, bins=({y_bin},{x_bin}), buttons={buttons_dec}, camera={camera_dec}")
            return self._wrap(buttons_dec, camera_dec)

        # fallback: no-op
        return build_action(camera_bins=None)