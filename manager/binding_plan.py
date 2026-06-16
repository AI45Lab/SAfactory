from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Set

from .repository import AgentDataRepository


@dataclass(frozen=True)
class BindingPlan:
    env_to_image: Dict[str, str]
    image_to_env: Dict[str, str]
    images_needed: Set[str]


def build_binding_plan(repo: AgentDataRepository) -> BindingPlan:
    """
    Build agent->image bindings and discover distinct images needed.
    """
    env_image_map = repo.get_env_image_map()
    if not env_image_map:
        return BindingPlan(env_to_image={}, image_to_env={}, images_needed=set())

    missing_images = [str(env_name) for env_name, image in env_image_map.items() if not (image or "").strip()]
    if missing_images:
        raise RuntimeError(
            "Each agent must have an explicit image in DB; missing images for: "
            + ", ".join(sorted(missing_images))
        )

    image_to_env = repo.get_image_to_env_map()
    images_needed: Set[str] = set(image_to_env.keys())

    final_env_image: Dict[str, str] = {}
    for env_name, image in env_image_map.items():
        env_name = str(env_name)
        effective_image = (image or "").strip()
        final_env_image[env_name] = effective_image
        images_needed.add(effective_image)
        if effective_image not in image_to_env:
            image_to_env[effective_image] = env_name

    return BindingPlan(
        env_to_image=final_env_image,
        image_to_env=image_to_env,
        images_needed=images_needed,
    )
