from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Set

from .repository import EnvDataRepository


@dataclass(frozen=True)
class BindingPlan:
    env_to_image: Dict[str, str]
    image_to_env: Dict[str, str]
    images_needed: Set[str]
    env_job_counts: Dict[str, int]    #Record how many RayJobs (clusters) we want per env in the initial start.


def build_binding_plan(repo: EnvDataRepository,
                       base_image: str,
                       env_job_counts: Optional[Dict[str, int]] = None
    ) -> BindingPlan:
    """
    Build env->image bindings and discover distinct images needed.

    base_image is required if some env rows have empty image.
    """
    env_image_map = repo.get_env_image_map()
    if not env_image_map:
        return BindingPlan(env_to_image={}, image_to_env={}, images_needed=set(), env_job_counts={})

    base_image = (base_image or "").strip()
    needs_base = any(not (img or "").strip() for img in env_image_map.values())
    if needs_base and not base_image:
        raise RuntimeError(
            "cluster.base_image must be set, or each env must have an explicit image in DB."
        )

    image_to_env = repo.get_image_to_env_map()
    images_needed: Set[str] = set(image_to_env.keys())

    final_env_image: Dict[str, str] = {}
    for env_name, image in env_image_map.items():
        env_name = str(env_name)
        effective_image = (image or "").strip() or base_image
        final_env_image[env_name] = effective_image
        if effective_image:
            images_needed.add(effective_image)
            if effective_image not in image_to_env:
                image_to_env[effective_image] = env_name

    # Remove accidental empty image keys
    images_needed.discard("")

    return BindingPlan(
        env_to_image=final_env_image,
        image_to_env=image_to_env,
        images_needed=images_needed,
        env_job_counts=dict(env_job_counts or {}),
    )
