"""Monkey-patch: change placement group strategy from PACK to SPREAD.

slime uses PACK strategy by default, which tries to pack all bundles on
one node. With 16 bundles (8 actor + 8 rollout) and 2 nodes (8 GPUs each),
PACK can result in all 16 bundles on one node with 2 bundles per GPU,
causing "Duplicate GPU detected" NCCL errors.

SPREAD strategy distributes bundles evenly across all available nodes,
ensuring each rank gets a unique GPU. This is required for PP>1 where
all 8 actor ranks must be on distinct GPUs.

NOTE: SPREAD is only needed in COLOCATE mode, where training and rollout
share the same placement group and need to be spread across nodes to avoid
GPU conflicts. In NON-COLOCATE mode, PACK is better: it naturally separates
training bundles (0-31) and rollout bundles (32-39) onto different nodes,
preventing the SGLang engine and Megatron actor from landing on the same GPU.
"""
import logging

logger = logging.getLogger(__name__)

_original_create_placement_group = None


def _patched_create_placement_group(num_gpus):
    """Replacement that uses SPREAD in colocate, PACK in non-colocate."""
    import os
    import ray
    from ray.util.placement_group import placement_group, PlacementGroupSchedulingStrategy

    # In non-colocate mode, use PACK so training and rollout bundles
    # are separated onto different nodes (training fills nodes 1-4,
    # rollout fills node 5). SPREAD would mix them on the same nodes,
    # causing SGLang and Megatron to share GPUs → OOM.
    colocate = os.environ.get("SLIME_COLOCATE", "false").lower() in ("true", "1")
    strategy = "SPREAD" if colocate else "PACK"
    logger.info(f"Placement group strategy: {strategy} (colocate={colocate})")

    bundles = [{"GPU": 1, "CPU": 1} for _ in range(num_gpus)]
    pg = placement_group(bundles, strategy=strategy)
    num_bundles = len(bundles)

    ray.get(pg.ready())

    # use info actor to get the GPU id
    from slime.ray.placement_group import InfoActor

    info_actors = []
    for i in range(num_bundles):
        info_actors.append(
            InfoActor.options(
                scheduling_strategy=PlacementGroupSchedulingStrategy(
                    placement_group=pg,
                    placement_group_bundle_index=i,
                )
            ).remote()
        )
    gpu_ids = ray.get([actor.get_ip_and_gpu_id.remote() for actor in info_actors])
    for actor in info_actors:
        ray.kill(actor)

    bundle_infos = [(i, gpu_ids[i][0], gpu_ids[i][1]) for i in range(num_bundles)]

    def sort_key(info):
        return (info[1], info[2])

    sorted_bundle_infos = sorted(bundle_infos, key=sort_key)
    pg_reordered_bundle_indices = [info[0] for info in sorted_bundle_infos]
    pg_reordered_gpu_ids = [gpu_ids[info[0]][1] for info in sorted_bundle_infos]

    for i in range(num_bundles):
        actual_bundle_index = pg_reordered_bundle_indices[i]
        logger.info(
            f"  bundle {i:4}, actual_bundle_index: {actual_bundle_index:4}, "
            f"node: {gpu_ids[actual_bundle_index][0]}, gpu: {gpu_ids[actual_bundle_index][1]}"
        )

    return pg, pg_reordered_bundle_indices, pg_reordered_gpu_ids


def apply_patch():
    global _original_create_placement_group
    from slime.ray import placement_group as slime_pg_module

    _original_create_placement_group = slime_pg_module._create_placement_group
    slime_pg_module._create_placement_group = _patched_create_placement_group
    logger.info("Patched _create_placement_group: strategy now depends on SLIME_COLOCATE")


apply_patch()
