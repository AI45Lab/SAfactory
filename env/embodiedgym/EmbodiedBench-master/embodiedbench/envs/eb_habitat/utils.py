#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
#
import inspect
import os.path as osp
from typing import Dict, List, Tuple
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import yaml
from habitat.tasks.rearrange.multi_task.pddl_action import PddlAction
from habitat.tasks.rearrange.multi_task.pddl_domain import PddlDomain
from habitat.tasks.rearrange.multi_task.pddl_logical_expr import (
    LogicalExpr, LogicalQuantifierType)
from habitat.tasks.rearrange.multi_task.rearrange_pddl import (
    ExprType, PddlEntity, SimulatorObjectType)
from transformers import (AutoConfig, AutoModelForSeq2SeqLM, AutoTokenizer,
                          LlamaForCausalLM, LlamaModel, LlamaTokenizer,
                          T5Model)
from habitat.utils.visualizations.utils import (tile_images, draw_collision)
import embodiedbench.envs.eb_habitat.config
import embodiedbench.envs.eb_habitat.dataset

# Also defined in the PDDL
PLACABLE_RECEP_TYPE = "place_receptacle"


def draw_text(img, text, position):
    draw = ImageDraw.Draw(img)
    font = ImageFont.load_default()
    font.size = 15
    text = str(text)
    # Draw the text onto the image
    draw.text(position, text, font=font, fill="red")  
    

def merge_to_file(img, image_path, draw_index=True, index=0):
    img_old = Image.open(image_path)
    total_width = img.width + img_old.width
    max_height = max(img.height, img_old.height)
    img_merge = Image.new('RGB', (total_width, max_height))
    img_merge.paste(img_old, (0, 0))
    img_merge.paste(img, (img_old.width, 0))
    if draw_index:
         draw_text(img_merge, index, position=(img_old.width+ 15, 15))
    img_merge.save(image_path)



def process_image_array(obs_k):
    if not isinstance(obs_k, np.ndarray):
        obs_k = obs_k.cpu().numpy()
    if obs_k.dtype != np.uint8:
        obs_k = obs_k * 255.0
        obs_k = obs_k.astype(np.uint8)
    if obs_k.shape[2] == 1:
        obs_k = np.concatenate([obs_k for _ in range(3)], axis=2)
    return obs_k

def observations_to_image(observation, info, key=None):
    render_obs_images = []
    if key is not None:
        obs_k = observation[key]
        obs_k = process_image_array(obs_k)
        render_obs_images.append(obs_k)
    else:
        for sensor_name in observation:
            if hasattr(observation[sensor_name], 'shape') and len(observation[sensor_name].shape) > 1:
                obs_k = observation[sensor_name]
                obs_k = process_image_array(obs_k)
                render_obs_images.append(obs_k)
    assert (
        len(render_obs_images) > 0
    ), "Expected at least one visual sensor enabled."

    shapes_are_equal = len(set(x.shape for x in render_obs_images)) == 1
    if not shapes_are_equal:
        render_frame = tile_images(render_obs_images)
    else:
        render_frame = np.concatenate(render_obs_images, axis=1)

    # draw collision
    collisions_key = "collisions"
    if collisions_key in info and info[collisions_key]["is_collision"]:
        render_frame = draw_collision(render_frame)

    top_down_map_key = "top_down_map"
    if top_down_map_key in info:
        top_down_map = maps.colorize_draw_agent_and_fit_to_height(
            info[top_down_map_key], render_frame.shape[0]
        )
        render_frame = np.concatenate((render_frame, top_down_map), axis=1)
    return render_frame



def get_allowed_actions(pddl, allowed_substrings):
    all_actions = pddl.get_possible_actions()

    def matches_any(s, allowed_acs):
        # returns if the string starts with any strings from allowed_acs
        return any(s.name.startswith(ac) for ac in allowed_acs)

    return [ac for ac in all_actions if matches_any(ac, allowed_substrings)]


def _recur_replace(expr, search, replace):
    if isinstance(expr, LogicalExpr):
        for subexpr in expr.sub_exprs:
            _recur_replace(subexpr, search, replace)
    else:
        for i, arg_val in enumerate(expr._arg_values):
            if arg_val == search:
                expr._arg_values[i] = replace


def flatten_actions(pddl: PddlDomain, obj_cats):
    new_acs = {}
    for ac_name, action in pddl.actions.items():
        found_i = -1
        # TODO: Currently this is a hack for only the pick action. This will
        # not work for other PDDL actions.

        for i, param in enumerate(action.params):
            if param.expr_type.name == SimulatorObjectType.MOVABLE_ENTITY.value:
                found_i = i
                break

        if found_i == -1:
            new_acs[ac_name] = action
            continue

        param = action.params[found_i]
        del action.params[found_i]
        for obj_cat in obj_cats:
            precond = action.precond.clone()
            assert len(precond.inputs) == 0, precond.quantifier is None

            obj_cat_type = pddl.expr_types[obj_cat]
            at_entity = PddlEntity(name="DYN_OBJ", expr_type=obj_cat_type)
            inputs = [at_entity]

            # Ignore the first expression which was about the robot position.
            precond = precond.sub_in({param: at_entity})

            postcond_pred = pddl.parse_predicate(
                f"holding({at_entity.name})", {at_entity.name: at_entity}
            )
            obj_action = PddlAction(
                f"{action.name}_{obj_cat}",
                action._params,
                pre_cond=LogicalExpr(
                    precond.expr_type,
                    precond.sub_exprs,
                    inputs,
                    LogicalQuantifierType.EXISTS,
                ),
                post_cond=[postcond_pred],
            )

            new_acs[obj_action.name] = obj_action
    pddl.set_actions(new_acs)
    return pddl


def get_obj_type(cat, pddl):
    return pddl.expr_types[SimulatorObjectType.MOVABLE_ENTITY.value]


def get_pddl(task_config, all_cats, obj_cats) -> PddlDomain:
    config_path = osp.dirname(__file__) 
    domain_file_path = osp.join(
        config_path,
        'config',
        task_config.task_spec_base_path,
        task_config.pddl_domain_def + ".yaml",
    )
    pddl = PddlDomain(
        domain_file_path,
        task_config,
    )

    # Add permanent entity types. (Object types and receptacles).
    for cat in all_cats:
        if cat in obj_cats:
            obj_type = get_obj_type(cat, pddl)
            entity_type = ExprType(cat, obj_type)
            pddl._expr_types[cat] = entity_type
        else:
            # Assume this is a receptacle in the scene. Permanently place, not per episode.
            pddl._constants[cat] = PddlEntity(cat, pddl.expr_types[PLACABLE_RECEP_TYPE])

    return flatten_actions(pddl, obj_cats)


def get_parser(llm_id):
    if "llama" in llm_id.lower():
        tokenizer = LlamaTokenizer.from_pretrained(llm_id)
        # llama has no pad token by default. As per this thread:
        # https://github.com/huggingface/transformers/issues/22312 we should
        # set pad token manually.
        tokenizer.pad_token = "[PAD]"
        return tokenizer
    else:
        tokenizer = AutoTokenizer.from_pretrained(llm_id)
        tokenizer.pad_token = "[PAD]"
        return tokenizer
