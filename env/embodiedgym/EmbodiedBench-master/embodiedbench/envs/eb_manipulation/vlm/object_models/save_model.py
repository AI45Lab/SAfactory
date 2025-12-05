import os
import json
from pyrep.objects.shape import Shape
from pyrep.pyrep import PyRep

from amsolver.backend.utils import WriteCustomDataBlock, get_local_grasp_pose

class Model_Modifer(object):
    def __init__(self, all_model_dir) -> None:
        super().__init__()
        self.pr = PyRep()
        self.pr.launch('', headless=True)
        self.all_model_dir = all_model_dir

    def import_model(self, model_config):
        models = []
        class_name = model_config["class"]
        object_name = model_config["name"]
        save_path = os.path.join(self.all_model_dir, class_name, object_name)
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        for i, m in enumerate(model_config["parts"]):
            part_name = m["name"]
            part = self.import_shape(m["model_path"], part_name)
            if m["graspable"]:
                WriteCustomDataBlock(part.get_handle(),"graspable","True")
            if i in model_config["manipulated_part"]:
                grasp_mesh_path = os.path.join(save_path, f"{part_name}.ply")
                # m["local_grasp_pose_path"] = grasp_mesh_path.replace('ply','pkl')
                m["local_grasp_pose_path"] = f"{part_name}.pkl"
                self.extra_grasp_poses(part, grasp_mesh_path)
            models.append(part)
        with open(os.path.join(save_path, f"{object_name}.json"), "w") as f:
            json.dump(model_config, f, indent=1)
        need_save_part = models[model_config["highest_part"]]
        need_save_part.save_model(os.path.join(save_path, f"{object_name}.ttm"))
    
    def extra_from_ttm(self, model_config, ttm_path):
        self.pr.import_model(ttm_path)
        class_name = model_config["class"]
        object_name = model_config["name"]
        save_path = os.path.join(self.all_model_dir, class_name, object_name)
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        for i, m in enumerate(model_config["parts"]):
            part = Shape(m["orginal_name"])
            part_name = m["name"]
            part.set_name(part_name)
            # part.compute_mass_and_inertia(1000)
            part.set_renderable(False)
            part.set_respondable(True)
            # part.set_dynamic(True)
            part.set_collidable(True)
            if m["graspable"]:
                WriteCustomDataBlock(part.get_handle(),"graspable","True")
            for children in part.get_objects_in_tree(exclude_base=True, first_generation_only=True):
                if "visible" in children.get_name() or "visual" in children.get_name():
                    children.set_name(part_name+"_visual")
                    children.set_renderable(True)
            if i in model_config["manipulated_part"]:
                if "local_grasp_pose_path" in m:
                    pose_path = os.path.join(save_path, "{}.pkl".format(m["local_grasp_pose_path"]))
                    if os.path.exists(pose_path):
                        m["local_grasp_pose_path"] = "{}.pkl".format(m["local_grasp_pose_path"])
                        continue
                if "local_grasp_pose_path" in m:
                    grasp_mesh_path = os.path.join(save_path, f"{m['local_grasp_pose_path']}.ply")
                    m["local_grasp_pose_path"] = f"{m['local_grasp_pose_path']}.pkl"
                else:
                    grasp_mesh_path = os.path.join(save_path, f"{part_name}.ply")
                    m["local_grasp_pose_path"] = f"{part_name}.pkl"
                self.extra_grasp_poses(part, grasp_mesh_path)

        with open(os.path.join(save_path, f"{object_name}.json"), "w") as f:
            json.dump(model_config, f, indent=1)
        need_save_part = Shape(model_config["parts"][model_config["highest_part"]]["name"])
        # for m in need_save_part.get_objects_in_tree(exclude_base=False):
        #     if "waypoint" in m.get_name():
        #         m.remove()
        if not need_save_part.is_model():
            need_save_part.set_model(True)
        need_save_part.save_model(os.path.join(save_path, f"{object_name}.ttm"))

    @staticmethod
    def extra_grasp_poses(grasp_obj, mesh_path):
        need_rebuild= True if not os.path.exists(mesh_path) else False
        crop_box = None
        for m in grasp_obj.get_objects_in_tree(exclude_base=True, first_generation_only=True):
            if "crop" in m.get_name():
                crop_box = m
                break
        grasp_pose = get_local_grasp_pose(grasp_obj, mesh_path, grasp_pose_path=os.path.dirname(mesh_path),
                need_rebuild = need_rebuild, crop_box=crop_box, use_meshlab=True)
    
    def object_json(self):
        obj_property = {
            "graspable"
        }
        
    def import_shape(self, model_path, name, save_path=None):
        model = Shape.import_shape(model_path, reorient_bounding_box=True)
        try:
            model = self.pr.merge_objects(model.ungroup())
        except:
            pass
        model.set_name(name)
        model_visual = model.copy()
        model_visual.reorient_bounding_box()
        model_visual.set_parent(model)
        model_visual.set_name(name+'_visual')
        model_visual.set_renderable(True)
        model_visual.set_respondable(False)
        model_visual.set_dynamic(False)

        model.set_transparency(0)
        model.get_convex_decomposition(morph=True,individual_meshes=True, use_vhacd=True, vhacd_pca=False)
        model.reorient_bounding_box()
        model.compute_mass_and_inertia(1000)
        model.set_renderable(False)
        model.set_respondable(True)
        model.set_dynamic(True)
        model.set_model(True)
        if save_path is not None:
            model.save_model(os.path.join(self.all_model_dir, f"{name}.ttm"))
        return model

if __name__=="__main__":
    modifer = Model_Modifer("./vlm/object_models")
    model_config = {
        "class": "cube",
        "name": "cube_normal",
        "articulated": False,
        "constraints": None,
        "highest_part":0,
        "manipulated_part":[0],
        "parts":[
            {
                "name": "cube_normal",
                "model_path": "./vlm/object_models/cube.ply",
                "graspable": True,
                "local_grasp_pose_path":None,
                "property":{
                    "shape": "cube",
                    "size": "medium",
                    "color": None,
                    "relative_pos": None
                }
            }
        ]
    }
    model_config_large = {
        "class": "cube",
        "name": "cube_large",
        "articulated": False,
        "constraints": None,
        "highest_part":0,
        "manipulated_part":[0],
        "parts":[
            {
                "name": "cube_large",
                "model_path": "./vlm/object_models/cube_large.ply",
                "graspable": True,
                "local_grasp_pose_path":None,
                "property":{
                    "shape": "cube",
                    "size": "large",
                    "color": None,
                    "relative_pos": None
                }
            }
        ]
    }
    model_config_small = {
        "class": "cube",
        "name": "cube_small",
        "articulated": False,
        "constraints": None,
        "highest_part":0,
        "manipulated_part":[0],
        "parts":[
            {
                "name": "cube_small",
                "model_path": "./vlm/object_models/cube_small.ply",
                "graspable": True,
                "local_grasp_pose_path":None,
                "property":{
                    "shape": "cube",
                    "size": "small",
                    "color": None,
                    "relative_pos": None
                }
            }
        ]
    }
    model_config_door1 = {
        "class": "door",
        "name": "door1",
        "articulated": True,
        "constraints": {
            "door_frame_joint":[0, 1],
            "door_handle_joint":[1, 2]
        },
        "highest_part":0,
        "manipulated_part":[2],
        "parts":[
            {
                "orginal_name":"door_frame",
                "name": "door1_base",
                "graspable": False,
                "property":{
                    "shape": "base of door",
                    "color": None,
                    "size": None,
                    "relative_pos": None
                }
            },
            {
                "orginal_name":"door_main",
                "name": "door1_main",
                "graspable": False,
                "property":{
                    "shape": "door",
                    "color": None,
                    "size": None,
                    "relative_pos": None
                }
            },
            {
                "orginal_name":"open_door_handle",
                "name": "door1_handle",
                "graspable": False,
                "property":{
                    "shape": "handle of door",
                    "color": None,
                    "size": None,
                    "relative_pos": None
                }
            }
        ]
    }
    model_config_door2 = {
        "class": "door",
        "name": "door2",
        "articulated": True,
        "constraints": {
            "Revolute_left_door_joint":[0, 1],
            "Revolute_left_handle_joint":[1, 2]
        },
        "highest_part":0,
        "manipulated_part":[2],
        "parts":[
            {
                "orginal_name":"door2_left_base",
                "name": "door2_base",
                "graspable": False,
                "property":{
                    "shape": "base of door",
                    "color": None,
                    "size": None,
                    "relative_pos": None
                }
            },
            {
                "orginal_name":"door2_left_door",
                "name": "door2_main",
                "graspable": False,
                "property":{
                    "shape": "door",
                    "color": None,
                    "size": None,
                    "relative_pos": None
                }
            },
            {
                "orginal_name":"door2_left_handler",
                "name": "door2_handle",
                "graspable": False,
                "property":{
                    "shape": "handle of door",
                    "color": None,
                    "size": None,
                    "relative_pos": None
                }
            }
        ]
    }
    mc_drawer = {
        "class": "drawer",
        "name": "drawer3",
        "articulated": True,
        "constraints": {
            "top_joint":[0, 1],
            "middle_joint":[0, 2],
            "bottom_joint":[0, 3]
        },
        "highest_part":0,
        "manipulated_part":[1,2,3],
        "parts":[
            {
                "orginal_name":"drawer_base",
                "name": "drawer3_base",
                "graspable": False,
                "property":{
                    "shape": "base of cabinet",
                    "color": None,
                    "size": None,
                    "relative_pos": None
                }
            },
            {
                "orginal_name":"top_drawer",
                "name": "drawer3_top",
                "graspable": False,
                "local_grasp_pose_path": "handle",
                "property":{
                    "shape": "top drawer",
                    "color": None,
                    "size": None,
                    "relative_pos": None
                }
            },
            {
                "orginal_name":"middle_drawer",
                "name": "drawer3_middle",
                "graspable": False,
                "local_grasp_pose_path": "handle",
                "property":{
                    "shape": "middle drawer",
                    "color": None,
                    "size": None,
                    "relative_pos": None
                }
            },
            {
                "orginal_name":"bottom_drawer",
                "name": "drawer3_bottom",
                "graspable": False,
                "local_grasp_pose_path": "handle",
                "property":{
                    "shape": "bottom drawer",
                    "color": None,
                    "size": None,
                    "relative_pos": None
                }
            }
        ]
    }
    star_config = {
        "class": "star",
        "name": "star_normal",
        "articulated": False,
        "constraints": None,
        "highest_part":0,
        "manipulated_part":[0],
        "parts":[
            {
                "orginal_name":"star",
                "name": "star_normal",
                "graspable": True,
                "property":{
                    "shape": "star",
                    "color": None,
                    "size": None,
                    "relative_pos": None
                }
            }
        ]
    }
    moon_config = {
        "class": "moon",
        "name": "moon_normal",
        "articulated": False,
        "constraints": None,
        "highest_part":0,
        "manipulated_part":[0],
        "parts":[
            {
                "orginal_name":"moon",
                "name": "moon_normal",
                "graspable": True,
                "property":{
                    "shape": "moon",
                    "color": None,
                    "size": None,
                    "relative_pos": None
                }
            }
        ]
    }
    triangular_config = {
        "class": "triangular",
        "name": "triangular_normal",
        "articulated": False,
        "constraints": None,
        "highest_part":0,
        "manipulated_part":[0],
        "parts":[
            {
                "orginal_name":"triangular_prism",
                "name": "triangular_normal",
                "graspable": True,
                "property":{
                    "shape": "triangular",
                    "color": None,
                    "size": None,
                    "relative_pos": None
                }
            }
        ]
    }
    cylinder_config = {
        "class": "cylinder",
        "name": "cylinder_normal",
        "articulated": False,
        "constraints": None,
        "highest_part":0,
        "manipulated_part":[0],
        "parts":[
            {
                "orginal_name":"cylinder",
                "name": "cylinder_normal",
                "graspable": True,
                "property":{
                    "shape": "cylinder",
                    "color": None,
                    "size": None,
                    "relative_pos": None
                }
            }
        ]
    }
    cube_basic_config = {
        "class": "cube",
        "name": "cube_basic",
        "articulated": False,
        "constraints": None,
        "highest_part":0,
        "manipulated_part":[0],
        "parts":[
            {
                "orginal_name":"cube",
                "name": "cube_basic",
                "graspable": True,
                "property":{
                    "shape": "cube",
                    "color": None,
                    "size": None,
                    "relative_pos": None
                }
            }
        ]
    }
    mug_config = {
        "class": "mug",
        "name": "mug6",
        "articulated": False,
        "constraints": None,
        "highest_part":0,
        "manipulated_part":[0],
        "parts":[
            {
                "orginal_name":"mug6",
                "name": "mug6",
                "graspable": True,
                "property":{
                    "shape": "mug",
                    "color": None,
                    "size": None,
                    "relative_pos": None
                }
            }
        ]
    }
    mug2_config = {
        "class": "mug",
        "name": "mug2",
        "articulated": False,
        "constraints": None,
        "highest_part":0,
        "manipulated_part":[0],
        "parts":[
            {
                "orginal_name":"mug2",
                "name": "mug2",
                "graspable": True,
                "property":{
                    "shape": "mug",
                    "color": None,
                    "size": None,
                    "relative_pos": None
                }
            }
        ]
    }
    pencile1_config = {
        "class": "pencil",
        "name": "pencil1",
        "articulated": False,
        "constraints": None,
        "highest_part":0,
        "manipulated_part":[0],
        "parts":[
            {
                "orginal_name":"pencil1",
                "name": "pencil1",
                "graspable": True,
                "property":{
                    "shape": "pencil",
                    "color": None,
                    "size": None,
                    "relative_pos": None
                }
            }
        ]
    }
    wiper1_config = {
        "class": "wiper",
        "name": "sponge",
        "articulated": False,
        "constraints": None,
        "highest_part":0,
        "manipulated_part":[0],
        "parts":[
            {
                "orginal_name":"sponge",
                "name": "sponge",
                "graspable": True,
                "property":{
                    "shape": "sponge",
                    "color": None,
                    "size": None,
                    "relative_pos": None
                }
            }
        ]
    }
    letters_config = {
        "class": "letters",
        "name": "letter_v",
        "articulated": False,
        "constraints": None,
        "highest_part":0,
        "manipulated_part":[0],
        "parts":[
            {
                "name": "letter_v",
                "model_path": "./vlm/object_models/letters/letter_v/letter_v.dae",
                "graspable": True,
                "local_grasp_pose_path":None,
                "property":{
                    "shape": "letter of 'v'",
                    "size": "medium",
                    "color": None,
                    "relative_pos": None
                }
            }
        ]
    }
    basics_config = {
        "class": "basic_shapes",
        "name": "flower",
        "articulated": False,
        "constraints": None,
        "highest_part":0,
        "manipulated_part":[0],
        "parts":[
            {
                "name": "flower",
                "model_path": "./vlm/object_models/basic_shapes/flower/flower.dae",
                "graspable": True,
                "local_grasp_pose_path":None,
                "property":{
                    "shape": "flower",
                    "size": "medium",
                    "color": None,
                    "relative_pos": None
                }
            }
        ]
    }
    model_config_fridge = {
        "class": "fridge",
        "name": "fridge5",
        "articulated": True,
        "constraints": {
            "Revolute_joint":[0, 1]
        },
        "highest_part":0,
        "manipulated_part":[1],
        "parts":[
            {
                "orginal_name":"fridge5_base",
                "name": "fridge5_base",
                "graspable": False,
                "property":{
                    "shape": "base of fridge",
                    "color": None,
                    "size": None,
                    "relative_pos": None
                }
            },
            {
                "orginal_name":"fridge5_door",
                "name": "fridge5_door",
                "graspable": False,
                "property":{
                    "shape": "door of fridge",
                    "color": None,
                    "size": None,
                    "relative_pos": None
                }
            },
            # {
            #     "orginal_name":"door_bottom",
            #     "name": "fridge1_bottom_door",
            #     "graspable": False,
            #     "property":{
            #         "shape": "bottom door of fridge",
            #         "color": None,
            #         "size": None,
            #         "relative_pos": None
            #     }
            # }
        ]
    }
    model_config_micro = {
        "class": "microwave",
        "name": "microwave6",
        "articulated": True,
        "constraints": {
            "Revolute_joint":[0, 1]
        },
        "highest_part":0,
        "manipulated_part":[1],
        "parts":[
            {
                "orginal_name":"microwave6_base",
                "name": "microwave6_base",
                "graspable": False,
                "property":{
                    "shape": "base of microwave",
                    "color": None,
                    "size": None,
                    "relative_pos": None
                }
            },
            {
                "orginal_name":"microwave6_door",
                "name": "microwave6_door",
                "graspable": False,
                "property":{
                    "shape": "door of microwave",
                    "color": None,
                    "size": None,
                    "relative_pos": None
                }
            }
        ]
    }
    model_config_grill = {
        "class": "grill",
        "name": "grill1",
        "articulated": True,
        "constraints": {
            "bottom_joint":[0, 1]
        },
        "highest_part":0,
        "manipulated_part":[1],
        "parts":[
            {
                "orginal_name":"grill",
                "name": "grill1_base",
                "graspable": False,
                "property":{
                    "shape": "base of grill",
                    "color": None,
                    "size": None,
                    "relative_pos": None
                }
            },
            {
                "orginal_name":"lid",
                "name": "grill1_lid",
                "graspable": False,
                "property":{
                    "shape": "lid of grill",
                    "color": None,
                    "size": None,
                    "relative_pos": None
                }
            }
        ]
    }
    model_config_cabinet = {
        "class": "cabinet",
        "name": "cabinet5",
        "articulated": True,
        "constraints": {
            "Prismatic_joint":[0, 1],
        },
        "highest_part":0,
        "manipulated_part":[1],
        "parts":[
            {
                "orginal_name":"cabinet5_base",
                "name": "cabinet5_base",
                "graspable": False,
                "property":{
                    "shape": "base of cabinet",
                    "color": None,
                    "size": None,
                    "relative_pos": None
                }
            },
            {
                "orginal_name":"cabinet5_door",
                "name": "cabinet5_door",
                "graspable": False,
                "local_grasp_pose_path": "cabinet5_handle",
                "property":{
                    "shape": "door of cabinet",
                    "color": None,
                    "size": None,
                    "relative_pos": None
                }
            }
        ]
    }
    # modifer.import_model(letters_config)
    modifer.extra_from_ttm(model_config_fridge,"./vlm/object_models/fridge/fridge5_original.ttm")
    modifer.pr.shutdown()
