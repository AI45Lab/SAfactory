import json
import random
import numpy as np
from ai2thor.controller import Controller
from ai2thor.platform import CloudRendering

# Constants
min_distance = 2.5  # Minimum distance between agent and target object
max_distance = 3.0
DEFAULT_HORIZONS = np.linspace(-30, 60, 30)
DEFAULT_ROTATIONS = range(0, 360, 90)

# controller = Controller(
#         agentMode="default",
#         visibilityDistance=10,
#         scene="FloorPlan1",
#         gridSize=0.25,
#         renderDepthImage=False,
#         renderInstanceSegmentation=False,
#         width=300,
#         height=300,
#         fieldOfView = 90,
#         platform = CloudRendering
#     )

# event = controller.step(action = "GetInteractablePoses", objectId = "Fridge|-02.10|+00.00|+01.07")

def load_scene_object_mapping(filepath):
    with open(filepath, 'r') as f:
        return json.load(f)

def get_object_ids_by_type(metadata, object_type):
    objects = []
    for obj in metadata["objects"]:
        if obj["objectType"] == object_type:
            objects.append(obj["objectId"])
    return objects

def calculate_distance(pos1, pos2):
    return np.sqrt((pos1['x'] - pos2['x'])**2 + (pos1['z'] - pos2['z'])**2)

def get_valid_pose(controller, target_object_id):

    event = controller.step(
        action="GetInteractablePoses",
        objectId=target_object_id,
        horizons=[0]
        # rotations=DEFAULT_ROTATIONS,
        # standings=[True]
    )

    print(len(event.metadata["actionReturn"]))
    
    # Get object position
    obj_metadata = next(obj for obj in controller.last_event.metadata["objects"] 
                       if obj["objectId"] == target_object_id)
    obj_position = {
        'x': obj_metadata['position']['x'],
        'z': obj_metadata['position']['z']
    }
    
    valid_poses = []
    poses = event.metadata["actionReturn"]
    random.shuffle(poses)
    for pose in poses:
        pos = {'x': pose['x'], 'z': pose['z']}
        if calculate_distance(pos, obj_position) >= min_distance and calculate_distance(pos, obj_position) <= max_distance:
            return pose
            # valid_poses.append(pose)
    
    return None
    #random.choice(valid_poses) if valid_poses else None

def generate_dataset(mapping_filepath, output_filepath):
    """Generate the complete navigation dataset."""
    # Initialize controller
    controller = Controller(
        agentMode="default",
        visibilityDistance=5,
        scene="FloorPlan1",
        gridSize=0.1,
        renderDepthImage=False,
        renderInstanceSegmentation=False,
        width=300,
        height=300,
        fieldOfView = 90,
        platform = CloudRendering
    )
    
    scene_mapping = load_scene_object_mapping(mapping_filepath)
    tasks = []
    
    for scene, target_type in scene_mapping.items():
        # Initialize scene
        controller.reset(scene=scene)
        
        # Get all objects of target type
        target_objects = get_object_ids_by_type(controller.last_event.metadata, target_type)
        
        if not target_objects:
            print(f"Warning: No {target_type} found in {scene}")
            continue
            
        # Select first target object
        target_object_id = target_objects[0]

        for obj in controller.last_event.metadata["objects"]:
            if obj["objectId"] == target_object_id:
                target_position = obj["position"]
                break
        
        # Get other objects to hide (all objects of same type except the target)
        objects_to_hide = target_objects[1:] if len(target_objects) > 1 else []

        # print(target_object_id)
        
        # Get valid initial pose
        pose = get_valid_pose(controller, target_object_id)
        if not pose:
            print(f"Warning: Could not find valid pose in {scene}")
            continue
            
        # Create task entry
        task = {
            "targetObjectType": target_type,
            "targetObjectIds": target_object_id,
            "target_position": target_position,
            "agentPose": {
                "position": {
                    "x": pose["x"],
                    "y": pose["y"],
                    "z": pose["z"]
                },
                "rotation": pose["rotation"],
                "horizon": pose["horizon"]
            },
            "scene": scene,
            "object_to_hide": objects_to_hide,
            "instruction": f"navigate to the {target_type} in the room and be as close as possible to it"
        }
        
        tasks.append(task)
    
    # Save dataset
    dataset = {"tasks": tasks}
    with open(output_filepath, 'w') as f:
        json.dump(dataset, f, indent=4)
    
    print(f"Generated dataset with {len(tasks)} tasks")
    return dataset

if __name__ == "__main__":
    mapping_filepath = "FloorPlan.json"  # Your input JSON file
    output_filepath = "base_navigation_new.json"     # Output dataset file
    dataset = generate_dataset(mapping_filepath, output_filepath)