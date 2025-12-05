#Copy From the rlbench: https://github.com/stepjam/RLBench

seen_colors = [
    ('red', (1.0, 0.0, 0.0)),
    ('maroon', (0.5, 0.0, 0.0)),
    ('lime', (0.0, 1.0, 0.0)),
    ('green', (0.0, 0.5, 0.0)),
    ('blue', (0.0, 0.0, 1.0)),
    ('navy', (0.0, 0.0, 0.5)),
    ('yellow', (1.0, 1.0, 0.0)),
    ('cyan', (0.0, 1.0, 1.0)),
    ('magenta', (1.0, 0.0, 1.0)),
    ('silver', (0.75, 0.75, 0.75)),
    ('gray', (0.5, 0.5, 0.5)),
    ('orange', (1.0, 0.5, 0.0)),
    ('olive', (0.5, 0.5, 0.0)),
    ('purple', (0.5, 0.0, 0.5)),
    ('teal', (0, 0.5, 0.5)),
    ('azure', (0.0, 0.5, 1.0)),
    ('violet', (0.5, 0.0, 1.0)),
    ('rose', (1.0, 0.0, 0.5)),
    ('black', (0.0, 0.0, 0.0)),
    ('white', (1.0, 1.0, 1.0)),
]
#unseen
unseen_colors = [
    ('brown', (0.65, 0.16, 0.16)),
    ('gold', (1.0, 0.65, 0.0)),
    ('pink', (1.0, 0.75, 0.8)),
    ('chocolate', (0.8, 0.4, 0.12)),
    ('coral', (1.0, 0.5, 0.32))
]

#seen objects
seen_object_shapes = {
    "star":{
        "path":"star/star_normal/star_normal.ttm"
    },
    "triangular":{
        "path":"triangular/triangular_normal/triangular_normal.ttm"
    },
    "cylinder":{
        "path":"cylinder/cylinder_normal/cylinder_normal.ttm"
    },
    "cube":{
        "path":"cube/cube_basic/cube_basic.ttm"
    },
    "moon":{
        "path":"moon/moon_normal/moon_normal.ttm"
    },
    # "letter_r":{
    #     "path":"letters/letter_r/letter_r.ttm"
    # },
    # "letter_a":{
    #     "path":"letters/letter_a/letter_a.ttm"
    # },
    # "letter_v":{
    #     "path":"letters/letter_v/letter_v.ttm"
    # },
    # "pentagon":{
    #     "path":"basic_shapes/pentagon/pentagon.ttm"
    # }
}
#unseen objects
unseen_object_shapes = {
    "cube":{
        "path":"cube/cube_basic/cube_basic.ttm"
    },
    "letter_t":{
        "path":"letters/letter_t/letter_t.ttm"
    },
    "cross":{
        "path":"basic_shapes/cross/cross.ttm"
    },
    "flower":{
        "path":"basic_shapes/flower/flower.ttm"
    }
}

seen_planes ={
    "rectangle":{
        "path":"basic_shapes/plane_rectangle.ttm",
        "directional": True
    },
    "round":{
        "path":"basic_shapes/plane_round.ttm",
        "directional": False
    },
    "star":{
        "path":"basic_shapes/plane_star.ttm",
        "directional": False
    },
    "triangle":{
        "path":"basic_shapes/plane_triangle.ttm",
        "directional": True
    }
}

unseen_planes ={
    "cross":{
        "path":"basic_shapes/plane_cross.ttm",
        "directional": True
    },
    "flower":{
        "path":"basic_shapes/plane_flower.ttm",
        "directional": False
    }
}

seen_sorter_objects = {
    "star":{
        "path":"star/star_normal/star_normal.ttm"
    },
    "triangular":{
        "path":"triangular/triangular_normal/triangular_normal.ttm"
    },
    "cylinder":{
        "path":"cylinder/cylinder_normal/cylinder_normal.ttm"
    },
    "cube":{
        "path":"cube/cube_basic/cube_basic.ttm"
    }
}

unseen_sorter_objects = {
    "moon":{
        "path":"moon/moon_normal/moon_normal.ttm"
    },
    "star":{
        "path":"star/star_normal/star_normal.ttm"
    },
    "triangular":{
        "path":"triangular/triangular_normal/triangular_normal.ttm"
    },
    "cylinder":{
        "path":"cylinder/cylinder_normal/cylinder_normal.ttm"
    },
    "cube":{
        "path":"cube/cube_basic/cube_basic.ttm"
    }
}

seen_drawer_list = [
        {
            "path": "drawer/drawer1/drawer1.ttm",
            "max_joint": 0.2
        },
        {
            "path": "drawer/drawer2/drawer2.ttm",
            "max_joint": 0.2
        }
]

unseen_drawer_list = [
        {
            "path": "drawer/drawer3/drawer3.ttm",
            "max_joint": 0.3
        }
]

seen_cabinet_list = [
        {
            "path": "cabinet/cabinet6/cabinet6.ttm",
            "max_joint": 0.19
        },
        {
            "path": "cabinet/cabinet5/cabinet5.ttm",
            "max_joint": 0.32
        }
]

unseen_cabinet_list = [
        {
            "path": "cabinet/cabinet3/cabinet3.ttm",
            "max_joint": 0.19
        }
]

seen_door_list = [
        {
            "path": "fridge/fridge1/fridge1.ttm"
        },
        {
            "path": "fridge/fridge2/fridge2.ttm"
        },
        {
            "path": "microwave/microwave1/microwave1.ttm"
        },
        {
            "path": "microwave/microwave4/microwave4.ttm"
        }
]

unseen_door_list = [
        {
            "path": "microwave/microwave6/microwave6.ttm"
        },
        {
            "path": "fridge/fridge5/fridge5.ttm"
        },
]

seen_complex_door_list = [
        {
            "path": "door/door1/door1.ttm"
        }
]

unseen_complex_door_list = [
        {
            "path": "door/door2/door2.ttm"
        }
]

seen_mug_list = [
    {
        "path": "mug/mug1/mug1.ttm"
    },
    {
        "path": "mug/mug2/mug2.ttm"
    },
    {
        "path": "mug/mug3/mug3.ttm"
    },
    {
        "path": "mug/mug4/mug4.ttm"
    },
]

unseen_mug_list = [
    {
        "path": "mug/mug5/mug5.ttm"
    },
    {
        "path": "mug/mug6/mug6.ttm"
    }
]

colors = seen_colors
object_shapes = seen_object_shapes
planes = seen_planes
sorter_objects = seen_sorter_objects
drawer_list = seen_drawer_list
cabinet_list = seen_cabinet_list
door_list = seen_door_list
mug_list = seen_mug_list
complex_door_list = seen_complex_door_list

"""
if users want to collect unseen data, please comment the above lines and uncomment the below lines. 
colors = unseen_colors
object_shapes = unseen_object_shapes
planes = unseen_planes
sorter_objects = unseen_sorter_objects
drawer_list = unseen_drawer_list
cabinet_list = unseen_cabinet_list
door_list = unseen_door_list
mug_list = unseen_mug_list
complex_door_list = unseen_complex_door_list
"""