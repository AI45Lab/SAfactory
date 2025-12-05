# examples for pick_cube_shape (2-shot)
pick_examples_vlm = [
"""Human Instruction: Pick up the star and place it into the yellow container.
Input: {'object 1': [45, 13, 18], 'object 2': [72, 20, 18], 'object 3': [50, 42, 17], 'object 4': [36, 42, 18], 'object 5': [70, 43, 17], 'object 6': [50, 77, 17]}
Output: {
    "visual_state_description": "From left to right, I can see a purple cylinder at [45, 13, 18], a blue moon at [72, 20, 18], a green star at [50, 42, 17], a purple cube at [36, 42, 18], a yellow container at [70, 43, 17] and a purple container at [50, 77, 17].",
    "reasoning_and_reflection": "The task is to pick up the star located at [50, 42, 17] and place it into the yellow container located at [70, 43, 17]. The plan involves moving the gripper to the star's position, closing the gripper to pick it up, moving to the yellow container's position, and opening the gripper to release the cube into the container.",
    "executable_plan": "[[50, 42, 17, 6, 61, 36, 1], [50, 42, 17, 6, 61, 36, 0], [47, 41, 26, 6, 61, 36, 0], [69, 39, 24, 88, 32, 77, 0], [69, 39, 24, 87, 32, 77, 1]]"
}""",
"""Human Instruction: Pick up the star and place it into the rose container.
Input: {'object 1': [78, 20, 17], 'object 2': [28, 30, 17], 'object 3': [39, 47, 17], 'object 4': [50, 72, 17], 'object 5': [39, 75, 18]}
Output: {
    "visual_state_description": "From left to right, I can see a rose container at [78, 20, 17], a black star at [28, 30, 17], a green container at [39, 47, 17], a green triangular prism at [50, 72, 17] and a grey cylinder at [39, 75, 18].",
    "reasoning_and_reflection": "The task is to pick up the star located at [28, 30, 17] and place it into the rose container located at [78, 20, 17]. The plan involves moving the gripper to the star's position, closing the gripper to pick it up, moving to the rose container's position, and opening the gripper to release the cube into the container.",
    "executable_plan": "[[28, 32, 26, 0, 60, 94, 1], [27, 32, 18, 0, 60, 94, 0], [27, 32, 28, 0, 60, 94, 0], [76, 19, 28, 6, 66, 109, 1]]"
}"""
]

pick_examples_llm = [
"""Human Instruction: Pick up the star and place it into the yellow container.
Input: {'object 1': [45, 13, 18], 'object 2': [72, 20, 18], 'object 3': [50, 42, 17], 'object 4': [36, 42, 18], 'object 5': [70, 43, 17], 'object 6': [50, 77, 17]}
Output: {
    "reasoning_and_reflection": "The task is to pick up the star located at [50, 42, 17] and place it into the yellow container located at [70, 43, 17]. The plan involves moving the gripper to the star's position, closing the gripper to pick it up, moving to the yellow container's position, and opening the gripper to release the cube into the container.",
    "executable_plan": "[[50, 42, 17, 6, 61, 36, 1], [50, 42, 17, 6, 61, 36, 0], [47, 41, 26, 6, 61, 36, 0], [69, 39, 24, 88, 32, 77, 0], [69, 39, 24, 87, 32, 77, 1]]"
}""",
"""Human Instruction: Pick up the star and place it into the rose container.
Input: {'object 1': [78, 20, 17], 'object 2': [28, 30, 17], 'object 3': [39, 47, 17], 'object 4': [50, 72, 17], 'object 5': [39, 75, 18]}
Output: {
    "reasoning_and_reflection": "The task is to pick up the star located at [28, 30, 17] and place it into the rose container located at [78, 20, 17]. The plan involves moving the gripper to the star's position, closing the gripper to pick it up, moving to the rose container's position, and opening the gripper to release the cube into the container.",
    "executable_plan": "[[28, 32, 26, 0, 60, 94, 1], [27, 32, 18, 0, 60, 94, 0], [27, 32, 28, 0, 60, 94, 0], [76, 19, 28, 6, 66, 109, 1]]"
}"""
]

# examples for stack_cubes_color (4-shot)
stack_examples_vlm = [
"""Human Instruction: Stack the red star and the olive star in sequence.
Input: {'object 1': [42, 24, 19], 'object 2': [54, 53, 19], 'object 3': [65, 68, 19], 'object 4': [39, 73, 18]}
Output: {
    "visual_state_description": "From left to right, I can see a purple star at [42, 24, 19], an olive star at [54, 53, 19], a grey star at [65, 68, 19] and a red star at [39, 73, 18].",
    "reasoning_and_reflection": "I need to understand the instruction first. To stack the objects in sequece, the red star at [39, 73, 18] should be at the bottom and the olive star at [54, 53, 19] should be placed on top. The plan involves moving the gripper to the olive star's position, closing the gripper to pick it up, moving to the top of the red star, and opening the gripper to release the olive star onto the red star.",
    "executable_plan": "[[54, 51, 28, 3, 66, 37, 1], [54, 54, 20, 3, 66, 37, 0], [54, 51, 30, 3, 66, 37, 0], [39, 74, 29, 3, 66, 7, 0], [39, 74, 26, 3, 66, 7, 1]]"
}""",
"""Human Instruction: Stack the red cube and the blue cube in sequence.
Input: {'object 1': [35, 17, 17], 'object 2': [57, 47, 17], 'object 3': [33, 76, 17], 'object 4': [77, 86, 17]}
Output: {
    "visual_state_description": "From left to right, I can see a blue cube at [35, 17, 17], a black cube at [57, 47, 17], a red cube at [33, 76, 17] and a teal cube at [77, 86, 17].",
    "reasoning_and_reflection": "I need to understand the instruction first. To stack the objects in sequece, the red cube at [33, 76, 17] should be at the bottom and the blue cube at [35, 17, 17] should be placed on top. The plan involves moving the gripper to the blue cube's position, closing the gripper to pick it up, moving to the top of the red cube, and opening the gripper to release the blue cube onto the red cube.",
    "executable_plan": "[[36, 16, 25, 0, 60, 34, 1], [36, 16, 17, 0, 60, 34, 0], [36, 16, 27, 0, 60, 34, 0], [32, 76, 24, 0, 60, 104, 0], [32, 76, 21, 0, 60, 104, 1]]"
}""",
"""Human Instruction: Stack the red moon and the maroon moon in sequence.
Input: {'object 1': [67, 43, 19], 'object 2': [80, 47, 19], 'object 3': [34, 58, 20], 'object 4': [84, 68, 19]}
Output: {
    "visual_state_description": "From left to right, I can see a yellow moon at [67, 43, 19], a red moon at [80, 47, 19], a maroon moon at [34, 58, 20] and a blue moon at [84, 68, 19].",
    "reasoning_and_reflection": "I need to understand the instruction first. To stack the objects in sequece, the red moon at [80, 47, 19] should be at the bottom and the maroon moon at [34, 58, 20] should be placed on top. The plan involves moving the gripper to the maroon moon's position, closing the gripper to pick it up, moving to the top of the red moon, and opening the gripper to release the maroon moon onto the red moon.",
    "executable_plan": "[[34, 58, 28, 0, 60, 25, 1], [34, 58, 20, 0, 60, 25, 0], [34, 58, 30, 0, 60, 25, 0], [83, 46, 30, 0, 60, 15, 0], [83, 46, 26, 0, 60, 15, 1]]"
}""",
"""Human Instruction: Stack the red triangular prism and the magenta triangular prism in sequence.
Input: {'object 1': [78, 20, 19], 'object 2': [50, 25, 20], 'object 3': [55, 72, 19], 'object 4': [56, 86, 19]}
Output: {
    "visual_state_description": "From left to right, I can see a blue triangular prism at [78, 20, 19], a magenta triangular prism at [50, 25, 20], a green triangular prism at [55, 72, 19] and a red triangular prism at [56, 86, 19].",
    "reasoning_and_reflection": "I need to understand the instruction first. To stack the objects in sequece, the red triangular prism at [56, 86, 19] should be at the bottom and the magenta triangular prism at [50, 25, 20] should be placed on top. The plan involves moving the gripper to the magenta triangular prism's position, closing the gripper to pick it up, moving to the top of the red triangular prism, and opening the gripper to release the magenta triangular prism onto the red triangular prism.",
    "executable_plan": "[[48, 25, 29, 1, 60, 92, 1], [48, 25, 21, 1, 60, 92, 0], [49, 25, 31, 1, 60, 92, 0], [54, 87, 31, 1, 60, 96, 0], [54, 87, 28, 1, 60, 96, 1]]"
}"""
]

stack_examples_llm = [
"""Human Instruction: Stack the red star and the olive star in sequence.
Input: {'object 1': [42, 24, 19], 'object 2': [54, 53, 19], 'object 3': [65, 68, 19], 'object 4': [39, 73, 18]}
Output: {
    "reasoning_and_reflection": "I need to understand the instruction first. To stack the objects in sequece, the red star at [39, 73, 18] should be at the bottom and the olive star at [54, 53, 19] should be placed on top. The plan involves moving the gripper to the olive star's position, closing the gripper to pick it up, moving to the top of the red star, and opening the gripper to release the olive star onto the red star.",
    "executable_plan": "[[54, 51, 28, 3, 66, 37, 1], [54, 54, 20, 3, 66, 37, 0], [54, 51, 30, 3, 66, 37, 0], [39, 74, 29, 3, 66, 7, 0], [39, 74, 26, 3, 66, 7, 1]]"
}""",
"""Human Instruction: Stack the red cube and the blue cube in sequence.
Input: {'object 1': [35, 17, 17], 'object 2': [57, 47, 17], 'object 3': [33, 76, 17], 'object 4': [77, 86, 17]}
Output: {
    "reasoning_and_reflection": "I need to understand the instruction first. To stack the objects in sequece, the red cube at [33, 76, 17] should be at the bottom and the blue cube at [35, 17, 17] should be placed on top. The plan involves moving the gripper to the blue cube's position, closing the gripper to pick it up, moving to the top of the red cube, and opening the gripper to release the blue cube onto the red cube.",
    "executable_plan": "[[36, 16, 25, 0, 60, 34, 1], [36, 16, 17, 0, 60, 34, 0], [36, 16, 27, 0, 60, 34, 0], [32, 76, 27, 0, 60, 104, 0], [32, 76, 20, 0, 60, 104, 1]]"
}""",
"""Human Instruction: Stack the red moon and the maroon moon in sequence.
Input: {'object 1': [67, 43, 19], 'object 2': [80, 47, 19], 'object 3': [34, 58, 20], 'object 4': [84, 68, 19]}
Output: {
    "reasoning_and_reflection": "I need to understand the instruction first. To stack the objects in sequece, the red moon at [80, 47, 19] should be at the bottom and the maroon moon at [34, 58, 20] should be placed on top. The plan involves moving the gripper to the maroon moon's position, closing the gripper to pick it up, moving to the top of the red moon, and opening the gripper to release the maroon moon onto the red moon.",
    "executable_plan": "[[34, 58, 28, 0, 60, 25, 1], [34, 58, 20, 0, 60, 25, 0], [34, 58, 30, 0, 60, 25, 0], [83, 46, 30, 0, 60, 15, 0], [83, 46, 26, 0, 60, 15, 1]]"
}""",
"""Human Instruction: Stack the red triangular prism and the magenta triangular prism in sequence.
Input: {'object 1': [78, 20, 19], 'object 2': [50, 25, 20], 'object 3': [55, 72, 19], 'object 4': [56, 86, 19]}
Output: {
    "reasoning_and_reflection": "I need to understand the instruction first. To stack the objects in sequece, the red triangular prism at [56, 86, 19] should be at the bottom and the magenta triangular prism at [50, 25, 20] should be placed on top. The plan involves moving the gripper to the magenta triangular prism's position, closing the gripper to pick it up, moving to the top of the red triangular prism, and opening the gripper to release the magenta triangular prism onto the red triangular prism.",
    "executable_plan": "[[48, 25, 29, 1, 60, 92, 1], [48, 25, 21, 1, 60, 92, 0], [49, 25, 31, 1, 60, 92, 0], [54, 87, 31, 1, 60, 96, 0], [54, 87, 28, 1, 60, 96, 1]]"
}"""
]

# examples for place_into_shape_sorter_color (4-shot)
shape_sorter_examples_vlm = [
"""Human Instruction: Put the red star into the shape sorter. 
Input: {'object 1': [65, 11, 17], 'object 2': [50, 38, 17], 'object 3': [40, 64, 37], 'object 4': [45, 83, 17]}
Output: {
    "visual_state_description": "From left to right, I can see a purple star at [65, 11, 17], a red star at [50, 38, 17], a shape sorter at [40, 64, 37] and a green star at [45, 83, 17].",
    "reasoning_and_reflection": "I need to identify the location of the star hole on the shape sorter first. I can see that the star hole is at the center of the shape sorter, which is at [40, 64, 37]. Then, the task is to pick up the red star at [50, 38, 17] and put it into the star hole on the shape sorter. The plan involves moving the gripper to the star's position, closing the gripper to pick it up, moving upwards, moving to the top of the star hole, and releasing the star into the hole.",
    "executable_plan": "[[51, 38, 25, 119, 59, 26, 1], [51, 38, 17, 119, 59, 26, 0], [51, 38, 27, 119, 59, 26, 0], [40, 65, 49, 0, 60, 45, 0], [40, 65, 41, 0, 60, 45, 1]]"
}""",
"""Human Instruction: Put the maroon star into the shape sorter. 
Input: {'object 1': [35, 30, 17], 'object 2': [29, 38, 17], 'object 3': [53, 38, 17], 'object 4': [64, 58, 37]}
Output: {
    "visual_state_description": "From left to right, I can see a purple star at [35, 30, 17], a maroon star at [29, 38, 17], a green star at [53, 38, 17] and a shape sorter at [64, 58, 37].",
    "reasoning_and_reflection": "I need to identify the location of the star hole on the shape sorter first. I can see that the star hole is at the center of the shape sorter, which is at [64, 58, 37]. Then, the task is to pick up the maroon star at [29, 38, 17] and put it into the star hole on the shape sorter. The plan involves moving the gripper to the star's position, closing the gripper to pick it up, moving upwards, moving to the top of the star hole, and releasing the star into the hole.",
    "executable_plan": "[[29, 39, 24, 1, 61, 109, 1], [29, 38, 16, 1, 61, 109, 0], [30, 40, 26, 1, 61, 109, 0], [65, 60, 48, 0, 61, 54, 0], [65, 60, 40, 0, 61, 54, 1]]"
}""",
"""Human Instruction: Put the lime star into the shape sorter. 
Input: {'object 1': [47, 13, 17], 'object 2': [27, 36, 16], 'object 3': [38, 50, 17], 'object 4': [54, 69, 37]}
Output: {
    "visual_state_description": "From left to right, I can see a grey star at [47, 13, 17], a purple star at [27, 36, 16], a lime star at [38, 50, 17] and a shape sorter at [54, 69, 37].",
    "reasoning_and_reflection": "I need to identify the location of the star hole on the shape sorter first. I can see that the star hole is at the center of the shape sorter, which is at [54, 69, 37]. Then, the task is to pick up the lime star at [38, 50, 17] and put it into the star hole on the shape sorter. The plan involves moving the gripper to the star's position, closing the gripper to pick it up, moving upwards, moving to the top of the star hole, and releasing the star into the hole.",
    "executable_plan": "[[38, 49, 23, 0, 60, 108, 1], [38, 49, 15, 0, 60, 108, 0], [38, 49, 25, 0, 60, 108, 0], [54, 69, 47, 0, 60, 3, 0], [54, 69, 39, 0, 60, 3, 1]]"
}""",
"""Human Instruction: Put the green star into the shape sorter. 
Input: {'object 1': [70, 27, 37], 'object 2': [66, 59, 17], 'object 3': [38, 63, 17], 'object 4': [63, 77, 17]}
Output: {
    "visual_state_description": "From left to right, I can see a shape sorter at [70, 27, 37], a white star at [66, 59, 17], a green star at [38, 63, 17] and a white star at [63, 77, 17].",
    "reasoning_and_reflection": "I need to identify the location of the star hole on the shape sorter first. I can see that the star hole is at the center of the shape sorter, which is at [70, 27, 37]. Then, the task is to pick up the green star at [38, 63, 17] and put it into the star hole on the shape sorter. The plan involves moving the gripper to the star's position, closing the gripper to pick it up, moving upwards, moving to the top of the star hole, and releasing the star into the hole.",
    "executable_plan": "[[39, 62, 23, 0, 60, 100, 1], [39, 62, 15, 0, 60, 100, 0], [39, 62, 25, 0, 60, 100, 0], [71, 30, 47, 0, 60, 37, 0], [71, 30, 39, 0, 60, 37, 1]]"
}"""
]

shape_sorter_examples_llm = [
"""Human Instruction: Put the red star into the shape sorter. 
Input: {'object 1': [65, 11, 17], 'object 2': [50, 38, 17], 'object 3': [40, 64, 37], 'object 4': [45, 83, 17]}
Output: {
    "reasoning_and_reflection": "I need to identify the location of the star hole on the shape sorter first. I can see that the star hole is at the center of the shape sorter, which is at [40, 64, 37]. Then, the task is to pick up the red star at [50, 38, 17] and put it into the star hole on the shape sorter. The plan involves moving the gripper to the star's position, closing the gripper to pick it up, moving upwards, moving to the top of the star hole, and releasing the star into the hole.",
    "executable_plan": "[[51, 38, 25, 119, 59, 26, 1], [51, 38, 17, 119, 59, 26, 0], [51, 38, 27, 119, 59, 26, 0], [40, 65, 49, 0, 60, 45, 0], [40, 65, 41, 0, 60, 45, 1]]"
}""",
"""Human Instruction: Put the maroon star into the shape sorter. 
Input: {'object 1': [35, 30, 17], 'object 2': [29, 38, 17], 'object 3': [53, 38, 17], 'object 4': [64, 58, 37]}
Output: {
    "reasoning_and_reflection": "I need to identify the location of the star hole on the shape sorter first. I can see that the star hole is at the center of the shape sorter, which is at [64, 58, 37]. Then, the task is to pick up the maroon star at [29, 38, 17] and put it into the star hole on the shape sorter. The plan involves moving the gripper to the star's position, closing the gripper to pick it up, moving upwards, moving to the top of the star hole, and releasing the star into the hole.",
    "executable_plan": "[[29, 39, 24, 1, 61, 109, 1], [29, 38, 16, 1, 61, 109, 0], [30, 40, 26, 1, 61, 109, 0], [65, 60, 48, 0, 61, 54, 0], [65, 60, 40, 0, 61, 54, 1]]"
}""",
"""Human Instruction: Put the lime star into the shape sorter. 
Input: {'object 1': [47, 13, 17], 'object 2': [27, 36, 16], 'object 3': [38, 50, 17], 'object 4': [54, 69, 37]}
Output: {
    "reasoning_and_reflection": "I need to identify the location of the star hole on the shape sorter first. I can see that the star hole is at the center of the shape sorter, which is at [54, 69, 37]. Then, the task is to pick up the lime star at [38, 50, 17] and put it into the star hole on the shape sorter. The plan involves moving the gripper to the star's position, closing the gripper to pick it up, moving upwards, moving to the top of the star hole, and releasing the star into the hole.",
    "executable_plan": "[[38, 49, 23, 0, 60, 108, 1], [38, 49, 15, 0, 60, 108, 0], [38, 49, 25, 0, 60, 108, 0], [54, 69, 47, 0, 60, 3, 0], [54, 69, 39, 0, 60, 3, 1]]"
}""",
"""Human Instruction: Put the green star into the shape sorter. 
Input: {'object 1': [70, 27, 37], 'object 2': [66, 59, 17], 'object 3': [38, 63, 17], 'object 4': [63, 77, 17]}
Output: {
    "reasoning_and_reflection": "I need to identify the location of the star hole on the shape sorter first. I can see that the star hole is at the center of the shape sorter, which is at [70, 27, 37]. Then, the task is to pick up the green star at [38, 63, 17] and put it into the star hole on the shape sorter. The plan involves moving the gripper to the star's position, closing the gripper to pick it up, moving upwards, moving to the top of the star hole, and releasing the star into the hole.",
    "executable_plan": "[[39, 62, 23, 0, 60, 100, 1], [39, 62, 15, 0, 60, 100, 0], [39, 62, 25, 0, 60, 100, 0], [71, 30, 47, 0, 60, 37, 0], [71, 30, 39, 0, 60, 37, 1]]"
}"""
]

# examples for wipe_table_direction (8-shot)
wipe_examples_vlm = [
"""Human Instruction: Wipe the horizontal area.
Input: {'object 1': [33, 36, 19], 'object 2': [52, 38, 15], 'object 3': [59, 68, 15]}
Output: {
    "visual_state_description": "From left to right, I can see a yellow sponge at [33, 36, 19], a green rectangle area at [52, 38, 15] and an orange rectangle area at [59, 68, 15].",
    "reasoning_and_reflection": "I need to identify which area is horizontal first. I can see that the orange rectangle area at [59, 68, 15] is horizontal since it is parallel to the green arrow representing the y-axis of the coordinate system annotated in the image. Then, the task is to pick up the sponge at [33, 36, 19] and use it to wipe the horizontal area at [59, 68, 15]. The plan involves moving the gripper to the sponge's position, closing the gripper to pick it up, moving to the side of the horizontal area, and moving the sponge along the main direction of the horizontal area to wipe it.",
    "executable_plan": "[[32, 34, 25, 0, 60, 34, 1], [32, 34, 17, 0, 60, 34, 0], [32, 34, 27, 0, 60, 34, 0], [60, 80, 18, 0, 61, 31, 0], [61, 54, 17, 0, 61, 31, 0]]"
}""",
"""Human Instruction: Wipe the horizontal area.
Input: {'object 1': [56, 19, 15], 'object 2': [35, 41, 19], 'object 3': [54, 47, 15]}
Output: {
    "visual_state_description": "From left to right, I can see a black rectangle area at [56, 19, 15], a yellow sponge at [35, 41, 19] and a white rectangle area at [54, 47, 15].",
    "reasoning_and_reflection": "I need to identify which area is horizontal first. I can see that the black rectangle area at [56, 19, 15] is horizontal since it is parallel to the green arrow representing the y-axis of the coordinate system annotated in the image. Then, the task is to pick up the sponge at [35, 41, 19] and use it to wipe the horizontal area at [56, 19, 15]. The plan involves moving the gripper to the sponge's position, closing the gripper to pick it up, moving to the right side of the horizontal area, and moving the sponge along the main direction of the horizontal area to wipe it.",
    "executable_plan": "[[33, 43, 25, 0, 60, 32, 1], [33, 43, 17, 0, 60, 32, 0], [33, 43, 27, 0, 60, 32, 0], [58, 35, 18, 0, 60, 28, 0], [56, 8, 18, 0, 60, 28, 0]]"
}""",
"""Human Instruction: Wipe the horizontal area.
Input: {'object 1': [35, 33, 19], 'object 2': [51, 41, 15], 'object 3': [67, 63, 15]}
Output: {
    "visual_state_description": "From left to right, I can see a yellow sponge at [35, 33, 19], a yellow triangle area at [51, 41, 15] and a blue triangle area at [67, 63, 15].",
    "reasoning_and_reflection": "I need to identify which area is horizontal first. I can see that the yellow triangle area at [51, 41, 15] is horizontal since it is parallel to the green arrow representing the y-axis of the coordinate system annotated in the image. Then, the task is to pick up the sponge at [35, 33, 19] and use it to wipe the horizontal area at [51, 41, 15]. The plan involves moving the gripper to the sponge's position, closing the gripper to pick it up, moving to the right side of the horizontal area, and moving the sponge along the main direction of the horizontal area to wipe it.",
    "executable_plan": "[[35, 34, 25, 0, 60, 20, 1], [35, 34, 17, 0, 60, 20, 0], [35, 34, 27, 0, 60, 20, 0], [52, 47, 18, 0, 60, 84, 0], [44, 19, 18, 0, 60, 84, 0]]"
}""",
"""Human Instruction: Wipe the horizontal area.
Input: {'object 1': [47, 34, 18], 'object 2': [72, 53, 15], 'object 3': [54, 81, 15]}
Output: {
    "visual_state_description": "From left to right, I can see a yellow sponge at [47, 34, 18], a yellow triangle area at [72, 53, 15] and a green triangle area at [54, 81, 15].",
    "reasoning_and_reflection": "I need to identify which area is horizontal first. I can see that the yellow triangle area at [72, 53, 15] is horizontal since it is parallel to the green arrow representing the y-axis of the coordinate system annotated in the image. Then, the task is to pick up the sponge at [47, 34, 18] and use it to wipe the horizontal area at [72, 53, 15]. The plan involves moving the gripper to the sponge's position, closing the gripper to pick it up, moving to the right side of the horizontal area, and moving the sponge along the main direction of the horizontal area to wipe it.",
    "executable_plan": "[[46, 31, 24, 0, 60, 18, 1], [46, 31, 16, 0, 60, 18, 0], [46, 31, 26, 0, 60, 18, 0], [72, 63, 17, 0, 60, 88, 0], [69, 37, 17, 0, 60, 88, 0]]"
}""",
"""Human Instruction: Wipe the vertical area.
Input: "{'object 1': [55, 28, 15], 'object 2': [42, 51, 19], 'object 3': [32, 63, 15]}"
Output: {
    "visual_state_description": "From left to right, I can see a white rectangle area at [55, 28, 15], a yellow sponge at [42, 51, 19] and a blue rectangle area at [32, 63, 15].",
    "reasoning_and_reflection": "I need to identify which area is vertical first. I can see that the white rectangle area at [55, 28, 15] is vertical since it is parallel to the red arrow representing the x-axis of the coordinate system annotated in the image. Then, the task is to pick up the sponge at [42, 51, 19] and use it to wipe the vertical area at [55, 28, 15]. The plan involves moving the gripper to the sponge's position, closing the gripper to pick it up, moving to the top side of the vertical area, and moving the sponge along the main direction of the vertical area to wipe it.",
    "executable_plan": "[[40, 48, 25, 0, 60, 12, 1], [40, 48, 17, 0, 60, 12, 0], [40, 48, 27, 0, 60, 12, 0], [38, 25, 17, 0, 59, 0, 0], [69, 25, 17, 0, 59, 0, 0]]"
}""",
"""Human Instruction: Wipe the vertical area.
Input: {'object 1': [53, 31, 19], 'object 2': [66, 34, 15], 'object 3': [59, 73, 15]}
Output: {
    "visual_state_description": "From left to right, I can see a yellow sponge at [53, 31, 19], a blue rectangle area at [66, 34, 15] and a green rectangle area at [59, 73, 15].",
    "reasoning_and_reflection": "I need to identify which area is vertical first. I can see that the blue rectangle area at [66, 34, 15] is vertical since it is parallel to the red arrow representing the x-axis of the coordinate system annotated in the image. Then, the task is to pick up the sponge at [53, 31, 19] and use it to wipe the vertical area at [66, 34, 15]. The plan involves moving the gripper to the sponge's position, closing the gripper to pick it up, moving to the top side of the vertical area, and moving the sponge along the main direction of the vertical area to wipe it.",
    "executable_plan": "[[52, 28, 26, 0, 60, 94, 1], [52, 28, 18, 0, 60, 94, 0], [52, 28, 28, 0, 60, 94, 0]], [46, 32, 18, 0, 60, 68, 0], [78, 35, 18, 0, 60, 75, 0]]"
}""",
"""Human Instruction: Wipe the vertical area.
Input: {'object 1': [37, 26, 15], 'object 2': [57, 49, 15], 'object 3': [56, 73, 18]}
Output: {
    "visual_state_description": "From left to right, I can see a white triangle area at [37, 26, 15], a black triangle area at [57, 49, 15] and a yellow sponge at [56, 73, 18].",
    "reasoning_and_reflection": "I need to identify which area is vertical first. I can see that the white triangle area at [37, 26, 15] is vertical since it is parallel to the red arrow representing the x-axis of the coordinate system annotated in the image. Then, the task is to pick up the sponge at [56, 73, 18] and use it to wipe the vertical area at [37, 26, 15]. The plan involves moving the gripper to the sponge's position, closing the gripper to pick it up, moving to the top side of the vertical area, and moving the sponge along the main direction of the vertical area to wipe it.",
    "executable_plan": "[[55, 70, 25, 0, 60, 26, 1], [55, 70, 17, 0, 60, 26, 0], [55, 70, 27, 0, 60, 26, 0], [35, 26, 18, 0, 60, 55, 0], [62, 19, 18, 0, 60, 55, 0]]"
}""",
"""Human Instruction: Wipe the vertical area.
Input: {'object 1': [64, 21, 15], 'object 2': [34, 39, 15], 'object 3': [51, 43, 19]}
Output: {
    "visual_state_description": "From left to right, I can see a green triangle area at [64, 21, 15], a red triangle area at [34, 39, 15] and a yellow sponge at [51, 43, 19].",
    "reasoning_and_reflection": "I need to identify which area is vertical first. I can see that the green triangle area at [64, 21, 15] is vertical since it is parallel to the red arrow representing the x-axis of the coordinate system annotated in the image. Then, the task is to pick up the sponge at [51, 43, 19] and use it to wipe the vertical area at [64, 21, 15]. The plan involves moving the gripper to the sponge's position, closing the gripper to pick it up, moving to the top side of the vertical area, and moving the sponge along the main direction of the vertical area to wipe it.",
    "executable_plan": "[[44, 46, 25, 116, 62, 97, 1], [45, 45, 17, 116, 62, 97, 0], [43, 46, 27, 116, 62, 97, 0], [50, 13, 18, 115, 60, 13, 0], [75, 26, 18, 115, 60, 99, 0]]"
}"""
]

wipe_examples_llm = [
"""Human Instruction: Wipe the horizontal area.
Input: {'object 1': [33, 36, 19], 'object 2': [52, 38, 15], 'object 3': [59, 68, 15]}
Output: {
    "reasoning_and_reflection": "I need to identify which area is horizontal first. I can see that the orange rectangle area at [59, 68, 15] is horizontal since it is parallel to the green arrow representing the y-axis of the coordinate system annotated in the image. Then, the task is to pick up the sponge at [33, 36, 19] and use it to wipe the horizontal area at [59, 68, 15]. The plan involves moving the gripper to the sponge's position, closing the gripper to pick it up, moving to the side of the horizontal area, and moving the sponge along the main direction of the horizontal area to wipe it.",
    "executable_plan": "[[32, 34, 25, 0, 60, 34, 1], [32, 34, 17, 0, 60, 34, 0], [32, 34, 27, 0, 60, 34, 0], [60, 80, 18, 0, 61, 31, 0], [61, 54, 17, 0, 61, 31, 0]]"
}""",
"""Human Instruction: Wipe the horizontal area.
Input: {'object 1': [56, 19, 15], 'object 2': [35, 41, 19], 'object 3': [54, 47, 15]}
Output: {
    "reasoning_and_reflection": "I need to identify which area is horizontal first. I can see that the black rectangle area at [56, 19, 15] is horizontal since it is parallel to the green arrow representing the y-axis of the coordinate system annotated in the image. Then, the task is to pick up the sponge at [35, 41, 19] and use it to wipe the horizontal area at [56, 19, 15]. The plan involves moving the gripper to the sponge's position, closing the gripper to pick it up, moving to the right side of the horizontal area, and moving the sponge along the main direction of the horizontal area to wipe it.",
    "executable_plan": "[[33, 43, 25, 0, 60, 32, 1], [33, 43, 17, 0, 60, 32, 0], [33, 43, 27, 0, 60, 32, 0], [58, 35, 18, 0, 60, 28, 0], [56, 8, 18, 0, 60, 28, 0]]"
}""",
"""Human Instruction: Wipe the horizontal area.
Input: {'object 1': [35, 33, 19], 'object 2': [51, 41, 15], 'object 3': [67, 63, 15]}
Output: {
    "reasoning_and_reflection": "I need to identify which area is horizontal first. I can see that the yellow triangle area at [51, 41, 15] is horizontal since it is parallel to the green arrow representing the y-axis of the coordinate system annotated in the image. Then, the task is to pick up the sponge at [35, 33, 19] and use it to wipe the horizontal area at [51, 41, 15]. The plan involves moving the gripper to the sponge's position, closing the gripper to pick it up, moving to the right side of the horizontal area, and moving the sponge along the main direction of the horizontal area to wipe it.",
    "executable_plan": "[[35, 34, 25, 0, 60, 20, 1], [35, 34, 17, 0, 60, 20, 0], [35, 34, 27, 0, 60, 20, 0], [52, 47, 18, 0, 60, 84, 0], [44, 19, 18, 0, 60, 84, 0]]"
}""",
"""Human Instruction: Wipe the horizontal area.
Input: {'object 1': [47, 34, 18], 'object 2': [72, 53, 15], 'object 3': [54, 81, 15]}
Output: {
    "reasoning_and_reflection": "I need to identify which area is horizontal first. I can see that the yellow triangle area at [72, 53, 15] is horizontal since it is parallel to the green arrow representing the y-axis of the coordinate system annotated in the image. Then, the task is to pick up the sponge at [47, 34, 18] and use it to wipe the horizontal area at [72, 53, 15]. The plan involves moving the gripper to the sponge's position, closing the gripper to pick it up, moving to the right side of the horizontal area, and moving the sponge along the main direction of the horizontal area to wipe it.",
    "executable_plan": "[[46, 31, 24, 0, 60, 18, 1], [46, 31, 16, 0, 60, 18, 0], [46, 31, 26, 0, 60, 18, 0], [72, 63, 17, 0, 60, 88, 0], [69, 37, 17, 0, 60, 88, 0]]"
}""",
"""Human Instruction: Wipe the vertical area.
Input: "{'object 1': [55, 28, 15], 'object 2': [42, 51, 19], 'object 3': [32, 63, 15]}"
Output: {
    "reasoning_and_reflection": "I need to identify which area is vertical first. I can see that the white rectangle area at [55, 28, 15] is vertical since it is parallel to the red arrow representing the x-axis of the coordinate system annotated in the image. Then, the task is to pick up the sponge at [42, 51, 19] and use it to wipe the vertical area at [55, 28, 15]. The plan involves moving the gripper to the sponge's position, closing the gripper to pick it up, moving to the top side of the vertical area, and moving the sponge along the main direction of the vertical area to wipe it.",
    "executable_plan": "[[40, 48, 25, 0, 60, 12, 1], [40, 48, 17, 0, 60, 12, 0], [40, 48, 27, 0, 60, 12, 0], [38, 25, 17, 0, 59, 0, 0], [69, 25, 17, 0, 59, 0, 0]]"
}""",
"""Human Instruction: Wipe the vertical area.
Input: {'object 1': [53, 31, 19], 'object 2': [66, 34, 15], 'object 3': [59, 73, 15]}
Output: {
    "reasoning_and_reflection": "I need to identify which area is vertical first. I can see that the blue rectangle area at [66, 34, 15] is vertical since it is parallel to the red arrow representing the x-axis of the coordinate system annotated in the image. Then, the task is to pick up the sponge at [53, 31, 19] and use it to wipe the vertical area at [66, 34, 15]. The plan involves moving the gripper to the sponge's position, closing the gripper to pick it up, moving to the top side of the vertical area, and moving the sponge along the main direction of the vertical area to wipe it.",
    "executable_plan": "[[52, 28, 26, 0, 60, 94, 1], [52, 28, 18, 0, 60, 94, 0], [52, 28, 28, 0, 60, 94, 0]], [46, 32, 18, 0, 60, 68, 0], [78, 35, 18, 0, 60, 75, 0]]"
}""",
"""Human Instruction: Wipe the vertical area.
Input: {'object 1': [37, 26, 15], 'object 2': [57, 49, 15], 'object 3': [56, 73, 18]}
Output: {
    "reasoning_and_reflection": "I need to identify which area is vertical first. I can see that the white triangle area at [37, 26, 15] is vertical since it is parallel to the red arrow representing the x-axis of the coordinate system annotated in the image. Then, the task is to pick up the sponge at [56, 73, 18] and use it to wipe the vertical area at [37, 26, 15]. The plan involves moving the gripper to the sponge's position, closing the gripper to pick it up, moving to the top side of the vertical area, and moving the sponge along the main direction of the vertical area to wipe it.",
    "executable_plan": "[[55, 70, 25, 0, 60, 26, 1], [55, 70, 17, 0, 60, 26, 0], [55, 70, 27, 0, 60, 26, 0], [35, 26, 18, 0, 60, 55, 0], [62, 19, 18, 0, 60, 55, 0]]"
}""",
"""Human Instruction: Wipe the vertical area.
Input: {'object 1': [64, 21, 15], 'object 2': [34, 39, 15], 'object 3': [51, 43, 19]}
Output: {
    "reasoning_and_reflection": "I need to identify which area is vertical first. I can see that the green triangle area at [64, 21, 15] is vertical since it is parallel to the red arrow representing the x-axis of the coordinate system annotated in the image. Then, the task is to pick up the sponge at [51, 43, 19] and use it to wipe the vertical area at [64, 21, 15]. The plan involves moving the gripper to the sponge's position, closing the gripper to pick it up, moving to the top side of the vertical area, and moving the sponge along the main direction of the vertical area to wipe it.",
    "executable_plan": "[[44, 46, 25, 116, 62, 97, 1], [45, 45, 17, 116, 62, 97, 0], [43, 46, 27, 116, 62, 97, 0], [50, 13, 18, 115, 60, 13, 0], [75, 26, 18, 115, 60, 99, 0]]"
}"""
]

wipe_examples_vlm_ablation = [
"""Human Instruction: Wipe the horizontal area.
Input: {'object 1': [33, 36, 19], 'object 2': [52, 38, 15], 'object 3': [59, 68, 15]}
Output: {
    "visual_state_description": "From left to right, I can see a yellow sponge at [33, 36, 19], a green rectangle area at [52, 38, 15] and an orange rectangle area at [59, 68, 15].",
    "reasoning_and_reflection": "I need to identify which area is horizontal first. I can see that the orange rectangle area at [59, 68, 15] is horizontal since it is parallel to the green arrow representing the y-axis of the coordinate system annotated in the image. Then, the task is to pick up the sponge at [33, 36, 19] and use it to wipe the horizontal area at [59, 68, 15]. The plan involves moving the gripper to the sponge's position, closing the gripper to pick it up, moving to the side of the horizontal area, and moving the sponge along the main direction of the horizontal area to wipe it.",
    "executable_plan": "[[32, 34, 25, 0, 60, 34, 1], [32, 34, 17, 0, 60, 34, 0], [32, 34, 27, 0, 60, 34, 0], [60, 80, 18, 0, 61, 31, 0], [61, 54, 17, 0, 61, 31, 0]]"
}""",
"""Human Instruction: Wipe the vertical area.
Input: "{'object 1': [55, 28, 15], 'object 2': [42, 51, 19], 'object 3': [32, 63, 15]}"
Output: {
    "visual_state_description": "From left to right, I can see a white rectangle area at [55, 28, 15], a yellow sponge at [42, 51, 19] and a blue rectangle area at [32, 63, 15].",
    "reasoning_and_reflection": "I need to identify which area is vertical first. I can see that the white rectangle area at [55, 28, 15] is vertical since it is parallel to the red arrow representing the x-axis of the coordinate system annotated in the image. Then, the task is to pick up the sponge at [42, 51, 19] and use it to wipe the vertical area at [55, 28, 15]. The plan involves moving the gripper to the sponge's position, closing the gripper to pick it up, moving to the top side of the vertical area, and moving the sponge along the main direction of the vertical area to wipe it.",
    "executable_plan": "[[40, 48, 25, 0, 60, 12, 1], [40, 48, 17, 0, 60, 12, 0], [40, 48, 27, 0, 60, 12, 0], [38, 25, 17, 0, 59, 0, 0], [69, 25, 17, 0, 59, 0, 0]]"
}""",
"""Human Instruction: Wipe the horizontal area.
Input: {'object 1': [56, 19, 15], 'object 2': [35, 41, 19], 'object 3': [54, 47, 15]}
Output: {
    "visual_state_description": "From left to right, I can see a black rectangle area at [56, 19, 15], a yellow sponge at [35, 41, 19] and a white rectangle area at [54, 47, 15].",
    "reasoning_and_reflection": "I need to identify which area is horizontal first. I can see that the black rectangle area at [56, 19, 15] is horizontal since it is parallel to the green arrow representing the y-axis of the coordinate system annotated in the image. Then, the task is to pick up the sponge at [35, 41, 19] and use it to wipe the horizontal area at [56, 19, 15]. The plan involves moving the gripper to the sponge's position, closing the gripper to pick it up, moving to the right side of the horizontal area, and moving the sponge along the main direction of the horizontal area to wipe it.",
    "executable_plan": "[[33, 43, 25, 0, 60, 32, 1], [33, 43, 17, 0, 60, 32, 0], [33, 43, 27, 0, 60, 32, 0], [58, 35, 18, 0, 60, 28, 0], [56, 8, 18, 0, 60, 28, 0]]"
}""",
"""Human Instruction: Wipe the vertical area.
Input: {'object 1': [53, 31, 19], 'object 2': [66, 34, 15], 'object 3': [59, 73, 15]}
Output: {
    "visual_state_description": "From left to right, I can see a yellow sponge at [53, 31, 19], a blue rectangle area at [66, 34, 15] and a green rectangle area at [59, 73, 15].",
    "reasoning_and_reflection": "I need to identify which area is vertical first. I can see that the blue rectangle area at [66, 34, 15] is vertical since it is parallel to the red arrow representing the x-axis of the coordinate system annotated in the image. Then, the task is to pick up the sponge at [53, 31, 19] and use it to wipe the vertical area at [66, 34, 15]. The plan involves moving the gripper to the sponge's position, closing the gripper to pick it up, moving to the top side of the vertical area, and moving the sponge along the main direction of the vertical area to wipe it.",
    "executable_plan": "[[52, 28, 26, 0, 60, 94, 1], [52, 28, 18, 0, 60, 94, 0], [52, 28, 28, 0, 60, 94, 0]], [46, 32, 18, 0, 60, 68, 0], [78, 35, 18, 0, 60, 75, 0]]"
}""",
"""Human Instruction: Wipe the horizontal area.
Input: {'object 1': [35, 33, 19], 'object 2': [51, 41, 15], 'object 3': [67, 63, 15]}
Output: {
    "visual_state_description": "From left to right, I can see a yellow sponge at [35, 33, 19], a yellow triangle area at [51, 41, 15] and a blue triangle area at [67, 63, 15].",
    "reasoning_and_reflection": "I need to identify which area is horizontal first. I can see that the yellow triangle area at [51, 41, 15] is horizontal since it is parallel to the green arrow representing the y-axis of the coordinate system annotated in the image. Then, the task is to pick up the sponge at [35, 33, 19] and use it to wipe the horizontal area at [51, 41, 15]. The plan involves moving the gripper to the sponge's position, closing the gripper to pick it up, moving to the right side of the horizontal area, and moving the sponge along the main direction of the horizontal area to wipe it.",
    "executable_plan": "[[35, 34, 25, 0, 60, 20, 1], [35, 34, 17, 0, 60, 20, 0], [35, 34, 27, 0, 60, 20, 0], [52, 47, 18, 0, 60, 84, 0], [44, 19, 18, 0, 60, 84, 0]]"
}""",
"""Human Instruction: Wipe the vertical area.
Input: {'object 1': [37, 26, 15], 'object 2': [57, 49, 15], 'object 3': [56, 73, 18]}
Output: {
    "visual_state_description": "From left to right, I can see a white triangle area at [37, 26, 15], a black triangle area at [57, 49, 15] and a yellow sponge at [56, 73, 18].",
    "reasoning_and_reflection": "I need to identify which area is vertical first. I can see that the white triangle area at [37, 26, 15] is vertical since it is parallel to the red arrow representing the x-axis of the coordinate system annotated in the image. Then, the task is to pick up the sponge at [56, 73, 18] and use it to wipe the vertical area at [37, 26, 15]. The plan involves moving the gripper to the sponge's position, closing the gripper to pick it up, moving to the top side of the vertical area, and moving the sponge along the main direction of the vertical area to wipe it.",
    "executable_plan": "[[55, 70, 25, 0, 60, 26, 1], [55, 70, 17, 0, 60, 26, 0], [55, 70, 27, 0, 60, 26, 0], [35, 26, 18, 0, 60, 55, 0], [62, 19, 18, 0, 60, 55, 0]]"
}""",
"""Human Instruction: Wipe the horizontal area.
Input: {'object 1': [47, 34, 18], 'object 2': [72, 53, 15], 'object 3': [54, 81, 15]}
Output: {
    "visual_state_description": "From left to right, I can see a yellow sponge at [47, 34, 18], a yellow triangle area at [72, 53, 15] and a green triangle area at [54, 81, 15].",
    "reasoning_and_reflection": "I need to identify which area is horizontal first. I can see that the yellow triangle area at [72, 53, 15] is horizontal since it is parallel to the green arrow representing the y-axis of the coordinate system annotated in the image. Then, the task is to pick up the sponge at [47, 34, 18] and use it to wipe the horizontal area at [72, 53, 15]. The plan involves moving the gripper to the sponge's position, closing the gripper to pick it up, moving to the right side of the horizontal area, and moving the sponge along the main direction of the horizontal area to wipe it.",
    "executable_plan": "[[46, 31, 24, 0, 60, 18, 1], [46, 31, 16, 0, 60, 18, 0], [46, 31, 26, 0, 60, 18, 0], [72, 63, 17, 0, 60, 88, 0], [69, 37, 17, 0, 60, 88, 0]]"
}""",
"""Human Instruction: Wipe the vertical area.
Input: {'object 1': [64, 21, 15], 'object 2': [34, 39, 15], 'object 3': [51, 43, 19]}
Output: {
    "visual_state_description": "From left to right, I can see a green triangle area at [64, 21, 15], a red triangle area at [34, 39, 15] and a yellow sponge at [51, 43, 19].",
    "reasoning_and_reflection": "I need to identify which area is vertical first. I can see that the green triangle area at [64, 21, 15] is vertical since it is parallel to the red arrow representing the x-axis of the coordinate system annotated in the image. Then, the task is to pick up the sponge at [51, 43, 19] and use it to wipe the vertical area at [64, 21, 15]. The plan involves moving the gripper to the sponge's position, closing the gripper to pick it up, moving to the top side of the vertical area, and moving the sponge along the main direction of the vertical area to wipe it.",
    "executable_plan": "[[44, 46, 25, 116, 62, 97, 1], [45, 45, 17, 116, 62, 97, 0], [43, 46, 27, 116, 62, 97, 0], [50, 13, 18, 115, 60, 13, 0], [75, 26, 18, 115, 60, 99, 0]]"
}"""
]

vlm_examples_baseline = {
    "pick": pick_examples_vlm,
    "stack": stack_examples_vlm,
    "place": shape_sorter_examples_vlm,
    "wipe": wipe_examples_vlm
}

llm_examples = {
    "pick": pick_examples_llm,
    "stack": stack_examples_llm,
    "place": shape_sorter_examples_llm,
    "wipe": wipe_examples_llm
}

vlm_examples_ablation = {
    "pick": pick_examples_vlm,
    "stack": stack_examples_vlm,
    "place": shape_sorter_examples_vlm,
    "wipe": wipe_examples_vlm_ablation
}