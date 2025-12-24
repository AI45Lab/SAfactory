"""
EmbodiedAlfredGym Prompt Templates
将所有 prompt 字符串集中管理，提高代码可读性
"""

# ============================================================================
# System Prompt
# ============================================================================

SYSTEM_PROMPT = (
    "You are an embodied AI agent completing household tasks. "
    "Tasks require multiple sequential steps. "
    "Output valid JSON with your next action plan. "
    "Do NOT repeat the same action - progress through the task steps."
)

# ============================================================================
# Task Instruction Header
# ============================================================================

TASK_HEADER = "## Task Instruction"

# ============================================================================
# Previous Action Templates
# ============================================================================

PREVIOUS_ACTION_HEADER = "## Previous Action (Step {step})"
PREVIOUS_ACTION_TEMPLATE = """Action: {action}
Success: {success_status}
Progress: {progress:.1f}% of task completed
Feedback: {feedback}"""

SUCCESS_STATUS_YES = "✓ YES - Action completed successfully"
SUCCESS_STATUS_NO = "✗ NO - Action failed"

# ============================================================================
# Instruction Templates - After Find Actions
# ============================================================================

# 找到目的地后且正在holding物品
INSTRUCTION_AFTER_FIND_DESTINATION_HOLDING = """⚠️ IMPORTANT INSTRUCTION:
- You just successfully FOUND the destination (progress: {progress:.1f}%)
- You are HOLDING an object
- DO NOT repeat 'find' or 'pick up' actions!
- **NEXT MANDATORY STEP**: put down the object in hand
- Look for action like 'put down the object in hand' in the Available Actions list
- This is the FINAL step to complete the task!"""

# 找到目的地后但没有holding物品
INSTRUCTION_AFTER_FIND_DESTINATION_NOT_HOLDING = """⚠️ IMPORTANT INSTRUCTION:
- You just successfully FOUND a receptacle (progress: {progress:.1f}%)
- You are NOT holding anything yet
- **NEXT STEP**: Look for the target object to pick up
- This receptacle may contain the object you need"""

# 找到设备后
INSTRUCTION_AFTER_FIND_EQUIPMENT = """⚠️ IMPORTANT INSTRUCTION:
- You just successfully FOUND equipment (progress: {progress:.1f}%)
- DO NOT repeat 'find' or 'pick up' actions!
- **NEXT MANDATORY STEP**: {next_action_hint}
- You are HOLDING an object and need to use this equipment on it"""

# 找到物品后
INSTRUCTION_AFTER_FIND_OBJECT = """⚠️ IMPORTANT INSTRUCTION:
- You just successfully FOUND the object (progress: {progress:.1f}%)
- DO NOT repeat 'find' action again!
- **NEXT MANDATORY STEP**: Look for 'pick up' action in the Available Actions list and use it
- Example: If you found a Ladle, now you must output {{"action_id": <id of 'pick up Ladle'>, ...}}"""

# ============================================================================
# Instruction Templates - After Pick Up Actions
# ============================================================================

INSTRUCTION_AFTER_PICKUP = """⚠️ IMPORTANT INSTRUCTION:
- You just successfully PICKED UP the object (progress: {progress:.1f}%)
- DO NOT repeat 'pick up' or 'find' the same object again!
- **NEXT MANDATORY STEP**: {next_step_hint}"""

# Pick up 后的下一步提示 - 根据任务类型
PICKUP_NEXT_STEP_WASH = "find a Faucet/Sink, then turn on/off the faucet"
PICKUP_NEXT_STEP_HEAT = "find a Microwave, then turn it on"
PICKUP_NEXT_STEP_COOL = "find a Fridge, then open and close it"
PICKUP_NEXT_STEP_SLICE = "find a Knife, then slice the object"
PICKUP_NEXT_STEP_DEFAULT = "navigate to the target destination and put down the object"

# ============================================================================
# Instruction Templates - After Navigation Actions
# ============================================================================

INSTRUCTION_AFTER_NAVIGATE = """⚠️ IMPORTANT INSTRUCTION:
- You just successfully NAVIGATED to a location (progress: {progress:.1f}%)
- DO NOT repeat navigation!
- **NEXT MANDATORY STEP**: Perform the required action (clean/heat/cool/slice) or PUT DOWN the object"""

# ============================================================================
# Instruction Templates - After Interaction Actions (turn on/off, open/close)
# ============================================================================

INSTRUCTION_AFTER_TURN_ON = """⚠️ IMPORTANT INSTRUCTION:
- You just successfully TURNED ON equipment (progress: {progress:.1f}%)
- DO NOT repeat 'turn on'!
- **NEXT MANDATORY STEP**: turn off the same equipment (e.g., 'turn off the Faucet')
- After turning off, navigate to destination and PUT DOWN the object"""

INSTRUCTION_AFTER_TURN_OFF = """⚠️ IMPORTANT INSTRUCTION:
- You just completed equipment interaction (progress: {progress:.1f}%)
- **NEXT MANDATORY STEP**: Navigate to final destination (e.g., find Table) and PUT DOWN the object"""

INSTRUCTION_AFTER_INTERACTION_DEFAULT = """⚠️ IMPORTANT INSTRUCTION:
- You just successfully performed an action (progress: {progress:.1f}%)
- **NEXT MANDATORY STEP**: Navigate to final destination and PUT DOWN the object"""

# ============================================================================
# Instruction Templates - Generic Success / Failure
# ============================================================================

INSTRUCTION_SUCCESS_GENERIC = "⚠️ Previous action succeeded (progress: {progress:.1f}%). Continue to next logical step."

INSTRUCTION_FAILURE = """⚠️ Previous action failed:
Feedback: {feedback}
Consider the feedback and choose the next appropriate action."""

# ============================================================================
# Current Observation Section
# ============================================================================

OBSERVATION_HEADER = "## Current Observation"
OBSERVATION_PROMPT = "Step {step}: Based on the image and previous actions, what is the NEXT action?"

# ============================================================================
# Available Actions Section
# ============================================================================

ACTIONS_HEADER = "## Available Actions ({count} total)"
ACTIONS_SUBHEADER = "Here are all available action IDs and their descriptions:"
ACTIONS_TRUNCATED = "  ... (total {total} actions available, showing {shown} most relevant)"

# ============================================================================
# Output Format Section
# ============================================================================

OUTPUT_FORMAT_SECTION = """## Output Format (CRITICAL)
You MUST output valid JSON in the following format:
```json
{{
  "reasoning": "Based on the current image and task progress, I need to...",
  "executable_plan": [
    {{"action_id": 123, "description": "the next action to take"}}
  ]
}}
```

RULES:
- Output ONLY valid JSON (no extra text before or after)
- executable_plan must contain at least one action
- ONLY use action_id from the list above (0 to {max_action_id})
- Do NOT make up action IDs - they must exist in the Available Actions list
- Do the NEXT step of the task progressively
- Typical flow: find object → pick up → navigate to equipment/destination → interact (turn on/off, etc.) → place down"""

# ============================================================================
# Equipment Keywords for Detection
# ============================================================================

EQUIPMENT_KEYWORDS = ['faucet', 'sink', 'microwave', 'fridge', 'stove']
DESTINATION_KEYWORDS = ['table', 'counter', 'shelf', 'cabinet', 'drawer', 'bed', 'sofa']

# ============================================================================
# Task Type Detection Keywords
# ============================================================================

TASK_WASH_KEYWORDS = ['wash', 'rinse', 'clean']
TASK_HEAT_KEYWORDS = ['heat', 'warm']
TASK_COOL_KEYWORDS = ['cool', 'chill']
TASK_SLICE_KEYWORDS = ['slice', 'cut']

# ============================================================================
# Equipment Next Action Hints
# ============================================================================

def get_equipment_next_action_hint(action_lower: str) -> str:
    """根据找到的设备类型返回下一步操作提示"""
    if 'faucet' in action_lower or 'sink' in action_lower:
        return "turn on the Faucet (look for action like 'turn on the Faucet' or ID containing 'turn on')"
    elif 'microwave' in action_lower:
        return "turn on the Microwave"
    elif 'fridge' in action_lower:
        return "open the Fridge"
    else:
        return "interact with the equipment (turn on/open)"


def get_pickup_next_step_hint(task_instruction: str) -> str:
    """根据任务类型返回 pick up 后的下一步提示"""
    task_lower = task_instruction.lower()
    
    if any(kw in task_lower for kw in TASK_WASH_KEYWORDS):
        return PICKUP_NEXT_STEP_WASH
    elif any(kw in task_lower for kw in TASK_HEAT_KEYWORDS):
        return PICKUP_NEXT_STEP_HEAT
    elif any(kw in task_lower for kw in TASK_COOL_KEYWORDS):
        return PICKUP_NEXT_STEP_COOL
    elif any(kw in task_lower for kw in TASK_SLICE_KEYWORDS):
        return PICKUP_NEXT_STEP_SLICE
    else:
        return PICKUP_NEXT_STEP_DEFAULT


def get_priority_keywords_after_action(last_action: str, task_instruction: str, is_holding: bool) -> list:
    """
    根据上一步动作类型，返回应优先展示的动作关键词列表
    
    Args:
        last_action: 上一步动作描述
        task_instruction: 任务指令
        is_holding: 是否正在holding物品
        
    Returns:
        优先关键词列表
    """
    action_lower = last_action.lower()
    task_lower = task_instruction.lower()
    
    if 'find' in action_lower:
        is_equipment = any(kw in action_lower for kw in EQUIPMENT_KEYWORDS)
        is_destination = any(kw in action_lower for kw in DESTINATION_KEYWORDS)
        
        if is_destination:
            if is_holding:
                return ['put', 'place', 'drop']
            else:
                return ['pick', 'find']
        elif is_equipment:
            return ['turn', 'toggle', 'open', 'close']
        else:
            return ['pick']
    
    elif 'pick' in action_lower or 'pickup' in action_lower:
        if any(kw in task_lower for kw in TASK_WASH_KEYWORDS):
            return ['find', 'go', 'turn', 'toggle', 'put']
        elif any(kw in task_lower for kw in TASK_HEAT_KEYWORDS):
            return ['find', 'go', 'turn', 'toggle', 'put']
        elif any(kw in task_lower for kw in TASK_COOL_KEYWORDS):
            return ['find', 'go', 'open', 'close', 'put']
        else:
            return ['find', 'go', 'put', 'place']
    
    elif 'turn' in action_lower or 'toggle' in action_lower or 'open' in action_lower or 'close' in action_lower:
        if 'turn on' in action_lower or 'toggle on' in action_lower or 'open' in action_lower:
            return ['turn off', 'toggle', 'close']
        elif 'turn off' in action_lower or 'toggle off' in action_lower or 'close' in action_lower:
            return ['find', 'go', 'put', 'place']
        else:
            return ['put', 'place', 'find', 'go']
    
    # 默认关键词
    return ['find', 'pick', 'go', 'put', 'place', 'turn', 'toggle', 'open', 'close', 'slice']


# ============================================================================
# Default Priority Keywords
# ============================================================================

DEFAULT_PRIORITY_KEYWORDS = ['find', 'pick', 'go', 'put', 'place', 'turn', 'toggle', 'open', 'close', 'slice']
OTHER_ACTION_KEYWORDS = ['find', 'pick', 'put', 'place', 'drop', 'go', 'turn', 'toggle', 'open', 'close', 'slice']

