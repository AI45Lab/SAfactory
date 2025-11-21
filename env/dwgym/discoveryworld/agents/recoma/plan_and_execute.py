from math import floor
import re
from recoma.models.core.base_model import BaseModel
from recoma.search.state import SearchState

from agents.recoma.discoveryworld_env_models import task_completed



@BaseModel.register("discoveryworld_plan_and_execute")
class DiscoveryworldPlanAndExecute(BaseModel):

    def __init__(self, plan_model: str, exec_model: str, **kwargs):
        super().__init__(**kwargs)
        self.plan_model = plan_model
        self.exec_model = exec_model
        self.step_regex = re.compile("Step [0-9]+: (.*)")

    def extract_step(self, plan_step):
        re_match = self.step_regex.match(plan_step)
        if re_match:
            return re_match.group(1)
        else:
            print("Cannot parse step from:{} in the generated plan! Using as-is!".format(plan_step))
            return plan_step

    def create_planner_input(self, state: SearchState, children):
        planner_input = state.example.task + "\n\nPlan:\n"
        for step_idx in range(floor(len(children) / 2)):
            plan_child = children[2*step_idx]
            planner_input += "ASSISTANT:\n"
            planner_input += plan_child.output #"Step {}: {}\n".format(int(step_idx + 1), plan_child.output)
            exec_child = children[2*step_idx + 1]
            planner_input += "\nUSER:\n"
            planner_input += " --" + exec_child.output + "\n"
        # planner_input += "Step {}:".format((len(children) / 2) + 1)
        return planner_input

    def create_executer_input(self, state: SearchState, children):
        exec_input = "**" + children[-1].output + "**\n\nPrevious Sub-Tasks:\n"
        for step_idx in range(floor(len(children) / 2)):
            plan_child = children[2*step_idx]
            exec_input += "Subtask {}: {}\n".format(int(step_idx + 1), self.extract_step(plan_child.output))
            exec_child = children[2*step_idx + 1]
            exec_input += " Result: " + exec_child.output + "\n"
        return exec_input

    def __call__(self, state: SearchState) -> list[SearchState]:
        new_state = state.clone(deep=True)
        current_node = new_state.get_open_node()
        children = new_state.get_children(current_node)
        if len(children) % 2 == 0:
            # Stop?
            if task_completed():
                current_node.close(children[-1].output)
                return [new_state]
            # call planner
            planner_input = self.create_planner_input(new_state, children)
            new_state.add_next_step(next_step_input=planner_input,
                                    next_step_model=self.plan_model,
                                    current_step_node=current_node)
            return [new_state]
        else:
            # Stop?
            last_child = children[-1]
            if last_child.output.endswith("Done!"):
                current_node.close(last_child.output)
                return [new_state]
            # if not, call executer for the last step in the plan
            executer_input = self.create_executer_input(new_state, children)
            new_state.add_next_step(next_step_input=executer_input,
                                    next_step_model=self.exec_model,
                                    current_step_node=current_node)
            return [new_state]