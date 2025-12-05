import os
import random
from pyrep.objects.shape import Shape
from amsolver.backend.unit_tasks import VLM_Object
from amsolver.const import cabinet_list
from vlm.tasks.open_drawer import OpenDrawer

class OpenDrawerCabinet(OpenDrawer):


    def variation_count(self) -> int:
        # TODO: The number of variations for this task.
        return 8

    def import_objects(self):
        self._selected_cabinet = random.choice(cabinet_list)
        model_path = self.model_dir+self._selected_cabinet["path"]
        self.drawer = VLM_Object(self.pyrep, model_path, 0)
        self._drawer_init_ori = self.drawer.get_orientation()
        self._drawer_init_state = self.drawer.get_configuration_tree()
        self.drawer.set_parent(Shape("boundary_root"))
        self._need_remove_objects.append(self.drawer)
