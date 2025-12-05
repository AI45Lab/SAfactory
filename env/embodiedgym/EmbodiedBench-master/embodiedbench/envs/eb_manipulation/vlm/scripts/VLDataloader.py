import os
from random import sample
import numpy as np
from torch.utils.data import Dataset
import torch
import cv2
from pathlib import Path
import pickle
from cliport.utils.utils import get_fused_heightmap
pickle.DEFAULT_PROTOCOL=pickle.HIGHEST_PROTOCOL
from amsolver.observation_config import ObservationConfig
from amsolver.utils import get_stored_demos
import time
import copy
from scipy.spatial.transform import Rotation as R
from num2words import num2words


class VLM_dataset(Dataset):
    def __init__(self, root, setd, img_size=(360, 360), 
                    unused_camera_list = ['left_shoulder', 'right_shoulder', 'overhead','wrist'], preprocess = True, 
                    use_fail_cases = True, sample_numbers = None, train_tasks = None, random_sample = False, args=None):
        self.root = root
        self.setd = setd
        self.dataset_path = Path(os.path.join(self.root, self.setd))
        self.episode_list = []
        self.variation_list = []
        self.task_list = {}
        self.fail_cases_list = []
        self.read_lists()
        self.use_fail_cases = use_fail_cases
        if train_tasks is not None:
            self.episode_list = []
            for t in train_tasks:
                for n in self.task_list:
                    if t in n:
                        self.episode_list += self.task_list[n]['success']
                        self.fail_cases_list += self.task_list[n]['fail']
        if use_fail_cases:
            self.episode_list += self.fail_cases_list
        #only train selected tasks

        self.valid_episodes, self.invalid_episodes = [],[]
        self.sample_numbers = sample_numbers
        self.random_sample = random_sample
        self.img_size = img_size
        self.preprocess = preprocess

        self.obs_config = ObservationConfig()
        self.obs_config.set_all(True)
        self.obs_config.right_shoulder_camera.image_size = self.img_size
        self.obs_config.left_shoulder_camera.image_size = self.img_size
        self.obs_config.overhead_camera.image_size = self.img_size
        self.obs_config.wrist_camera.image_size = self.img_size
        self.obs_config.front_camera.image_size = self.img_size

        self.views = list(set(['left_shoulder', 'right_shoulder', 'overhead', 'wrist', 'front']) - set(unused_camera_list))

        if 'left_shoulder' in unused_camera_list:
            self.obs_config.left_shoulder_camera.set_all(False)
        if 'right_shoulder' in unused_camera_list:
            self.obs_config.right_shoulder_camera.set_all(False)
        if 'overhead' in unused_camera_list:
            self.obs_config.overhead_camera.set_all(False)
        if 'wrist' in unused_camera_list:
            self.obs_config.wrist_camera.set_all(False)
        if 'front' in unused_camera_list:
            self.obs_config.front_camera.set_all(False)
        
        self.relative = False
        self.renew_obs = False
        self.add_low_lang = False
        if args is not None:
            self.relative = args.relative
            self.renew_obs = args.renew_obs
            self.add_low_lang = args.add_low_lang

    def read_lists(self):
        tasks_list_path = self.dataset_path / '{}_list.pkl'.format(self.setd)
        if not tasks_list_path.is_file():
            self.task_list = {}
            self.variation_list =set()
            for path in self.dataset_path.rglob('low_dim_obs*'):
                path = path.relative_to(self.dataset_path)
                task_name = str(path.parents[3])
                if task_name not in self.task_list:
                    self.task_list[task_name]={'success':[], 'fail':[]}
                self.variation_list.add(path.parents[2])
                if 'fail_cases' in str(path):
                    self.fail_cases_list.append(path.parent)
                    self.task_list[task_name]['fail'].append(path.parent)
                else:
                    self.episode_list.append(path.parent)
                    self.task_list[task_name]['success'].append(path.parent)
            self.variation_list = list(self.variation_list)
            with open(tasks_list_path,'wb') as f:
                pickle.dump({'task_list': self.task_list, 
                            'episode_list': self.episode_list,
                            'fail_cases_list': self.fail_cases_list,
                            'variation_list': self.variation_list}, f)
        else:
            with open(tasks_list_path,'rb') as f:
                info_dict = pickle.load(f)
                self.task_list = info_dict['task_list']
                self.episode_list = info_dict['episode_list']
                self.variation_list = info_dict['variation_list']
                self.fail_cases_list = info_dict['fail_cases_list']

    def __getitem__(self, index):
        if index in self.invalid_episodes:
            index = sample(self.valid_episodes, 1)[0]
        episode = self.episode_list[index]
        variation_path = episode.parents[1]
        task_name = episode.parents[2]
        fail_cases = 'fail_cases' in str(episode)

        low_dim_obs = self.dataset_path/episode/"low_dim_obs.pkl"
        with open(low_dim_obs, 'rb') as f:
            demo_temple = pickle.load(f)
        
        sequence_length = len(demo_temple._observations)
        obs_select_inds = np.arange(sequence_length)
        if self.sample_numbers:
            if self.random_sample:
                obs_select_inds = np.sort(np.random.choice(obs_select_inds, self.sample_numbers, replace=False))
            else:
                obs_select_inds = obs_select_inds[0:self.sample_numbers]
        split_by_waypoint = True
        if split_by_waypoint:
            obs_select_inds = [0]
            previous_waypoint="waypoint0"
            self.all_waypoints = [previous_waypoint]
            for i, obs in enumerate(demo_temple._observations):
                if obs.current_waypoint_name == previous_waypoint:
                    continue
                else:
                    previous_waypoint = obs.current_waypoint_name
                    self.all_waypoints.append(previous_waypoint)
                    obs_select_inds.append(i)
            # for i in range(len(obs_select_inds)):
            #     if i+1<len(obs_select_inds):
            #         random_i = np.random.randint(obs_select_inds[i], obs_select_inds[i+1])
            #     else:
            #         random_i = np.random.randint(obs_select_inds[i], sequence_length)
            #     obs_select_inds[i] = random_i
        if self.preprocess:
            preprocess_data_folder = self.dataset_path/episode/'preprocess_data'

            need_rebuild = False
            if not preprocess_data_folder.is_dir():
                preprocess_data_folder.mkdir()
                need_rebuild = True
            if not hasattr(demo_temple, 'observation_config'):
                need_rebuild = True
            else:
                need_rebuild = not (demo_temple.observation_config == self.obs_config)
            obs_list = os.listdir(preprocess_data_folder)
            if len(obs_list)<len(obs_select_inds):
                need_rebuild=True
            elif len(obs_list)>0:
                obs = []
                for i in obs_select_inds:
                    ob_path = preprocess_data_folder/(str(i)+'_preprocess.pkl')
                    if ob_path.is_file():
                        try:
                            with open(ob_path, 'rb') as f:
                                obs.append(pickle.load(f))
                        except:
                            need_rebuild = True
                            break
                    else:
                        need_rebuild = True
                        break
            if need_rebuild:
                episode_name = episode.name
                variation_number = int(variation_path.name.replace('variation',''))
                demos = get_stored_demos(1, False, self.dataset_path, variation_number, 
                                    task_name, self.obs_config, episode_name, fail_cases)
                data = demos[0]
                obs = data._observations
                for i in obs_select_inds:
                    file_name = preprocess_data_folder/ "{}_preprocess.pkl".format(i)
                    with open(file_name, 'wb') as f:
                        pickle.dump(obs[i], f)
                print('Finish {} preprocess!'.format(episode))
                demo_temple.observation_config = self.obs_config
                with open(low_dim_obs, 'wb') as f:
                    pickle.dump(demo_temple, f)
                obs = [obs[i] for i in obs_select_inds]
        else:
            # episode_number = int(episode.name.replace('episode',''))
            episode_name = episode.name
            variation_number = int(variation_path.name.replace('variation',''))
            demos = get_stored_demos(1, False, self.dataset_path, variation_number, 
                                    task_name, self.obs_config, episode_name, fail_cases, obs_select_inds)
            data = demos[0]
            obs = data._observations
            obs = [obs[i] for i in obs_select_inds]
        output_dict = self.get_cliport_gt(obs, demo_temple.high_level_instructions, episode)
        if output_dict['valid']:
            self.valid_episodes.append(index)
        else:
            self.invalid_episodes.append(index)
            if len(self.valid_episodes) == 0:
                other_indexs = list(set(range(self.__len__())) - set(self.invalid_episodes))
                valid_index = sample(other_indexs, 1)[0]
            else:
                valid_index = sample(self.valid_episodes, 1)[0]
            output_dict = self.__getitem__(valid_index)
        return output_dict

    def get_cliport_gt(self, data, languages, episode):
        z_max = 1.2
        if 'door' in str(episode) or 'drawer' in str(episode):
            z_max = 1.8
        bounds = np.array([[-0.05,0.67],[-0.45, 0.45], [0.7, z_max]])
        pixel_size = 5.625e-3
        target_obj = None
        cmaps, hmaps = [], []
        high_l = np.random.choice(languages, 1)[0]
        if high_l[-1]!=".":
            high_l+="."
        language_instructions, target_points = [], []
        attention_points = []
        step_list, point_list = [], []
        attention_id = data[0].object_informations["waypoint0"]["target_obj"]
        step_img_id = 0
        for i, wp in enumerate(self.all_waypoints):
            waypoint_info = data[0].object_informations[wp]
            waypoint_type = waypoint_info['waypoint_type']
            if "pre" in waypoint_type:
                focus_wp = self.all_waypoints[i+1]
            elif "post" in waypoint_type:
                focus_wp = self.all_waypoints[i-1]
            else:
                focus_wp = wp
            if focus_wp not in point_list:
                focus_info = data[0].object_informations[focus_wp]
                focus_type = focus_info['waypoint_type']
                # attention_id = waypoint_info["target_obj"]
                if "grasp" in focus_type:
                    attention_id = waypoint_info["target_obj"]
                    step_img_id = i
                    related_rotation = False
                else:
                    related_rotation = self.relative
                point_list.append(focus_wp)
                if self.renew_obs:
                    step_list.append([focus_wp, i, attention_id, related_rotation])
                else:
                    step_list.append([focus_wp, step_img_id, attention_id, related_rotation])
        
        for i, step in enumerate(step_list):
            # if i == 0:
            #     continue
            current_waypoint, index, attention_id, related_rotation = step
            obs = data[index]
            waypoint_info = obs.object_informations[current_waypoint]
            for name, obj in obs.object_informations.items():
                if "id" in obj and "waypoint" not in name:
                    if obj["id"] == attention_id:
                        target_obj = name
            front_rgb = obs.front_rgb
            wrist_rgb = obs.wrist_rgb
            left_rgb = obs.left_shoulder_rgb
            right_rgb = obs.right_shoulder_rgb
            overhead_rgb = obs.overhead_rgb

            front_point_cloud = obs.front_point_cloud
            wrist_point_cloud = obs.wrist_point_cloud
            left_point_cloud = obs.left_shoulder_point_cloud
            right_point_cloud = obs.right_shoulder_point_cloud
            overhead_point_cloud = obs.overhead_point_cloud

            colors = [front_rgb, wrist_rgb, left_rgb, right_rgb, overhead_rgb]
            pcds = [front_point_cloud, wrist_point_cloud, left_point_cloud, right_point_cloud, overhead_point_cloud]
            cmap, hmap = get_fused_heightmap(colors, pcds, bounds, pixel_size)
            cmaps.append(cmap)
            hmaps.append(hmap)
            lang = high_l+f" Step {num2words(i)}."
            if self.add_low_lang:
                lang += waypoint_info['low_level_descriptions']
            language_instructions.append(lang)
            # language_instructions.append(obs.object_informations[obs.current_waypoint_name]['low_level_descriptions'])
            target_point = waypoint_info["pose"][0]
            if related_rotation:
                prev_target_pose = obs.object_informations[step_list[i-1][0]]["pose"][0]
                prev_rot = R.from_quat(prev_target_pose[3:])
                current_rot = R.from_quat(target_point[3:])
                delta_r = prev_rot.inv()*current_rot
                target_point[3:] = delta_r.as_quat()
            target_points.append(target_point)
            attention_point = obs.object_informations[target_obj]["pose"]#[0]
            attention_points.append(attention_point)
        cmaps = np.stack(cmaps, axis=0)
        hmaps = np.tile((np.stack(hmaps, axis=0))[..., None], (1,1,1,3))
        img = np.concatenate([cmaps, hmaps], axis=-1)
        attention_points = np.stack(attention_points, axis=0)
        target_points = np.stack(target_points, axis=0)
        output_dict = {
            "img":img,
            "attention_points":attention_points,
            "target_points": target_points,
            "language": language_instructions,
            "bounds":bounds,
            "pixel_size":pixel_size,
            "episode":str(episode),
            'valid':1
        }
        if (attention_points[:, :2]>bounds[:2,1]).any() or (attention_points[:, :2]<bounds[:2,0]).any() \
            or (target_points[:, :2]>bounds[:2,1]).any() or (target_points[:, :2]<bounds[:2,0]).any():
            output_dict['valid']=0
        return output_dict

    @staticmethod
    def depth2normal(d_im):
        d_im = d_im.astype("float32")
        # zy, zx = np.gradient(d_im)
        # You may also consider using Sobel to get a joint Gaussian smoothing and differentation
        # to reduce noise
        zx = cv2.Sobel(d_im, cv2.CV_64F, 1, 0, ksize=3)
        zy = cv2.Sobel(d_im, cv2.CV_64F, 0, 1, ksize=3)
        normal = np.dstack((-zx, -zy, np.ones_like(d_im)))
        n = np.linalg.norm(normal, axis=2)
        normal[:, :, 0] /= n
        normal[:, :, 1] /= n
        normal[:, :, 2] /= n
        # offset and rescale values to be in 0-1
        normal += 1
        normal /= 2
        return normal

    @staticmethod
    def extract_bboxes(mask):
        """Compute bounding boxes from masks.
        mask: [height, width, num_instances]. Mask pixels are either 1 or 0.
        Returns: bbox array [num_instances, (y1, x1, y2, x2)].
        """
        horizontal_indicies = np.where(np.any(mask, axis=0))[0]
        vertical_indicies = np.where(np.any(mask, axis=1))[0]
        if horizontal_indicies.shape[0]:
            x1, x2 = horizontal_indicies[[0, -1]]
            y1, y2 = vertical_indicies[[0, -1]]
            # x2 and y2 should not be part of the box. Increment by 1.
            x2 += 1
            y2 += 1
        else:
            # No mask for this instance. Might happen due to
            # resizing or cropping. Set bbox to zeros
            x1, x2, y1, y2 = 0, 0, 0, 0
        return np.array([x1, y1, x2, y2])

    def __len__(self):
        return len(self.episode_list)