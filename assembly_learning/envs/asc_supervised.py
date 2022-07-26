from assembly_learning.envs import AssemblyCloudFeatures

class AssemblyCloudSupervised(AssemblyCloudFeatures):

    def __init__(self, **kwargs):
        self.conn_site_dict2name = {}
        super(AssemblyCloudSupervised, self).__init__(**kwargs)
        
    def _get_conn_site_dict(self):
        conn_site_dict = {}
        for _, obj_name in enumerate(self._object_name_list):
            conn_site_dict[obj_name] = {}
            self.conn_site_dict2name[obj_name] = {}
            k = 0
            for _, site_name in enumerate(self._conn_site2body_dict.keys()):
                if 'conn_site' in site_name and self._conn_site2body_dict[site_name] == obj_name:
                    rel_pose = self._conn_site_rel_pose_dict[site_name]
                    conn_site_dict[obj_name][k] = rel_pose
                    self.conn_site_dict2name[obj_name][k] = site_name
                    k+=1
        return conn_site_dict
    
    def step(self, a):
        self._num_step += 1
        change = self._step_pc(a)
        reward, done, info = self._compute_reward(a, change)
        ob = self._get_obs()
        return ob, reward, done, info  

    def _compute_reward(self, a, change):
        reward = -1
        done = self._is_done()
        target_obj_name, source_obj_name, target_conn_site, source_conn_site = self.pair_dict[int(a)]
        target_conn_site_name = self.conn_site_dict2name[target_obj_name][target_conn_site]
        source_conn_site_name = self.conn_site_dict2name[source_obj_name][source_conn_site]
        target_cs_pair = target_conn_site_name.split(",")[0].split("-")
        source_cs_pair = source_conn_site_name.split(",")[0].split("-")
        if change == "no_change":
            reward = -1
            info = {"reward": reward, "supervision": "no_change"}
        elif "fake" in target_conn_site_name or "fake" in source_conn_site_name:
            reward = -5
            info = {"reward": reward, "supervision": "fake"}
        elif target_cs_pair[0] == source_cs_pair[1] and target_cs_pair[1] == source_cs_pair[0]:
            reward = 5
            info = {"reward": reward, "supervision": "correct"}
        else:
            reward = -5
            info = {"reward": reward, "supervision": "wrong"}
        return reward, done, info



