from gym.envs.registration import register

register(
    id='assembly-cloud-partial-threshold-lack-v0',
    entry_point='assembly_learning.envs:AssemblyCloudFeatures',
    kwargs={
        "as_type": 360,
        "rew_type": "partial_th",
        "points_threshold": 0.015,
        "num_threshold": 55,
        "main_object": "4_part4",
        "pc_sample_size": 5000,
        "max_ep_length": 10,
        "furniture_id": 10,
    },          
        )
register(
    id='assembly-cloud-partial-threshold-lack-end-v0',
    entry_point='assembly_learning.envs:AssemblyCloudFeatures',
    kwargs={
        "as_type": 360,
        "rew_type": "partial_th_end",
        "points_threshold": 0.015,
        "num_threshold": 10,
        "main_object": "4_part4",
        "pc_sample_size": 5000,
        "max_ep_length": 10,
        "furniture_id": 10,
    },          
        )
register(
    id='assembly-cloud-partial-threshold-lack-corr-v0',
    entry_point='assembly_learning.envs:AssemblyCloudFeatures',
    kwargs={
        "as_type": 360,
        "rew_type": "partial_th_corr",
        "points_threshold": 0.015,
        "num_threshold": 10,
        "main_object": "4_part4",
        "pc_sample_size": 5000,
        "max_ep_length": 10,
        "furniture_id": 10,
    },          
        )    

register(
    id='assembly-cloud-partial-threshold-lack-comp-v0',
    entry_point='assembly_learning.envs:AssemblyCloudFeatures',
    kwargs={
        "as_type": 360,
        "rew_type": "partial_th_comp",
        "points_threshold": 0.015,
        "num_threshold": 10,
        "main_object": "4_part4",
        "pc_sample_size": 5000,
        "max_ep_length": 10,
        "furniture_id": 10,
    },          
        )       
register(
    id='assembly-cloud-partial-chamfer-v0',
    entry_point='assembly_learning.envs:AssemblyCloudFeatures',
    kwargs={
        "as_type": 360,
        "rew_type": "partial_chamfer",
        "main_object": "5_table_top",
        "pc_sample_size": 5000,
        "max_ep_length": 10,
        "furniture_id": 10,
    },          
        )

register(
    id='assembly-cloud-partial-chamfer-change-v0',
    entry_point='assembly_learning.envs:AssemblyCloudFeatures',
    kwargs={
        "as_type": 360,
        "rew_type": "partial_chamfer_change",
        "main_object": "5_table_top",
        "pc_sample_size": 5000,
        "max_ep_length": 10,
        "furniture_id": 10,
    },          
        )
register(
    id='assembly-cloud-end-chamfer-v0',
    entry_point='assembly_learning.envs:AssemblyCloudFeatures',
    kwargs={
        "as_type": 360,
        "rew_type": "end_chamfer",
        "pc_sample_size": 5000,
        "max_ep_length": 10,
        "furniture_id": 10,
    },          
        )

register(
    id='assembly-cloud-partial-end-chamfer-v0',
    entry_point='assembly_learning.envs:AssemblyCloudFeatures',
    kwargs={
        "as_type": 360,
        "rew_type": "partial_end_chamfer",
        "main_object": "5_table_top",
        "pc_sample_size": 5000,
        "max_ep_length": 10,
        "furniture_id": 10,
    },          
        )

register(
    id='assembly-cloud-partial-threshold-ivar-v0',
    entry_point='assembly_learning.envs:AssemblyCloudFeatures',
    kwargs={
        "as_type": 360,
        "rew_type": "partial_th",
        "points_threshold": 0.015,
        "num_threshold": 10,
        "main_object": "0_part0",
        "pc_sample_size": 5000,
        "max_ep_length": 10,
        "furniture_id": 4,
    },          
        )

register(
    id='assembly-cloud-partial-threshold-ivar-end-v0',
    entry_point='assembly_learning.envs:AssemblyCloudFeatures',
    kwargs={
        "as_type": 360,
        "rew_type": "partial_th_end",
        "points_threshold": 0.015,
        "num_threshold": 10,
        "main_object": "0_part0",
        "pc_sample_size": 5000,
        "max_ep_length": 10,
        "furniture_id": 4,
    },          
        )

register(
    id='assembly-cloud-partial-threshold-ivar-corr-v0',
    entry_point='assembly_learning.envs:AssemblyCloudFeatures',
    kwargs={
        "as_type": 360,
        "rew_type": "partial_th_corr",
        "points_threshold": 0.015,
        "num_threshold": 10,
        "main_object": "0_part0",
        "pc_sample_size": 5000,
        "max_ep_length": 10,
        "furniture_id": 4,
    },          
        )

register(
    id='assembly-cloud-partial-threshold-ivar-comp-v0',
    entry_point='assembly_learning.envs:AssemblyCloudFeatures',
    kwargs={
        "as_type": 360,
        "rew_type": "partial_th_comp",
        "points_threshold": 0.015,
        "num_threshold": 10,
        "main_object": "0_part0",
        "pc_sample_size": 5000,
        "max_ep_length": 10,
        "furniture_id": 4,
    },          
        )
register(
    id='assembly-cloud-partial-threshold-agne-v0',
    entry_point='assembly_learning.envs:AssemblyCloudFeatures',
    kwargs={
        "as_type": 360,
        "rew_type": "partial_th",
        "points_threshold": 0.015,
        "num_threshold": 10,
        "main_object": "2_part2",
        "pc_sample_size": 5000,
        "max_ep_length": 10,
        "furniture_id": 1,
    },          
        )
register(
    id='assembly-cloud-partial-threshold-agne-end-v0',
    entry_point='assembly_learning.envs:AssemblyCloudFeatures',
    kwargs={
        "as_type": 360,
        "rew_type": "partial_th_end",
        "points_threshold": 0.015,
        "num_threshold": 10,
        "main_object": "2_part2",
        "pc_sample_size": 5000,
        "max_ep_length": 10,
        "furniture_id": 1,
    },          
        )
register(
    id='assembly-cloud-partial-threshold-agne-corr-v0',
    entry_point='assembly_learning.envs:AssemblyCloudFeatures',
    kwargs={
        "as_type": 360,
        "rew_type": "partial_th_corr",
        "points_threshold": 0.015,
        "num_threshold": 10,
        "main_object": "2_part2",
        "pc_sample_size": 5000,
        "max_ep_length": 10,
        "furniture_id": 1,
    },          
        )
register(
    id='assembly-cloud-partial-threshold-agne-comp-v0',
    entry_point='assembly_learning.envs:AssemblyCloudFeatures',
    kwargs={
        "as_type": 360,
        "rew_type": "partial_th_comp",
        "points_threshold": 0.015,
        "num_threshold": 10,
        "main_object": "2_part2",
        "pc_sample_size": 5000,
        "max_ep_length": 10,
        "furniture_id": 1,
    },          
        )
register(
    id='assembly-cloud-partial-threshold-bernhard-v0',
    entry_point='assembly_learning.envs:AssemblyCloudFeatures',
    kwargs={
        "as_type": 360,
        "rew_type": "partial_th",
        "points_threshold": 0.015,
        "num_threshold": 10,
        "main_object": "2_part2",
        "pc_sample_size": 5000,
        "max_ep_length": 10,
        "furniture_id": 2,
    },          
        )
register(
    id='assembly-cloud-partial-threshold-bernhard-end-v0',
    entry_point='assembly_learning.envs:AssemblyCloudFeatures',
    kwargs={
        "as_type": 360,
        "rew_type": "partial_th_end",
        "points_threshold": 0.015,
        "num_threshold": 10,
        "main_object": "2_part2",
        "pc_sample_size": 5000,
        "max_ep_length": 10,
        "furniture_id": 2,
    },          
        )
register(
    id='assembly-cloud-partial-threshold-bernhard-corr-v0',
    entry_point='assembly_learning.envs:AssemblyCloudFeatures',
    kwargs={
        "as_type": 360,
        "rew_type": "partial_th_corr",
        "points_threshold": 0.015,
        "num_threshold": 10,
        "main_object": "2_part2",
        "pc_sample_size": 5000,
        "max_ep_length": 10,
        "furniture_id": 2,
    },          
        )

register(
    id='assembly-cloud-partial-threshold-bernhard-comp-v0',
    entry_point='assembly_learning.envs:AssemblyCloudFeatures',
    kwargs={
        "as_type": 360,
        "rew_type": "partial_th_comp",
        "points_threshold": 0.015,
        "num_threshold": 10,
        "main_object": "2_part2",
        "pc_sample_size": 5000,
        "max_ep_length": 10,
        "furniture_id": 2,
    },          
        )
register(
    id='assembly-cloud-furniture-bernhard-v0',
    entry_point='assembly_learning.envs:AssemblyCloudFurniture',
    kwargs={
        "as_type": 360,
        "rew_type": "partial_th",
        "points_threshold": 0.015,
        "num_threshold": 10,
        "main_object": "2_part2",
        "pc_sample_size": 5000,
        "max_ep_length": 10,
        "furniture_id": 2,
        "use_mujoco": True,
        # "rp_file_name": "/home/kovan/git/assembly_learning/bernhard_rp.npy"
    },          
        )

register(
    id='assembly-cloud-furniture-bernhard-end-v0',
    entry_point='assembly_learning.envs:AssemblyCloudFurniture',
    kwargs={
        "as_type": 360,
        "rew_type": "partial_th_end",
        "points_threshold": 0.015,
        "num_threshold": 10,
        "main_object": "2_part2",
        "pc_sample_size": 5000,
        "max_ep_length": 10,
        "furniture_id": 2,
        "use_mujoco": True,
        # "rp_file_name": "/home/kovan/git/assembly_learning/bernhard_rp.npy"
    },          
        )

register(
    id='assembly-cloud-partial-threshold-bertil-v0',
    entry_point='assembly_learning.envs:AssemblyCloudFeatures',
    kwargs={
        "as_type": 360,
        "rew_type": "partial_th",
        "points_threshold": 0.015,
        "num_threshold": 10,
        "main_object": "0_part0",
        "pc_sample_size": 5000,
        "max_ep_length": 10,
        "furniture_id": 3,
    },          
        )

register(
    id='assembly-cloud-partial-threshold-bertil-end-v0',
    entry_point='assembly_learning.envs:AssemblyCloudFeatures',
    kwargs={
        "as_type": 360,
        "rew_type": "partial_th_end",
        "points_threshold": 0.015,
        "num_threshold": 10,
        "main_object": "0_part0",
        "pc_sample_size": 5000,
        "max_ep_length": 10,
        "furniture_id": 3,
    },          
        )
register(
    id='assembly-cloud-partial-threshold-bertil-corr-v0',
    entry_point='assembly_learning.envs:AssemblyCloudFeatures',
    kwargs={
        "as_type": 360,
        "rew_type": "partial_th_corr",
        "points_threshold": 0.015,
        "num_threshold": 10,
        "main_object": "0_part0",
        "pc_sample_size": 5000,
        "max_ep_length": 10,
        "furniture_id": 3,
    },          
        )
register(
    id='assembly-cloud-partial-threshold-bertil-comp-v0',
    entry_point='assembly_learning.envs:AssemblyCloudFeatures',
    kwargs={
        "as_type": 360,
        "rew_type": "partial_th_comp",
        "points_threshold": 0.015,
        "num_threshold": 10,
        "main_object": "0_part0",
        "pc_sample_size": 5000,
        "max_ep_length": 10,
        "furniture_id": 3,
    },          
        )
register(
    id='assembly-cloud-furniture-bertil-v0',
    entry_point='assembly_learning.envs:AssemblyCloudFurniture',
    kwargs={
        "as_type": 360,
        "rew_type": "partial_th",
        "points_threshold": 0.015,
        "num_threshold": 50,
        "main_object": "0_part0",
        "pc_sample_size": 5000,
        "max_ep_length": 10,
        "furniture_id": 3,
        "use_mujoco": True,
    },          
        )

register(
    id='assembly-cloud-furniture-bertil-end-v0',
    entry_point='assembly_learning.envs:AssemblyCloudFurniture',
    kwargs={
        "as_type": 360,
        "rew_type": "partial_th_end",
        "points_threshold": 0.015,
        "num_threshold": 50,
        "main_object": "0_part0",
        "pc_sample_size": 5000,
        "max_ep_length": 10,
        "furniture_id": 3,
        "use_mujoco": True,
    },          
        )

register(
    id='assembly-cloud-partial-threshold-mikael-v0',
    entry_point='assembly_learning.envs:AssemblyCloudFeatures',
    kwargs={
        "as_type": 360,
        "rew_type": "partial_th",
        "points_threshold": 0.015,
        "num_threshold": 155,
        "main_object": "3_part3",
        "pc_sample_size": 5000,
        "max_ep_length": 10,
        "furniture_id": 5,
    },          
        )

register(
    id='assembly-cloud-partial-threshold-mikael-end-v0',
    entry_point='assembly_learning.envs:AssemblyCloudFeatures',
    kwargs={
        "as_type": 360,
        "rew_type": "partial_th_end",
        "points_threshold": 0.015,
        "num_threshold": 155,
        "main_object": "3_part3",
        "pc_sample_size": 5000,
        "max_ep_length": 10,
        "furniture_id": 5,
    },          
        )
register(
    id='assembly-cloud-partial-threshold-mikael-corr-v0',
    entry_point='assembly_learning.envs:AssemblyCloudFeatures',
    kwargs={
        "as_type": 360,
        "rew_type": "partial_th_corr",
        "points_threshold": 0.015,
        "num_threshold": 155,
        "main_object": "3_part3",
        "pc_sample_size": 5000,
        "max_ep_length": 10,
        "furniture_id": 5,
    },          
        )
register(
    id='assembly-cloud-partial-threshold-mikael-comp-v0',
    entry_point='assembly_learning.envs:AssemblyCloudFeatures',
    kwargs={
        "as_type": 360,
        "rew_type": "partial_th_comp",
        "points_threshold": 0.015,
        "num_threshold": 155,
        "main_object": "3_part3",
        "pc_sample_size": 5000,
        "max_ep_length": 10,
        "furniture_id": 5,
    },          
        )
register(
    id='assembly-cloud-furniture-mikael-v0',
    entry_point='assembly_learning.envs:AssemblyCloudFurniture',
    kwargs={
        "as_type": 360,
        "rew_type": "partial_th",
        "points_threshold": 0.015,
        "num_threshold": 155,
        "main_object": "3_part3",
        "pc_sample_size": 5000,
        "max_ep_length": 10,
        "furniture_id": 5,
        "use_mujoco": True,
    },          
        )

register(
    id='assembly-cloud-furniture-mikael-end-v0',
    entry_point='assembly_learning.envs:AssemblyCloudFurniture',
    kwargs={
        "as_type": 360,
        "rew_type": "partial_th_end",
        "points_threshold": 0.015,
        "num_threshold": 155,
        "main_object": "3_part3",
        "pc_sample_size": 5000,
        "max_ep_length": 10,
        "furniture_id": 5,
        "use_mujoco": True,
    },          
        )

register(
    id='assembly-cloud-partial-threshold-sivar-v0',
    entry_point='assembly_learning.envs:AssemblyCloudFeatures',
    kwargs={
        "as_type": 360,
        "rew_type": "partial_th",
        "points_threshold": 0.015,
        "num_threshold": 1,
        "main_object": "0_part0",
        "pc_sample_size": 5000,
        "max_ep_length": 10,
        "furniture_id": 6,
    },          
        )
register(
    id='assembly-cloud-partial-threshold-sivar-end-v0',
    entry_point='assembly_learning.envs:AssemblyCloudFeatures',
    kwargs={
        "as_type": 360,
        "rew_type": "partial_th_end",
        "points_threshold": 0.015,
        "num_threshold": 40,
        "main_object": "0_part0",
        "pc_sample_size": 5000,
        "max_ep_length": 10,
        "furniture_id": 6,
    },          
        )
register(
    id='assembly-cloud-partial-threshold-sivar-corr-v0',
    entry_point='assembly_learning.envs:AssemblyCloudFeatures',
    kwargs={
        "as_type": 360,
        "rew_type": "partial_th_corr",
        "points_threshold": 0.015,
        "num_threshold": 40,
        "main_object": "0_part0",
        "pc_sample_size": 5000,
        "max_ep_length": 10,
        "furniture_id": 6,
    },          
        )
register(
    id='assembly-cloud-partial-threshold-sivar-comp-v0',
    entry_point='assembly_learning.envs:AssemblyCloudFeatures',
    kwargs={
        "as_type": 360,
        "rew_type": "partial_th_comp",
        "points_threshold": 0.015,
        "num_threshold": 40,
        "main_object": "0_part0",
        "pc_sample_size": 5000,
        "max_ep_length": 10,
        "furniture_id": 6,
    },          
        )
register(
    id='assembly-cloud-furniture-sivar-v0',
    entry_point='assembly_learning.envs:AssemblyCloudFurniture',
    kwargs={
        "as_type": 360,
        "rew_type": "partial_th",
        "points_threshold": 0.015,
        "num_threshold": 40,
        "main_object": "0_part0",
        "pc_sample_size": 5000,
        "max_ep_length": 10,
        "furniture_id": 6,
        "use_mujoco": True,
    },          
        )

register(
    id='assembly-cloud-furniture-sivar-end-v0',
    entry_point='assembly_learning.envs:AssemblyCloudFurniture',
    kwargs={
        "as_type": 360,
        "rew_type": "partial_th_end",
        "points_threshold": 0.015,
        "num_threshold": 40,
        "main_object": "0_part0",
        "pc_sample_size": 5000,
        "max_ep_length": 10,
        "furniture_id": 6,
        "use_mujoco": True,
    },          
        )

register(
    id='assembly-cloud-partial-threshold-sliden-v0',
    entry_point='assembly_learning.envs:AssemblyCloudFeatures',
    kwargs={
        "as_type": 360,
        "rew_type": "partial_th",
        "points_threshold": 0.015,
        "num_threshold": 10,
        "main_object": "9_part9",
        "pc_sample_size": 5000,
        "max_ep_length": 10,
        "furniture_id": 7,
    },          
        )
register(
    id='assembly-cloud-partial-threshold-sliden-end-v0',
    entry_point='assembly_learning.envs:AssemblyCloudFeatures',
    kwargs={
        "as_type": 360,
        "rew_type": "partial_th_end",
        "points_threshold": 0.015,
        "num_threshold": 10,
        "main_object": "9_part9",
        "pc_sample_size": 5000,
        "max_ep_length": 10,
        "furniture_id": 7,
    },          
        )
register(
    id='assembly-cloud-partial-threshold-sliden-corr-v0',
    entry_point='assembly_learning.envs:AssemblyCloudFeatures',
    kwargs={
        "as_type": 360,
        "rew_type": "partial_th_corr",
        "points_threshold": 0.015,
        "num_threshold": 10,
        "main_object": "9_part9",
        "pc_sample_size": 5000,
        "max_ep_length": 10,
        "furniture_id": 7,
    },          
        )
register(
    id='assembly-cloud-partial-threshold-sliden-comp-v0',
    entry_point='assembly_learning.envs:AssemblyCloudFeatures',
    kwargs={
        "as_type": 360,
        "rew_type": "partial_th_comp",
        "points_threshold": 0.015,
        "num_threshold": 10,
        "main_object": "9_part9",
        "pc_sample_size": 5000,
        "max_ep_length": 10,
        "furniture_id": 7,
    },          
        )
register(
    id='assembly-cloud-furniture-sliden-v0',
    entry_point='assembly_learning.envs:AssemblyCloudFurniture',
    kwargs={
        "as_type": 360,
        "rew_type": "partial_th",
        "points_threshold": 0.015,
        "num_threshold": 20,
        "main_object": "9_part9",
        "pc_sample_size": 5000,
        "max_ep_length": 10,
        "furniture_id": 7,
        "use_mujoco": False,
        "rp_file_name": "sliden_rp.npy"
    },          
        )

register(
    id='assembly-cloud-furniture-sliden-end-v0',
    entry_point='assembly_learning.envs:AssemblyCloudFurniture',
    kwargs={
        "as_type": 360,
        "rew_type": "partial_th_end",
        "points_threshold": 0.015,
        "num_threshold": 20,
        "main_object": "9_part9",
        "pc_sample_size": 5000,
        "max_ep_length": 10,
        "furniture_id": 7,
        "use_mujoco": False,
        "rp_file_name": "sliden_rp.npy"
    },          
        )

register(
    id='assembly-cloud-partial-threshold-swivel-v0',
    entry_point='assembly_learning.envs:AssemblyCloudFeatures',
    kwargs={
        "as_type": 360,
        "rew_type": "partial_th",
        "points_threshold": 0.015,
        "num_threshold": 10,
        "main_object": "3_chair_seat",
        "pc_sample_size": 5000,
        "max_ep_length": 10,
        "furniture_id": 8,
    },          
        )

register(
    id='assembly-cloud-partial-threshold-swivel-end-v0',
    entry_point='assembly_learning.envs:AssemblyCloudFeatures',
    kwargs={
        "as_type": 360,
        "rew_type": "partial_th_end",
        "points_threshold": 0.015,
        "num_threshold": 10,
        "main_object": "3_chair_seat",
        "pc_sample_size": 5000,
        "max_ep_length": 10,
        "furniture_id": 8,
    },          
        )

register(
    id='assembly-cloud-partial-threshold-swivel-corr-v0',
    entry_point='assembly_learning.envs:AssemblyCloudFeatures',
    kwargs={
        "as_type": 360,
        "rew_type": "partial_th_corr",
        "points_threshold": 0.015,
        "num_threshold": 10,
        "main_object": "3_chair_seat",
        "pc_sample_size": 5000,
        "max_ep_length": 10,
        "furniture_id": 8,
    },          
        )

register(
    id='assembly-cloud-partial-threshold-swivel-comp-v0',
    entry_point='assembly_learning.envs:AssemblyCloudFeatures',
    kwargs={
        "as_type": 360,
        "rew_type": "partial_th_comp",
        "points_threshold": 0.015,
        "num_threshold": 10,
        "main_object": "3_chair_seat",
        "pc_sample_size": 5000,
        "max_ep_length": 10,
        "furniture_id": 8,
    },          
        )
register(
    id='assembly-cloud-furniture-swivel-v0',
    entry_point='assembly_learning.envs:AssemblyCloudFurniture',
    kwargs={
        "as_type": 360,
        "rew_type": "partial_th",
        "points_threshold": 0.015,
        "num_threshold": 10,
        "main_object": "3_chair_seat",
        "pc_sample_size": 5000,
        "max_ep_length": 10,
        "furniture_id": 8,
        "use_mujoco": True,
    },          
        )

register(
    id='assembly-cloud-furniture-swivel-end-v0',
    entry_point='assembly_learning.envs:AssemblyCloudFurniture',
    kwargs={
        "as_type": 360,
        "rew_type": "partial_th_end",
        "points_threshold": 0.015,
        "num_threshold": 10,
        "main_object": "3_chair_seat",
        "pc_sample_size": 5000,
        "max_ep_length": 10,
        "furniture_id": 8,
        "use_mujoco": True,
    },          
        )

register(
    id='assembly-cloud-partial-threshold-klubbo-v0',
    entry_point='assembly_learning.envs:AssemblyCloudFeatures',
    kwargs={
        "as_type": 360,
        "rew_type": "partial_th",
        "points_threshold": 0.015,
        "num_threshold": 60,
        "main_object": "5_part5",
        "pc_sample_size": 5000,
        "max_ep_length": 10,
        "furniture_id": 9,
    },          
        )

register(
    id='assembly-cloud-partial-threshold-klubbo-end-v0',
    entry_point='assembly_learning.envs:AssemblyCloudFeatures',
    kwargs={
        "as_type": 360,
        "rew_type": "partial_th_end",
        "points_threshold": 0.015,
        "num_threshold": 60,
        "main_object": "5_part5",
        "pc_sample_size": 5000,
        "max_ep_length": 10,
        "furniture_id": 9,
    },          
        )
register(
    id='assembly-cloud-partial-threshold-klubbo-corr-v0',
    entry_point='assembly_learning.envs:AssemblyCloudFeatures',
    kwargs={
        "as_type": 360,
        "rew_type": "partial_th_corr",
        "points_threshold": 0.015,
        "num_threshold": 60,
        "main_object": "5_part5",
        "pc_sample_size": 5000,
        "max_ep_length": 10,
        "furniture_id": 9,
    },          
        )
register(
    id='assembly-cloud-partial-threshold-klubbo-comp-v0',
    entry_point='assembly_learning.envs:AssemblyCloudFeatures',
    kwargs={
        "as_type": 360,
        "rew_type": "partial_th_comp",
        "points_threshold": 0.015,
        "num_threshold": 60,
        "main_object": "5_part5",
        "pc_sample_size": 5000,
        "max_ep_length": 10,
        "furniture_id": 9,
    },          
        )
register(
    id='assembly-cloud-furniture-klubbo-v0',
    entry_point='assembly_learning.envs:AssemblyCloudFurniture',
    kwargs={
        "as_type": 360,
        "rew_type": "partial_th",
        "points_threshold": 0.015,
        "num_threshold": 10,
        "main_object": "5_part5",
        "pc_sample_size": 5000,
        "max_ep_length": 10,
        "furniture_id": 9,
        "use_mujoco": True,
    },          
        )

register(
    id='assembly-cloud-furniture-klubbo-end-v0',
    entry_point='assembly_learning.envs:AssemblyCloudFurniture',
    kwargs={
        "as_type": 360,
        "rew_type": "partial_th_end",
        "points_threshold": 0.015,
        "num_threshold": 10,
        "main_object": "5_part5",
        "pc_sample_size": 5000,
        "max_ep_length": 10,
        "furniture_id": 9,
        "use_mujoco": True,
    },          
        )

register(
    id='assembly-cloud-partial-threshold-tvunit-v0',
    entry_point='assembly_learning.envs:AssemblyCloudFeatures',
    kwargs={
        "as_type": 360,
        "rew_type": "partial_th",
        "points_threshold": 0.015,
        "num_threshold": 150,
        "main_object": "4_part4",
        "pc_sample_size": 10000,
        "max_ep_length": 10,
        "furniture_id": 11,
    },          
        )
register(
    id='assembly-cloud-partial-threshold-tvunit-end-v0',
    entry_point='assembly_learning.envs:AssemblyCloudFeatures',
    kwargs={
        "as_type": 360,
        "rew_type": "partial_th_end",
        "points_threshold": 0.015,
        "num_threshold": 150,
        "main_object": "4_part4",
        "pc_sample_size": 5000,
        "max_ep_length": 10,
        "furniture_id": 11,
    },          
        )
register(
    id='assembly-cloud-partial-threshold-tvunit-corr-v0',
    entry_point='assembly_learning.envs:AssemblyCloudFeatures',
    kwargs={
        "as_type": 360,
        "rew_type": "partial_th_corr",
        "points_threshold": 0.015,
        "num_threshold": 150,
        "main_object": "4_part4",
        "pc_sample_size": 5000,
        "max_ep_length": 10,
        "furniture_id": 11,
    },          
        )
register(
    id='assembly-cloud-partial-threshold-tvunit-comp-v0',
    entry_point='assembly_learning.envs:AssemblyCloudFeatures',
    kwargs={
        "as_type": 360,
        "rew_type": "partial_th_comp",
        "points_threshold": 0.015,
        "num_threshold": 150,
        "main_object": "4_part4",
        "pc_sample_size": 5000,
        "max_ep_length": 10,
        "furniture_id": 11,
    },          
        )
register(
    id='assembly-cloud-furniture-tvunit-v0',
    entry_point='assembly_learning.envs:AssemblyCloudFurniture',
    kwargs={
        "as_type": 360,
        "rew_type": "partial_th",
        "points_threshold": 0.015,
        "num_threshold": 150,
        "main_object": "4_part4",
        "pc_sample_size": 5000,
        "max_ep_length": 10,
        "furniture_id": 11,
        "use_mujoco": True,
    },          
        )

register(
    id='assembly-cloud-furniture-tvunit-end-v0',
    entry_point='assembly_learning.envs:AssemblyCloudFurniture',
    kwargs={
        "as_type": 360,
        "rew_type": "partial_th_end",
        "points_threshold": 0.015,
        "num_threshold": 150,
        "main_object": "4_part4",
        "pc_sample_size": 5000,
        "max_ep_length": 10,
        "furniture_id": 11,
        "use_mujoco": True,
    },          
        )

register(
    id='assembly-cloud-furniture-ivar-v0',
    entry_point='assembly_learning.envs:AssemblyCloudFurniture',
    kwargs={
        "as_type": 360,
        "rew_type": "partial_th",
        "points_threshold": 0.015,
        "num_threshold": 20,
        "main_object": "0_part0",
        "pc_sample_size": 5000,
        "max_ep_length": 10,
        "furniture_id": 4,
        "use_mujoco": True,
    },          
        )

register(
    id='assembly-cloud-furniture-ivar-end-v0',
    entry_point='assembly_learning.envs:AssemblyCloudFurniture',
    kwargs={
        "as_type": 360,
        "rew_type": "partial_th_end",
        "points_threshold": 0.015,
        "num_threshold": 20,
        "main_object": "0_part0",
        "pc_sample_size": 5000,
        "max_ep_length": 10,
        "furniture_id": 4,
        "use_mujoco": True,
    },          
        )

register(
    id='assembly-cloud-furniture-lack-v0',
    entry_point='assembly_learning.envs:AssemblyCloudFurniture',
    kwargs={
        "as_type": 360,
        "rew_type": "partial_th",
        "points_threshold": 0.015,
        "num_threshold": 55,
        "main_object": "4_part4",
        "pc_sample_size": 5000,
        "max_ep_length": 10,
        "furniture_id": 10,
        "use_mujoco": True,
    },      
)

register(
    id='assembly-cloud-furniture-lack-end-v0',
    entry_point='assembly_learning.envs:AssemblyCloudFurniture',
    kwargs={
        "as_type": 360,
        "rew_type": "partial_th_end",
        "points_threshold": 0.015,
        "num_threshold": 55,
        "main_object": "4_part4",
        "pc_sample_size": 5000,
        "max_ep_length": 10,
        "furniture_id": 10,
        "use_mujoco": True,
    },      
)

register(
    id='assembly-cloud-furniture-agne-v0',
    entry_point='assembly_learning.envs:AssemblyCloudFurniture',
    kwargs={
        "as_type": 360,
        "rew_type": "partial_th",
        "points_threshold": 0.015,
        "num_threshold": 10,
        "main_object": "2_part2",
        "pc_sample_size": 5000,
        "max_ep_length": 10,
        "furniture_id": 1,
        "use_mujoco": True,
    },          
        )

register(
    id='assembly-cloud-furniture-agne-end-v0',
    entry_point='assembly_learning.envs:AssemblyCloudFurniture',
    kwargs={
        "as_type": 360,
        "rew_type": "partial_th_end",
        "points_threshold": 0.015,
        "num_threshold": 10,
        "main_object": "2_part2",
        "pc_sample_size": 5000,
        "max_ep_length": 10,
        "furniture_id": 1,
        "use_mujoco": True,
    },          
        )

register(
    id='assembly-cloud-supervised-agne-v0',
    entry_point='assembly_learning.envs:AssemblyCloudSupervised',
    kwargs={
        "as_type": 360,
        "rew_type": "supervised",
        "pc_sample_size": 5000,
        "max_ep_length": 10,
        "furniture_id": 1,
    },          
        )

register(
    id='assembly-cloud-supervised-bernhard-v0',
    entry_point='assembly_learning.envs:AssemblyCloudSupervised',
    kwargs={
        "as_type": 360,
        "rew_type": "supervised",
        "pc_sample_size": 5000,
        "max_ep_length": 10,
        "furniture_id": 2,
    },          
        )

register(
    id='assembly-cloud-supervised-bertil-v0',
    entry_point='assembly_learning.envs:AssemblyCloudSupervised',
    kwargs={
        "as_type": 360,
        "rew_type": "supervised",
        "pc_sample_size": 5000,
        "max_ep_length": 10,
        "furniture_id": 3,
    },          
        )

register(
    id='assembly-cloud-supervised-ivar-v0',
    entry_point='assembly_learning.envs:AssemblyCloudSupervised',
    kwargs={
        "as_type": 360,
        "rew_type": "supervised",
        "pc_sample_size": 5000,
        "max_ep_length": 10,
        "furniture_id": 4,
    },          
        )

register(
    id='assembly-cloud-supervised-mikael-v0',
    entry_point='assembly_learning.envs:AssemblyCloudSupervised',
    kwargs={
        "as_type": 360,
        "rew_type": "supervised",
        "pc_sample_size": 5000,
        "max_ep_length": 10,
        "furniture_id": 5,
    },          
        )

register(
    id='assembly-cloud-supervised-sivar-v0',
    entry_point='assembly_learning.envs:AssemblyCloudSupervised',
    kwargs={
        "as_type": 360,
        "rew_type": "supervised",
        "pc_sample_size": 5000,
        "max_ep_length": 10,
        "furniture_id": 6,
    },          
        )

register(
    id='assembly-cloud-supervised-sliden-v0',
    entry_point='assembly_learning.envs:AssemblyCloudSupervised',
    kwargs={
        "as_type": 360,
        "rew_type": "supervised",
        "pc_sample_size": 5000,
        "max_ep_length": 10,
        "furniture_id": 7,
    },          
        )

register(
    id='assembly-cloud-supervised-swivel-v0',
    entry_point='assembly_learning.envs:AssemblyCloudSupervised',
    kwargs={
        "as_type": 360,
        "rew_type": "supervised",
        "pc_sample_size": 5000,
        "max_ep_length": 10,
        "furniture_id": 8,
    },          
        )

register(
    id='assembly-cloud-supervised-klubbo-v0',
    entry_point='assembly_learning.envs:AssemblyCloudSupervised',
    kwargs={
        "as_type": 360,
        "rew_type": "supervised",
        "pc_sample_size": 5000,
        "max_ep_length": 10,
        "furniture_id": 9,
    },          
        )

register(
    id='assembly-cloud-supervised-lack-v0',
    entry_point='assembly_learning.envs:AssemblyCloudSupervised',
    kwargs={
        "as_type": 360,
        "rew_type": "supervised",
        "pc_sample_size": 5000,
        "max_ep_length": 10,
        "furniture_id": 10,
    },          
        )

register(
    id='assembly-cloud-supervised-tvunit-v0',
    entry_point='assembly_learning.envs:AssemblyCloudSupervised',
    kwargs={
        "as_type": 360,
        "rew_type": "supervised",
        "pc_sample_size": 5000,
        "max_ep_length": 10,
        "furniture_id": 11,
    },          
        )

register(
    id='assembly-cloud-partial-chamfer-agne-v0',
    entry_point='assembly_learning.envs:AssemblyCloudFeatures',
    kwargs={
        "as_type": 360,
        "rew_type": "partial_chamfer",
        "main_object": "2_part2",
        "pc_sample_size": 5000,
        "max_ep_length": 10,
        "furniture_id": 1,
    },          
        )

register(
    id='assembly-cloud-partial-chamfer-bernhard-v0',
    entry_point='assembly_learning.envs:AssemblyCloudFeatures',
    kwargs={
        "as_type": 360,
        "rew_type": "partial_chamfer",
        "main_object": "2_part2",
        "pc_sample_size": 5000,
        "max_ep_length": 10,
        "furniture_id": 2,
    },          
        )

register(
    id='assembly-cloud-partial-chamfer-bertil-v0',
    entry_point='assembly_learning.envs:AssemblyCloudFeatures',
    kwargs={
        "as_type": 360,
        "rew_type": "partial_chamfer",
        "main_object": "0_part0",
        "pc_sample_size": 5000,
        "max_ep_length": 10,
        "furniture_id": 3,
    },          
        )

register(
    id='assembly-cloud-partial-chamfer-ivar-v0',
    entry_point='assembly_learning.envs:AssemblyCloudFeatures',
    kwargs={
        "as_type": 360,
        "rew_type": "partial_chamfer",
        "main_object": "0_part0",
        "pc_sample_size": 5000,
        "max_ep_length": 10,
        "furniture_id": 4,
    },          
        )

register(
    id='assembly-cloud-partial-chamfer-mikael-v0',
    entry_point='assembly_learning.envs:AssemblyCloudFeatures',
    kwargs={
        "as_type": 360,
        "rew_type": "partial_chamfer",
        "main_object": "3_part3",
        "pc_sample_size": 5000,
        "max_ep_length": 10,
        "furniture_id": 5,
    },          
        )

register(
    id='assembly-cloud-partial-chamfer-sivar-v0',
    entry_point='assembly_learning.envs:AssemblyCloudFeatures',
    kwargs={
        "as_type": 360,
        "rew_type": "partial_chamfer",
        "main_object": "0_part0",
        "pc_sample_size": 5000,
        "max_ep_length": 10,
        "furniture_id": 6,
    },          
        )

register(
    id='assembly-cloud-partial-chamfer-sliden-v0',
    entry_point='assembly_learning.envs:AssemblyCloudFeatures',
    kwargs={
        "as_type": 360,
        "rew_type": "partial_chamfer",
        "main_object": "9_part9",
        "pc_sample_size": 5000,
        "max_ep_length": 10,
        "furniture_id": 7,
    },          
        )

register(
    id='assembly-cloud-partial-chamfer-swivel-v0',
    entry_point='assembly_learning.envs:AssemblyCloudFeatures',
    kwargs={
        "as_type": 360,
        "rew_type": "partial_chamfer",
        "main_object": "3_chair_seat",
        "pc_sample_size": 5000,
        "max_ep_length": 10,
        "furniture_id": 8,
    },          
        )

register(
    id='assembly-cloud-partial-chamfer-klubbo-v0',
    entry_point='assembly_learning.envs:AssemblyCloudFeatures',
    kwargs={
        "as_type": 360,
        "rew_type": "partial_chamfer",
        "main_object": "5_part5",
        "pc_sample_size": 5000,
        "max_ep_length": 10,
        "furniture_id": 9,
    },          
        )

register(
    id='assembly-cloud-partial-chamfer-lack-v0',
    entry_point='assembly_learning.envs:AssemblyCloudFeatures',
    kwargs={
        "as_type": 360,
        "rew_type": "partial_chamfer",
        "main_object": "4_part4",
        "pc_sample_size": 5000,
        "max_ep_length": 10,
        "furniture_id": 10,
    },      
)

register(
    id='assembly-cloud-partial-chamfer-tvunit-v0',
    entry_point='assembly_learning.envs:AssemblyCloudFeatures',
    kwargs={
        "as_type": 360,
        "rew_type": "partial_chamfer",
        "main_object": "4_part4",
        "pc_sample_size": 5000,
        "max_ep_length": 10,
        "furniture_id": 11,
    },          
        )

register(
    id='assembly-cloud-partial-chamfer-agne-change-v0',
    entry_point='assembly_learning.envs:AssemblyCloudFeatures',
    kwargs={
        "as_type": 360,
        "rew_type": "partial_chamfer_change",
        "main_object": "2_part2",
        "pc_sample_size": 5000,
        "max_ep_length": 10,
        "furniture_id": 1,
    },          
        )

register(
    id='assembly-cloud-partial-chamfer-bernhard-change-v0',
    entry_point='assembly_learning.envs:AssemblyCloudFeatures',
    kwargs={
        "as_type": 360,
        "rew_type": "partial_chamfer_change",
        "main_object": "2_part2",
        "pc_sample_size": 5000,
        "max_ep_length": 10,
        "furniture_id": 2,
    },          
        )

register(
    id='assembly-cloud-partial-chamfer-bertil-change-v0',
    entry_point='assembly_learning.envs:AssemblyCloudFeatures',
    kwargs={
        "as_type": 360,
        "rew_type": "partial_chamfer_change",
        "main_object": "0_part0",
        "pc_sample_size": 5000,
        "max_ep_length": 10,
        "furniture_id": 3,
    },          
        )

register(
    id='assembly-cloud-partial-chamfer-ivar-change-v0',
    entry_point='assembly_learning.envs:AssemblyCloudFeatures',
    kwargs={
        "as_type": 360,
        "rew_type": "partial_chamfer_change",
        "main_object": "0_part0",
        "pc_sample_size": 5000,
        "max_ep_length": 10,
        "furniture_id": 4,
    },          
        )

register(
    id='assembly-cloud-partial-chamfer-mikael-change-v0',
    entry_point='assembly_learning.envs:AssemblyCloudFeatures',
    kwargs={
        "as_type": 360,
        "rew_type": "partial_chamfer_change",
        "main_object": "3_part3",
        "pc_sample_size": 5000,
        "max_ep_length": 10,
        "furniture_id": 5,
    },          
        )

register(
    id='assembly-cloud-partial-chamfer-sivar-change-v0',
    entry_point='assembly_learning.envs:AssemblyCloudFeatures',
    kwargs={
        "as_type": 360,
        "rew_type": "partial_chamfer_change",
        "main_object": "0_part0",
        "pc_sample_size": 5000,
        "max_ep_length": 10,
        "furniture_id": 6,
    },          
        )

register(
    id='assembly-cloud-partial-chamfer-sliden-change-v0',
    entry_point='assembly_learning.envs:AssemblyCloudFeatures',
    kwargs={
        "as_type": 360,
        "rew_type": "partial_chamfer_change",
        "main_object": "9_part9",
        "pc_sample_size": 5000,
        "max_ep_length": 10,
        "furniture_id": 7,
    },          
        )

register(
    id='assembly-cloud-partial-chamfer-swivel-change-v0',
    entry_point='assembly_learning.envs:AssemblyCloudFeatures',
    kwargs={
        "as_type": 360,
        "rew_type": "partial_chamfer_change",
        "main_object": "3_chair_seat",
        "pc_sample_size": 5000,
        "max_ep_length": 10,
        "furniture_id": 8,
    },          
        )

register(
    id='assembly-cloud-partial-chamfer-klubbo-change-v0',
    entry_point='assembly_learning.envs:AssemblyCloudFeatures',
    kwargs={
        "as_type": 360,
        "rew_type": "partial_chamfer_change",
        "main_object": "5_part5",
        "pc_sample_size": 5000,
        "max_ep_length": 10,
        "furniture_id": 9,
    },          
        )

register(
    id='assembly-cloud-partial-chamfer-lack-change-v0',
    entry_point='assembly_learning.envs:AssemblyCloudFeatures',
    kwargs={
        "as_type": 360,
        "rew_type": "partial_chamfer_change",
        "main_object": "4_part4",
        "pc_sample_size": 5000,
        "max_ep_length": 10,
        "furniture_id": 10,
    },      
)

register(
    id='assembly-cloud-partial-chamfer-tvunit-change-v0',
    entry_point='assembly_learning.envs:AssemblyCloudFeatures',
    kwargs={
        "as_type": 360,
        "rew_type": "partial_chamfer_change",
        "main_object": "4_part4",
        "pc_sample_size": 5000,
        "max_ep_length": 10,
        "furniture_id": 11,
    },          
        )

register(
    id='assembly-cloud-goal-v0',
    entry_point='assembly_learning.envs:AssemblyCloudGoal',
    kwargs={
        "as_type": 360,
    },
)
register(
    id='assembly-cloud-adjacency-v0',
    entry_point='assembly_learning.envs:AssemblyCloudAdjacency',
    kwargs={
        "as_type": 360,
    },
)

register(
    id='mini-cloud-v0',
    entry_point='assembly_learning.envs:MiniCloud',
)