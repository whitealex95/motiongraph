# GMR-retargeted LAFAN1 → G1 (locomotion clips)

The three GenoView locomotion clips (`walk1_subject5`, `run1_subject5`,
`pushAndStumble1_subject5`) retargeted to the Unitree G1 with **GMR** (General Motion
Retargeting, ~/Projects/GMR), copied from `GMR/motion_data/lafan1-g1/`.

Each `.pkl` is a dict: `root_pos (N,3)`, `root_rot (N,4)` quaternion **xyzw**,
`dof_pos (N,29)`, `fps=30`. GMR targets `g1_mocap_29dof.xml`, whose 29 hinge-joint order
is identical to our menagerie `unitree_g1` model, so the data maps straight to our qpos
(`data.py:_gmr_to_qpos`). These replace the lvhaidong CSV walk/run as the locomotion source
(`config.GMR_LOCO_CLIPS`), adding pushAndStumble (which lvhaidong never retargeted).
