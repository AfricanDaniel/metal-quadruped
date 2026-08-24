#!/usr/bin/env python3
"""Interactively pose the dog in the MuJoCo viewer, then save the final
joint angles to a .txt file for reference (e.g. comparing a hand-set pose
against real hardware calibration).

Usage:
    python3 save_pose.py [--mjcf dog.mjcf.xml] [--out pose.txt]

Drag joints / let the model settle under gravity in the viewer window,
then close it -- qpos at the moment you close is what gets saved.

Note on posing calves: dog.mjcf.xml's raw calf joint is a plain
thigh-relative hinge (the real belt/pulley decoupling is only modeled in
software, in dog_gym/envs/dog_env.py's step()/_get_obs(), never in the
raw MJCF) -- so moving a thigh in this viewer will visually drag that
leg's calf mesh along with it, unlike the real, belt-decoupled robot.
The RAW qpos saved below is still exactly what STANDING_QPOS_DEG needs
(DogEnv.reset() writes it straight into qpos, no conversion) -- just pose
each leg's calf LAST (after that leg's thigh is already where you want
it) so its final visual position is correct. The printed ABSOLUTE column
below (raw - thigh, matching what the real motor's own encoder would
read, since calf_belt_sign=1 for every leg since AXIS_FLIP) is a sanity
check against known real-hardware angles, not what gets saved.
"""
import argparse
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np

HERE = Path(__file__).resolve().parent


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mjcf', type=Path, default=HERE / 'dog.mjcf.xml')
    p.add_argument('--out', type=Path, default=HERE / 'pose.txt')
    return p.parse_args()


def main():
    args = parse_args()
    m = mujoco.MjModel.from_xml_path(str(args.mjcf))
    d = mujoco.MjData(m)

    with mujoco.viewer.launch_passive(m, d) as viewer:
        while viewer.is_running():
            mujoco.mj_step(m, d)
            viewer.sync()

    # calf_belt_sign=1 for every leg since AXIS_FLIP (generate_dog_mjcf.py,
    # verified directly) -- absolute = raw - thigh, matching dog_env.py's
    # _get_obs() exactly. Pairing detected
    # generically by joint name, same convention as dog_env.py's
    # calf_idx/calf_thigh_idx and dog_deploy's find_calf_thigh_pairs().
    CALF_BELT_SIGN = 1.0
    qpos_rad_by_name = {}
    for jid in range(m.njnt):
        if m.jnt_type[jid] == mujoco.mjtJoint.mjJNT_HINGE:
            name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, jid)
            qpos_rad_by_name[name] = d.qpos[m.jnt_qposadr[jid]]

    lines = []
    for jid in range(m.njnt):
        name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, jid)
        adr = m.jnt_qposadr[jid]
        jtype = m.jnt_type[jid]
        if jtype == mujoco.mjtJoint.mjJNT_FREE:
            pos = d.qpos[adr:adr + 3]
            quat = d.qpos[adr + 3:adr + 7]
            lines.append(f'{name} (free)  pos={pos}  quat={quat}')
        elif jtype == mujoco.mjtJoint.mjJNT_HINGE:
            rad = d.qpos[adr]
            line = f'{name:16s} {rad:9.5f} rad  {np.degrees(rad):8.3f} deg'
            if name.endswith('_calf'):
                thigh_name = name[:-len('_calf')] + '_thigh'
                thigh_rad = qpos_rad_by_name.get(thigh_name)
                if thigh_rad is not None:
                    absolute_rad = rad - CALF_BELT_SIGN * thigh_rad
                    line += (f'   (ABSOLUTE, matches real motor encoder: '
                             f'{np.degrees(absolute_rad):8.3f} deg -- sanity check only, '
                             f'RAW above is what gets saved)')
            lines.append(line)
        else:
            lines.append(f'{name} type={jtype} qpos={d.qpos[adr]}')

    args.out.write_text('\n'.join(lines) + '\n')
    print(f'Wrote {args.out}')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
