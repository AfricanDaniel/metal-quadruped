#!/usr/bin/env python3
"""Interactively pose the dog in the MuJoCo viewer, then save the final
joint angles to a .txt file for reference (e.g. comparing a hand-set pose
against real hardware calibration).

Usage:
    python3 save_pose.py [--mjcf dog.mjcf.xml] [--out pose.txt]

Drag joints / let the model settle under gravity in the viewer window,
then close it -- qpos at the moment you close is what gets saved.
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
            lines.append(f'{name:16s} {rad:9.5f} rad  {np.degrees(rad):8.3f} deg')
        else:
            lines.append(f'{name} type={jtype} qpos={d.qpos[adr]}')

    args.out.write_text('\n'.join(lines) + '\n')
    print(f'Wrote {args.out}')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
