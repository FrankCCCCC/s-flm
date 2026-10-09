"""Checkpoint storage of sweep runs (Agent.md, "Training & Sampling Run Setup").

On ch2263@unicorn and shengyenc@ARC a run's checkpoints live on the compute node's scratch disk,
/scratch/<user>/syc_workspace/sfm_output/<project>/<run>/checkpoints, and ${OUTPUT_DIR}/checkpoints
is a symlink to it; for other users they stay in ${OUTPUT_DIR}/checkpoints. The scratch disk exists
only on compute nodes, so a job calls this before training:

  python experiment.py <output_dir> <project> <run>
"""
import getpass
import os
import sys

SCRATCH = {'ch2263': '/scratch/ch2263/syc_workspace/sfm_output',
           'shengyenc': '/scratch/shengyenc/syc_workspace/sfm_output'}


def link_checkpoints(output_dir, project, run):
  """Make `output_dir`/checkpoints a symlink to the user's scratch checkpoint directory."""
  root = SCRATCH.get(getpass.getuser())
  if root is None:
    return
  target = os.path.join(root, project, run, 'checkpoints')
  link = os.path.join(output_dir, 'checkpoints')
  os.makedirs(target, exist_ok=True)
  os.makedirs(output_dir, exist_ok=True)
  if not os.path.islink(link):
    os.symlink(target, link)   # refuses to shadow a real checkpoints/ directory


if __name__ == '__main__':
  link_checkpoints(*sys.argv[1:4])
