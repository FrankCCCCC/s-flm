"""Run-setup helpers shared by the experiment sweeps (Agent.md, "Training & Sampling Run Setup").

On the borrowed accounts the checkpoints live on node-local scratch, which is only mounted on compute nodes, so the
link is made from inside the job, before training starts:

    python -c "import experiment; experiment.link_scratch_checkpoints('<output_dir>', '<project>', '<run>')"
"""
import os

SCRATCH_ROOTS = {
    'ch2263': '/scratch/ch2263/syc_workspace/sfm_output',
    'shengyenc': '/scratch/shengyenc/syc_workspace/sfm_output',
}


def link_scratch_checkpoints(output_dir, project, run, user=None):
    """Create `<scratch>/<project>/<run>/checkpoints` and point `<output_dir>/checkpoints` at it.

    Idempotent: an existing link to the same target is kept (so a resubmitted run resumes from
    `checkpoints/last.ckpt` there); an existing real directory is left alone and reported."""
    user = user or os.environ.get('USER')
    target = os.path.join(SCRATCH_ROOTS[user], project, run, 'checkpoints')
    os.makedirs(target, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)
    link = os.path.join(output_dir, 'checkpoints')
    if os.path.islink(link):
        if os.path.realpath(link) != os.path.realpath(target):
            os.remove(link)
            os.symlink(target, link)
    elif os.path.exists(link):
        print(f'[experiment] {link} is a real directory; leaving it (checkpoints stay there)')
        return link
    else:
        os.symlink(target, link)
    print(f'[experiment] {link} -> {target}')
    return link
