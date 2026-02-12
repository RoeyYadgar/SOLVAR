import argparse
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import starfile


def relion_refine(star, ref, output_dir, custom_args=None, seed=0):
    """Calls relion_refine and extracts the final star and mrc file after refinement."""
    star_obj = starfile.read(star)
    star_data = star_obj["particles"]
    pose_keys = [
        "rlnAngleRot",
        "rlnAngleTilt",
        "rlnAnglePsi",
        "rlnOriginXAngst",
        "rlnOriginYAngst",
        "rlnOriginX",
        "rlnOriginY",
    ]

    contains_pose = any([c in star_data.columns for c in pose_keys])
    _, orig_star_name = os.path.split(star)

    if contains_pose:

        # Create new star without pose info
        star_obj["particles"] = star_data.drop(columns=pose_keys, errors="ignore")
        star = f"{star}.no_pose.tmp.star"
        starfile.write(star_obj, star)
        print(f"Starfile contains particle poses info, writing a new temp starfile to {star}")

    ref = os.path.abspath(ref)
    star_dir, star_name = os.path.split(star)

    assert Path(star_dir) != Path(output_dir), "Output directory cannot be the same as the input directory"

    os.makedirs(output_dir, exist_ok=True)

    with tempfile.TemporaryDirectory() as refinement_dir:

        refinement_dir = os.path.join(refinement_dir, "refine")

        refine_cmd = (
            f"mpirun -np 3 relion_refine_mpi --o {refinement_dir} --auto_refine --split_random_halves"
            f" --i {star_name} --ref {ref} --firstiter_cc --ini_high 15 --dont_combine_weights_via_disc"
            " --pool 3 --pad 2  --skip_gridding  --ctf --flatten_solvent --zero_mask"
            " --oversampling 1 --healpix_order 2 --auto_local_healpix_order 4 --offset_range 15 --offset_step 2"
            f' --low_resol_join_halves 40 --norm --scale  --j 1 --gpu "" --random_seed {seed}'
        )

        if custom_args is not None:
            refine_cmd += " " + " ".join(custom_args)

        refine_cmd = f"cd {star_dir} && {refine_cmd}"
        subprocess.run(refine_cmd, shell=True, check=True)

        output_star = f"{refinement_dir}_data.star"
        output_mrc = f"{refinement_dir}_class001.mrc"

        shutil.move(output_star, os.path.join(output_dir, "relion_refine_particles.star"))
        shutil.move(output_mrc, os.path.join(output_dir, "relion_refine_mean.mrc"))

    if contains_pose:
        print(f"Removing temp starifle {star}")
        try:
            os.remove(star)
        except FileNotFoundError:
            print(f"Temp starfile {star} not found")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run relion_refine on a STAR file and reference MRC.")
    parser.add_argument("star", help="Input STAR file")
    parser.add_argument("ref", help="Reference MRC file")
    parser.add_argument("--output-dir", required=True, help="Directory to save output files")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    args, custom_args = parser.parse_known_args()

    relion_refine(args.star, args.ref, args.output_dir, custom_args, args.seed)
