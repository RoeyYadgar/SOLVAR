import os
import subprocess

import click
import comet_ml
from comet_pipeline import log_cryobench_analysis_output

from external.cryobench_analyze import cryobench_analyze
from solvar.analyze import analyze


@click.command()
@click.option("-n", "--name", type=str, help="name of comet run")
@click.option("--alg", type=str, help="Which algorithm to use (recovar,cryodrgn)")
@click.option("-m", "--mrc", type=str, help="path to mrc file")
@click.option("-z", "--zdim", type=int, help="Latent space dimension")
@click.option("--num-epochs", type=int, help="Number of epochs to train CRYODRGN")
@click.option("--mask", default="sphere", type=str, help="mask type for recovar")
@click.option("--poses", type=str, default=None, help="path to pkl containing poses")
@click.option("--ctf", type=str, default=None, help="path to pkl containing ctf")
@click.option("-o", "--output-dir", type=str, default=None, help="path to output directory")
@click.option("--disable-comet", is_flag=True, default=False, help="wether to disable logging of run to comet")
@click.option("--run-analysis", is_flag=True, help="wether to run analysis script after algorithm execution")
@click.option("--gt-dir", type=str, help="Directory of ground truth volumes")
@click.option("--gt-latent", type=str, help="Path to pkl containing ground truth embedding")
@click.option("--gt-labels", type=str, help="Path to pkl containing ground truth labels")
@click.option("--skip-computation", is_flag=True, help="Whether to skip covariance estimation computation")
@click.option("--num-vols", type=int, help="Number of GT volumes to use for FSC computation")
def run_pipeline(
    name,
    alg,
    mrc,
    zdim,
    num_epochs,
    mask,
    poses=None,
    ctf=None,
    output_dir=None,
    run_analysis=False,
    gt_dir=None,
    gt_latent=None,
    gt_labels=None,
    num_vols=None,
    skip_computation=False,
    disable_comet=False,
):
    mrcdir = os.path.split(mrc)[0]
    starfile = mrc.replace(".mrcs", ".star")
    poses = os.path.join(mrcdir, "poses.pkl") if poses is None else poses
    ctf = os.path.join(mrcdir, "ctf.pkl") if ctf is None else ctf
    output_path = os.path.join(mrcdir, f"{alg}_results") if output_dir is None else output_dir
    os.makedirs(output_path, exist_ok=True)
    if alg == "cryodrgn":
        command = (
            f"cryodrgn train_vae {mrc} --poses {poses} --ctf {ctf} --zdim {zdim} "
            f"-n {num_epochs} -o {output_path} --multigpu"
        )
        analyze_command = f"cryodrgn analyze {output_path} {num_epochs-1}"
    if alg == "recovar":
        command = (
            f"recovar pipeline {starfile} --poses {poses} --ctf {ctf} --zdim {zdim} "
            f"-o {output_path} --mask {mask} --correct-contrast --low-memory-option"
        )
        analyze_command = f"recovar analyze --zdim {zdim} {output_path} --skip-centers --n-trajectories 0"
    if not disable_comet:
        run_config = {"inputfile": starfile, "zdim": zdim, "command": command, "analyze_command": analyze_command}
        exp = comet_ml.Experiment(project_name="3d_cov", parse_args=False)
        exp.set_name(name)
        exp.log_parameters(run_config)

    if not skip_computation:
        subprocess.run(
            f"bash -c 'source $(conda info --base)/etc/profile.d/conda.sh && conda activate {alg} && {command}'",
            shell=True,
        )
        subprocess.run(
            (
                f"bash -c 'source $(conda info --base)/etc/profile.d/conda.sh && "
                f"conda activate {alg} && {analyze_command}'"
            ),
            shell=True,
        )

    analyze_dir = (
        os.path.join(output_path, f"analyze.{num_epochs-1}")
        if alg == "cryodrgn"
        else os.path.join(output_path, f"output/analysis_{zdim}/umap")
    )
    if gt_labels is not None:
        umap_file = (
            os.path.join(analyze_dir, "umap.pkl") if alg == "cryodrgn" else os.path.join(analyze_dir, "embedding.pkl")
        )
        umap_image = os.path.join(analyze_dir, "umap_labeled.jpg")
        os.system(f"python scripts/umap_figure.py -u {umap_file} -l {gt_labels} -o {umap_image}")
    else:
        umap_image = (
            os.path.join(analyze_dir, "umap.png") if alg == "cryodrgn" else os.path.join(analyze_dir, "sns.png")
        )

    if run_analysis and alg == "recovar":
        os.system(f'python {os.path.join(os.path.dirname(__file__), "convert_recovar_model.py")} {output_path}')
        analysis_figures = analyze(
            os.path.join(output_path, "recorded_data.pkl"),
            analyze_with_gt=True,
            skip_reconstruction=True,
            gt_labels=gt_labels,
            num_clusters=0,
        )

        for fig_name, fig_path in analysis_figures.items():
            exp.log_image(image_data=fig_path, name=fig_name)

        cryobench_analyze(
            output_path,
            gt_dir=gt_dir,
            gt_latent=gt_latent,
            gt_labels=gt_labels,
            num_vols=num_vols,
            mask=mask if os.path.isfile(mask) else None,
        )
        log_cryobench_analysis_output(exp, output_path, gt_dir, gt_latent, gt_labels)

    if not disable_comet:
        exp.log_image(image_data=umap_image, name="umap_coords_est")
        exp.end()


if __name__ == "__main__":
    run_pipeline()
