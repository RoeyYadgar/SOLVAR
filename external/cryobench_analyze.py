import os
import subprocess

import click


def run_with_conda_env(command):
    # TODO : print STDOUT live
    process = subprocess.Popen(
        command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1, shell=True
    )

    for line in process.stdout:
        print(line, end="")

    process.wait()


def analysis_click_decorator(func):
    @click.option("--gt-dir", type=str, help="Directory of ground truth volumes")
    @click.option("--gt-latent", type=str, help="Path to pkl containing ground truth embedding")
    @click.option("--gt-labels", type=str, help="Path to pkl containing ground truth labels")
    @click.option("--mask", type=str, help="Mask mrc file used for FSC computation")
    @click.option("--num-vols", type=int, help="Number of GT volumes to use for FSC computation")
    def wrapper(*args, **kwargs):
        kwargs = {k: v for k, v in kwargs.items() if v is not None}
        return func(*args, **kwargs)

    return wrapper


@click.command()
@click.option("-i", "--result_dir", type=str, help="Result dir of algorithm's output")
@analysis_click_decorator
def cryobench_analyze_cli(result_dir, **kwargs):
    cryobench_analyze(result_dir, **kwargs)


def cryobench_analyze(result_dir, gt_dir=None, gt_latent=None, gt_labels=None, mask=None, num_vols=None):

    output_dir = os.path.join(result_dir, "cryobench_output")
    os.makedirs(output_dir, exist_ok=True)

    if gt_latent is not None:
        script_path = os.path.join(os.path.dirname(__file__), "compute_latent_embedding_metrics.py")
        neighb_sim = f"python {script_path} {result_dir} -o {output_dir} --gt-latent {gt_latent}"
        run_with_conda_env(neighb_sim)

    if gt_labels is not None:
        script_path = os.path.join(os.path.dirname(__file__), "compute_clustering_metrics.py")
        clustering_metrics = f"python {script_path} {result_dir} -o {output_dir} --gt-labels {gt_labels}"
        run_with_conda_env(clustering_metrics)

    if gt_dir is not None:
        if num_vols is None:
            num_vols = len(os.listdir(gt_dir))
            print(f"num-vols was not provided. Using all {num_vols} GT volumes from {gt_dir}")

        script_path = os.path.join(os.path.dirname(__file__), "compute_fsc.py")
        fsc_no_mask = f"python {script_path} {result_dir} -o {output_dir} --gt-dir {gt_dir} --num-vols {num_vols}"
        run_with_conda_env(fsc_no_mask + " --overwrite")  # Should only overwrite the first time

        if mask is not None:
            fsc_mask = fsc_no_mask + f" --mask {mask}"
            run_with_conda_env(fsc_mask)


if __name__ == "__main__":
    cryobench_analyze_cli()
