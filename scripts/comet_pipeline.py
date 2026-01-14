import os
import pickle
import sys
from glob import glob

import click
import comet_ml
import numpy as np
import pandas as pd
import torch
from aspire.storage import StarFile
from sklearn.metrics import auc

from cov3d.workflow import covar_workflow, workflow_click_decorator

# This ensures comet is able to log git info even when running script outside of repo directory
os.environ["COMET_GIT_DIRECTORY"] = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def log_cryobench_analysis_output(exp, result_dir, gt_dir, gt_latent, gt_labels):
    cryobench_output_dir = os.path.join(result_dir, "cryobench_output")
    if gt_latent is not None:
        neighbor_sim_output = np.loadtxt(os.path.join(cryobench_output_dir, "neighbor_sim_output.txt"))
        for i in range(neighbor_sim_output.shape[0]):
            exp.log_metric(
                name="latent_matching_neighbors_ratio",
                value=neighbor_sim_output[i, 1] / neighbor_sim_output[i, 0],
                step=neighbor_sim_output[i, 0],
            )
            exp.log_metric(
                name="latent_matching_neighbors_std", value=neighbor_sim_output[i, 2], step=neighbor_sim_output[i, 0]
            )

        information_imbalance = np.loadtxt(os.path.join(cryobench_output_dir, "information_imbalance.txt"))
        exp.log_metric(name="information_imbalance_est_to_gt", value=information_imbalance[0], step=0)
        exp.log_metric(name="information_imbalance_gt_to_est", value=information_imbalance[1], step=0)

        exp.log_image(image_data=os.path.join(cryobench_output_dir, "neighbor_sim.png"), name="neighbor_sim")

    if gt_labels is not None:
        clustering_metrics = np.loadtxt(os.path.join(cryobench_output_dir, "clustering_metrics.txt"))
        exp.log_metric(name="AMI", value=clustering_metrics[0], step=0)
        exp.log_metric(name="ARI", value=clustering_metrics[1], step=0)

    if gt_dir is not None:
        # This is the same code from cryobench's plot_fsc - used to compute FSC AUC.
        # TODO: log computed AUC in plot_fsc itself?
        fsc_output_dir = cryobench_output_dir
        fsc_dirs = [
            d
            for d in os.listdir(fsc_output_dir)
            if d.startswith("fsc_") and os.path.isdir(os.path.join(fsc_output_dir, d))
        ]
        for fsc_lbl in fsc_dirs:
            subdir = os.path.join(fsc_output_dir, fsc_lbl)
            fsc_files = glob(os.path.join(subdir, "*.txt"))

            fsc_list = list()
            auc_lst = list()
            for i, fsc_file in enumerate(fsc_files):
                fsc = pd.read_csv(fsc_file, sep=" ")
                fsc_list.append(fsc.assign(vol=i))
                auc_lst.append(auc(fsc.pixres, fsc.fsc))
            auc_metrics = {
                f"{fsc_lbl}_mean_auc": np.nanmean(auc_lst),
                f"{fsc_lbl}_std_auc": np.nanstd(auc_lst),
                f"{fsc_lbl}_median_auc": np.nanmedian(auc_lst),
            }
            exp.log_metrics(auc_metrics)
            exp.log_image(image_data=os.path.join(fsc_output_dir, f"{fsc_lbl}.png"), name=fsc_lbl)
            exp.log_image(image_data=os.path.join(fsc_output_dir, f"{fsc_lbl}_means.png"), name=f"{fsc_lbl}_means")


def log_metrics_from_dict(exp, data, keys):
    for key in keys:
        if key in data:
            values = data[key]
            for i, v in enumerate(values):
                exp.log_metric(name=key, value=v, step=i)


@click.command()
@click.option("-n", "--name", type=str, help="name of wandb run")
@click.option("--disable-comet", is_flag=True, default=False, help="wether to disable logging of run to comet")
@click.option("--run-analysis", is_flag=True, help="wether to run analysis script after algorithm execution")
@click.option("--num-clusters", type=int, default=0, help="Number of Kmeans cluster to use in analysis")
@click.option("--gt-dir", type=str, help="Directory of ground truth volumes")
@click.option("--gt-latent", type=str, help="Path to pkl containing ground truth embedding")
@click.option("--gt-labels", type=str, help="Path to pkl containing ground truth labels")
@click.option("--skip-computation", is_flag=True, help="Whether to skip covariance estimation computation")
@click.option("--num-vols", type=int, help="Number of GT volumes to use for FSC computation")
@workflow_click_decorator
def run_pipeline(
    name,
    inputfile,
    rank,
    whiten,
    mask,
    run_analysis,
    num_clusters=0,
    gt_dir=None,
    gt_latent=None,
    gt_labels=None,
    num_vols=None,
    disable_comet=False,
    skip_computation=False,
    **training_kwargs,
):
    if not disable_comet:
        image_size = int(float(StarFile(inputfile)["optics"]["_rlnImageSize"][0]))
        run_config = {"image_size": image_size, "rank": rank, "inputfile": inputfile, "whiten": whiten}
        run_config.update(training_kwargs)
        run_config["cli_command"] = " ".join(sys.argv)
        exp = comet_ml.Experiment(project_name="3d_cov", parse_args=False)
        exp.set_name(name)
        exp.log_parameters(run_config)
    training_kwargs = {k: v for k, v in training_kwargs.items() if v is not None}

    output_dir = training_kwargs.get("output_dir", None)
    if output_dir is None:
        output_dir = os.path.join(os.path.split(inputfile)[0], "result_data")

    if not skip_computation:
        data_dict, training_data, training_kwargs = covar_workflow(
            inputfile, rank, whiten=whiten, mask=mask, **training_kwargs
        )
    else:
        with open(os.path.join(output_dir, "recorded_data.pkl"), "rb") as fid:
            data_dict = pickle.load(fid)

        training_data = torch.load(os.path.join(output_dir, "training_results.bin"))

    if run_analysis:
        # Only import analysis functions when necesseary
        from cov3d.analyze import analyze
        from external.cryobench_analyze import cryobench_analyze

        # TODO: handle disable_comet
        # Run analysis
        analysis_figures = analyze(
            os.path.join(output_dir, "recorded_data.pkl"),
            analyze_with_gt=True,
            skip_reconstruction=num_clusters == 0,
            gt_labels=gt_labels,
            num_clusters=num_clusters,
        )
        for fig_name, fig_path in analysis_figures.items():
            exp.log_image(image_data=fig_path, name=fig_name)

        if gt_dir is not None or gt_latent is not None or gt_labels is not None:
            # Run cryobench analysis (compares to GT latent embedding and volume states)
            cryobench_analyze(
                output_dir,
                gt_dir=gt_dir,
                gt_latent=gt_latent,
                gt_labels=gt_labels,
                num_vols=num_vols,
                mask=mask if os.path.isfile(mask) else None,
            )
            log_cryobench_analysis_output(exp, output_dir, gt_dir, gt_latent, gt_labels)

    if not disable_comet:
        exp.log_parameters(training_kwargs)
        exp.log_metrics({"eigenval_est": data_dict["eigenval_est"]})

        if "eigenvals_GT" in data_dict.keys():
            metrics = {
                "frobenius_norm_error": training_data["fro_err"][-1],
                "eigen_vector_cosine_sim": training_data["cosine_sim"][-1],
                "covar_fsc": training_data["covar_fsc_mean"][-1],
                "eigenvals_GT": data_dict["eigenvals_GT"],
            }
            exp.log_metrics(metrics)
            [exp.log_metric(name="fro_norm_err", value=v, step=i) for i, v in enumerate(training_data["fro_err"])]
            [exp.log_metric(name="covar_fsc", value=v, step=i) for i, v in enumerate(training_data["covar_fsc_mean"])]
            [exp.log_metric(name="log_epoch_ind", value=v, step=i) for i, v in enumerate(training_data["epoch_ind"])]

        keys_to_log = [
            "rot_angle_dist",
            "in_plane_rot_angle_dist",
            "offsets_mean_dist",
            "mean_vol_norm_err",
            "mean_vol_fsc",
            "contrast_corr",
            "contrast_mean_dist",
        ]
        log_metrics_from_dict(exp, training_data, keys_to_log)

        data_artifact = comet_ml.Artifact("produced_data", "data")
        data_artifact.add(os.path.join(output_dir, "recorded_data.pkl"))
        exp.log_artifact(data_artifact)
        training_artifact = comet_ml.Artifact("training_data", "data")
        training_artifact.add(os.path.join(output_dir, "training_results.bin"))
        exp.log_artifact(training_artifact)

        exp.end()


if __name__ == "__main__":
    run_pipeline()
