import itertools
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Union

import click

DATASET_PATH = "data/scratch_data"
RUN_COMMANDS = os.environ.get("RUN_COMMANDS", "True") == "True"


def get_full_path(path_list):

    return [os.path.join(DATASET_PATH, p) if p is not None else None for p in path_list]


def assert_files_exists(files):
    for file in files:
        if not os.path.isfile(file):
            raise FileNotFoundError(f"File {file} not found")


def generate_alg_params(alg_fixed_params, alg_var_params):
    if isinstance(alg_var_params, list):
        params_output = [generate_alg_params(alg_fixed_params, p) for p in alg_var_params]
        alg_params, param_description = zip(*params_output)
        return list(itertools.chain.from_iterable(alg_params)), list(itertools.chain.from_iterable(param_description))

    alg_params = []
    param_description = []
    keys = alg_var_params.keys()
    alg_var_params = [dict(zip(keys, values)) for values in itertools.product(*alg_var_params.values())]
    for param in alg_var_params:
        alg_params.append({**alg_fixed_params, **param})
        param_description.append(
            ", ".join([f"{k}={v}" if not os.path.isfile(v) else f"{k}={os.path.split(v)[1]}" for k, v in param.items()])
        )

    return alg_params, param_description


def run_alg(
    datasets, dataset_names, run_prefix, params, params_description, run_analysis=True, unique_output_dir=False
):

    for L, dataset in datasets.items():
        for i, data in enumerate(dataset):
            dataset_name = dataset_names[i]
            for param, param_description in zip(params, params_description):
                run_name = f"{run_prefix}_{dataset_name}_L{L}_{param_description}"
                alg_param = {**param, "inputfile": data["dataset"], "name": f'"{run_name}"'}
                if data["mask"] is not None:
                    alg_param["mask"] = data["mask"]
                if data["poses"] is not None:
                    alg_param["poses"] = data["poses"]
                if "output-dir" in alg_param.keys():
                    alg_param["output-dir"] = os.path.join(
                        os.path.split(alg_param["inputfile"])[0], alg_param["output-dir"]
                    )
                if unique_output_dir:
                    alg_param["output-dir"] = os.path.join(
                        os.path.split(alg_param["inputfile"])[0], "output_" + param_description.replace(", ", "_")
                    )

                def keyvalue2cli(key, value):
                    if value is None:
                        return f"--{key}"
                    elif isinstance(value, bool):
                        return f"--{key}" if value else ""
                    else:
                        return f"--{key} {value}"

                command = "solvar comet-pipeline " + " ".join([keyvalue2cli(k, v) for k, v in alg_param.items()])

                if run_analysis:
                    command += " --run-analysis"
                    if data["gt_latent"] is not None:
                        command += f" --gt-latent {data['gt_latent']}"
                    if data["gt_dir"] is not None:
                        command += f" --gt-dir {data['gt_dir']}"
                    if data["gt_labels"] is not None:
                        command += f" --gt-labels {data['gt_labels']}"
                    if data.get("gt_pose", None) is not None:
                        command += f" --gt-pose {data['gt_pose']}"

                if RUN_COMMANDS:
                    os.system(command)
                else:
                    print(command)


def run_recovar_alg(datasets, dataset_names, run_prefix, run_analysis=True, zdim=10, output_dir=None):

    for L, dataset in datasets.items():
        for i, data in enumerate(dataset):
            dataset_name = dataset_names[i]
            run_name = f"{run_prefix}_{dataset_name}_L{L}"
            alg_param = {"mrc": data["dataset"], "name": f'"{run_name}"', "alg": "recovar", "zdim": zdim}
            if data["mask"] is not None:
                alg_param["mask"] = data["mask"]
            if data["poses"] is not None:
                alg_param["poses"] = data["poses"]
            if output_dir is not None:
                alg_param["output-dir"] = os.path.join(os.path.split(alg_param["mrc"])[0], output_dir)
            command = "python scripts/comet_cryodrgn.py " + " ".join(
                [f'--{k} {v if v is not None else ""}' for k, v in alg_param.items()]
            )

            if run_analysis:
                command += " --run-analysis"
                if data["gt_latent"] is not None:
                    command += f" --gt-latent {data['gt_latent']}"
                if data["gt_dir"] is not None:
                    command += f" --gt-dir {data['gt_dir']}"
                if data["gt_labels"] is not None:
                    command += f" --gt-labels {data['gt_labels']}"
                if data.get("gt_pose", None) is not None:
                    command += f" --gt-pose {data['gt_pose']}"
            if RUN_COMMANDS:
                os.system(command)
            else:
                print(command)


def filter_datasets(datasets, dataset_names):
    return {k: [d for d in v if any(name in d["dataset"] for name in dataset_names)] for k, v in datasets.items()}


dataset_names = [
    "igg_1d",
    "igg_1d_noisiest",
    "igg_rl",
    "Ribosembly",
    "Spike-MD",
    "Tomotwin-100",
    "Empiar10076",
    "Empiar10180",
]

datasets_L64 = [
    "igg_1d/images/snr0.01/downsample_L64/snr0.01.star",
    "igg_1d/images/snr0.001/downsample_L64/snr0.001.star",
    "igg_rl/images/snr0.01/downsample_L64/snr0.01.star",
    "Ribosembly/images/downsample_L64/snr0.01.star",
    "Spike-MD/images/snr0.1/downsample_L64/particles.star",
    "Tomotwin-100/images/snr0.01/downsample_L64/snr0.01.star",
    "empiar10076/downsample_L64/L17Combine_weight_local_preprocessed_L64.star",
    "empiar10180/downsample_L64/particles.star",
]

dataset_masks = [
    "igg_1d/init_mask/mask.mrc",
    "igg_1d/init_mask/mask.mrc",
    "igg_rl/init_mask/mask.mrc",
    "Ribosembly/init_mask/mask.mrc",
    "Spike-MD/init_mask/mask_128.mrc",
    "Tomotwin-100/init_mask/mask.mrc",
    "empiar10076/mask_full.mrc",
    "empiar10180/Mask-and-Ref/global_mask.mrc",
]

gt_dir = [
    "igg_1d/vols/128_org",
    "igg_1d/vols/128_org",
    "igg_rl/vols/128_org",
    "Ribosembly/vols/128_org",
    "Spike-MD/all_vols",
    "Tomotwin-100/vols/128_org",
    "data/scratch_data/empiar10076/analysis/minor_classes",
    None,
]

gt_latent = [
    "igg_1d/igg_1d_gt_latents.pkl",
    "igg_1d/igg_1d_gt_latents.pkl",
    "igg_rl/igg_rl_gt_latents.pkl",
    None,  # TODO: is there GT latent for ribosembly?
    "Spike-MD/gt_latents.pkl",
    None,  # TODO: is tehre GT latent for tomotwin
    None,
    None,
]

gt_labels = [
    "igg_1d/gt_latents.pkl",
    "igg_1d/gt_latents.pkl",
    "igg_rl/labels.pkl",
    "Ribosembly/gt_latents.pkl",
    "Spike-MD/labels.pkl",
    "Tomotwin-100/gt_latents.pkl",
    "empiar10076/downsample_L128/filtered_labels.pkl",
    None,
]

gt_pose = [
    "igg_1d/images/snr0.01/downsample_L128/poses.pkl",
    "igg_1d/images/snr0.001/downsample_L128/poses.pkl",
    "igg_rl/images/snr0.01/downsample_L128/poses.pkl",
    "Ribosembly/images/downsample_L128/poses.pkl",
    "Spike-MD/images/snr0.1/downsample_L128/poses.pkl",
    "Tomotwin-100/images/snr0.01/downsample_L128/poses.pkl",
    "empiar10076/downsample_L128/poses.pkl",  # Not actually GT but treated as for reference
    "empiar10180/downsample_L128/poses.pkl",
]

datasets_L128 = [dataset.replace("L64", "L128") for dataset in datasets_L64]


@dataclass
class Experiment:
    alg_fixed_params: Dict[str, Any]
    alg_var_params: Union[List[Dict[str, Any]], Dict[str, List[Any]]]
    run_prefix: str
    datasets: List[str] = None


reg_scheme_experiment = Experiment(
    alg_fixed_params={
        "rank": 15,
        "lr": 1e-1,
        "reg": 1,
        "max-epochs": 20,
        "batch-size": 4096,
        "orthogonal-projection": False,
        "nufft-disc": "bilinear",
    },
    alg_var_params=[
        {
            "use-halfsets": [False],
            "num-reg-update-iters": [0, 2],
        },
        {
            "use-halfsets": [True],
            "num-reg-update-iters": [2],
        },
    ],
    run_prefix="test_reg_scheme",
)

cryobench_analysis = Experiment(
    alg_fixed_params={
        "rank": 10,
        "reg": 1,
        "max-epochs": 20,
        "use-halfsets True": True,
        "num-reg-update-iters": 1,
        "output-dir": "reg_refactor_results",
    },
    alg_var_params={
        "objective-func": ["ml", "ls"],
    },
    run_prefix="Cryobench_final",
)

cost_func_reg_experiment = Experiment(
    alg_fixed_params={
        "rank": 10,
        "reg": 1,
        "max-epochs": 10,
        "use-halfsets": False,
        "debug": None,
    },
    alg_var_params={
        "objective-func": ["ml", "ls"],
        "reg": [1, 10, 100, 0.1, 1e-2, 1e-3],
        "lr": [1e-1, 1e-2, 1e0, 1e1],
    },
    run_prefix="obj_func_reg_experiment",
    datasets=["igg_1d/images/snr0.01"],
)

empiar_experiment = Experiment(
    alg_fixed_params={
        "rank": 15,
        "reg": 1,
        "max-epochs": 15,
        "batch-size": 2048,
        "orthogonal-projection": False,
        "nufft-disc": "bilinear",
        "use-halfsets": False,
        "num-reg-update-iters": 2,
        "debug": None,
    },
    alg_var_params={"lr": [1e-2]},
    run_prefix="Empiar_final",
    datasets=["empiar10076"],
)


def run_alg_from_exp(exp: Experiment, **kwargs):
    alg_params_list, alg_param_description = generate_alg_params(exp.alg_fixed_params, exp.alg_var_params)
    run_alg(exp.datasets, dataset_names, exp.run_prefix, alg_params_list, alg_param_description, **kwargs)


@click.command()
@click.option("--skip-reconstruction", is_flag=True, help="Skip reconstruction step")
@click.option("--print-run", is_flag=True, help="Print run command instead of running")
@click.option("--run-recovar", is_flag=True)
def main(skip_reconstruction, print_run, run_recovar):
    global RUN_COMMANDS, gt_dir
    if print_run:
        RUN_COMMANDS = False

    if skip_reconstruction:
        gt_dir = [None for _ in gt_dir]

    dataset_vars = ["dataset", "mask", "gt_dir", "gt_latent", "gt_labels", "gt_pose"]
    dataset_values = [datasets_L64, dataset_masks, gt_dir, gt_latent, gt_labels, gt_pose]
    datasets = {}
    # datasets[64] = [dict(zip(dataset_vars,get_full_path(values))) for values in zip(*dataset_values)]
    dataset_values[0] = datasets_L128
    datasets[128] = [dict(zip(dataset_vars, get_full_path(values))) for values in zip(*dataset_values)]

    exp = cryobench_analysis
    alg_params_list, alg_param_description = generate_alg_params(exp.alg_fixed_params, exp.alg_var_params)
    datasets_to_run = datasets if exp.datasets is None else filter_datasets(datasets, exp.datasets)
    if not run_recovar:
        run_alg(datasets_to_run, dataset_names, exp.run_prefix, alg_params_list, alg_param_description)
    else:
        run_recovar_alg(datasets_to_run, dataset_names, f"RECOVAR_{exp.run_prefix}", zdim=10)


if __name__ == "__main__":
    main()
