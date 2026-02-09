# SOLVAR
SOLVAR is a tool for heterogeneity analysis in cryo-EM. SOLVAR estimates the principal components of the conformations, and allows for the refinement of particle poses to account for the heterogeneity within the dataset.


## Installation

SOLVAR uses [cryoDRGN](https://github.com/ml-struct-bio/cryodrgn) for reading particle data and has the same input interface as cryoDRGN, and uses [RECOVAR](https://github.com/ma-gilles/recovar) for heterogneous reconstruction (from the latent embedding produced by SOLVAR).
Follow the instructions bellow for installing SOLVAR.
```
conda create --name solvar python=3.11
conda activate solvar

pip install --upgrade "jax[cuda12]>=0.5.0,<0.6.0" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
pip install -e .

#Install recovar with no dependecies - due to some dependecies confclits
pip install recovar --no-deps
```
**Note:** For reconstruction alternatives ([see below](#running-the-analysis-pipeline)), SOLVAR relies on [RELION](https://github.com/3dem/relion). To use these features, you must have RELION installed and ensure that `relion_reconstruct` is available in your system path.


## Running SOLVAR

### Preprocessing
SOLVAR uses cryoDRGN's input interface, and the preprocessing steps are identical to cryoDRGN (see cryoDRGN's documentation [downsampling](https://github.com/ml-struct-bio/cryodrgn?tab=readme-ov-file#1-preprocess-image-stack), [parsing poses](https://github.com/ml-struct-bio/cryodrgn?tab=readme-ov-file#2-parse-image-poses-from-a-consensus-homogeneous-reconstruction), [parsing CTF parameters](https://github.com/ml-struct-bio/cryodrgn?tab=readme-ov-file#2-parse-image-poses-from-a-consensus-homogeneous-reconstruction)).

### Running the main pipeline


```
solvar workflow -i <inputfile> -r <rank> -o <output_dir> -p <poses> -c <ctf>
```
<details><summary><code>$ solvar workflow -h</code></summary>
    Usage: solvar workflow [OPTIONS]

    Options:
    -i, --inputfile TEXT            path to star/txt/mrcs file.
    -r, --rank INTEGER              rank of covariance to be estimated.
    -o, --output-dir TEXT           path to output directory. when not provided
                                    a `result_data` directory will be used with
                                    the same path as the provided starfile
    -p, --poses TEXT                Path to pkl file containing particle pose
                                    information in cryoDRGN format
    -c, --ctf TEXT                  Path to pkl file containing CTF information
                                    in cryoDRGN format
    --lazy                          Whether to use lazy dataset. If set, the
                                    dataset will not be loaded into (CPU) memory
                                    and will be processed when accessed. This is
                                    useful for large datasets.
    --ind TEXT                      Path to pkl file with particle indices to be
                                    used (optional)
    -w, --whiten BOOLEAN            whether to whiten the images before
                                    processing
    --mask TEXT                     Type of mask to be used on the dataset. Can
                                    be either 'fuzzy' or path to a volume file/
                                    Defaults to 'fuzzy'
    --optimize-pose                 Whether to optimize over image pose
    --optimize-contrast             Whether to correct for contrast in particle
                                    images (can only be used with --optimize-
                                    pose flag)
    --class-vols TEXT               Path to GT volumes directory. Used if
                                    provided to log eigen vectors error metrics
                                    while training. Additionally, GT embedding
                                    is computed and logged
    --gt-pose TEXT                  Path to GT pkl pose file (cryoDRGN format).
                                    Used if provided to log pose error metrics
                                    while training
    --debug                         debugging mode
    --gt-path TEXT                  Path to pkl file containing GT dataclass
    --batch-size INTEGER            training batch size
    --max-epochs INTEGER            number of epochs to train
    --lr FLOAT                      training learning rate
    --reg FLOAT                     regularization scaling
    --gamma-lr FLOAT                learning rate decay rate
    --orthogonal-projection BOOLEAN
                                    force orthogonality of eigen vectors while
                                    training (default True)
    --nufft-disc [bilinear|nearest|nufft]
                                    Discretisation of NUFFT computation
    --fourier-upsampling INTEGER    Upsaming factor in fourier domain for
                                    Discretisation of NUFFT. Only used when
                                    --nufft-disc is provided (default 2)
    --num-reg-update-iters INTEGER  Number of iterations to update
                                    regularization
    --use-halfsets BOOLEAN          Whether to split data into halfsets for
                                    regularization update
    --objective-func [ml|ls]        Which objective function to opimize. Either
                                    ml (maximum liklihood) or ls (least squares)
    --log-level [DEBUG|INFO|WARNING|ERROR|CRITICAL]
    -h, --help                      Show this message and exit.
</details>

To enable pose refinement, we can use the `--optimize-pose` flag (and optionally `--optimize-contrast`). It is recommened however to use these additional arguments together:
```
solvar workflow -i <inputfile> -r <rank> -o <output_dir> -p <poses> -c <ctf> --optimize-pose --optimize-contrast --num-reg-update-iters 0
```
If SOLVAR was run with `--optimize-pose`, the refined poses will be saved in the output directory as `refined_poses.pkl`.
And similarly for the refined contrast: if `--optimize-contrast` was used, the refined contrast will be saved in the output directory as `contrast.pkl`.

Providing a mask (with `--mask /path/to/mask.mrc`) can have a great impact on SNR (and the results in the pipeline). By default, if a mask is not provided, SOLVAR uses a spherical mask.

By default SOLVAR uses trilinear interpolation of the projection operator with a fourier upsampling factor of 2. We can modify the interpolation and upsampling factor to trade-off speed and accuracy (For example use `--nufft-disc nearest` and/or `--fourier-upsampling 1` if we want a quicker run with the potential loss of accuracy).



### Running the analysis pipeline

After running SOLVAR, we can analyze the latent space and reconstruct volumes using `solvar analyze`. This command generates figures of the latent embeddings, and displays the estimated principal components.

```
solvar analyze -i <run_directory>
```

<details><summary><code>$ solvar analyze -h</code></summary>
    Usage: solvar analyze [OPTIONS]

    Command line interface for the analyze function.

    Args:     result_data: Path to pickle file containing algorithm results
    **kwargs: Additional keyword arguments passed to analyze function

    Options:
    -i, --result-data TEXT          Output directory of SOLVAR run or path to
                                    pkl output of the run
    -o, --output-dir TEXT           directory to store analysis output (same
                                    directory as result_data by default)
    --analyze-with-gt               whether to also perform analysis with
                                    embedding from gt eigenvolumes (if
                                    availalbe)
    --num-clusters INTEGER          number of k-means clusters used to
                                    reconstruct from embedding
    --latent-coords TEXT            path to pkl containing latent coords to be
                                    used as cluster centers instead of k-means
    --reconstruct-method [recovar|relion|reprojection|relion_disjoint]
                                    which volume reconstruction method to use
    --skip-reconstruction           whether to skip reconstruction of k-means
                                    cluster centers
    --skip-coor-analysis            whether to skip coordinate analysis (kmeans
                                    clustering & umap)
    --num-trajectories INTEGER      Number of trajectories to compute (default
                                    0)
    --gt-labels TEXT                path to pkl file containing gt labels. if
                                    provided used for coloring embedding figures
    --override-particles TEXT...    Override particles for volume reconstruction
                                    (Used for reconstruction at original
                                    resolution, before downsampling). Specify
                                    particles_path ctf_path poses_path
    -h, --help                      Show this message and exit.
</details>

This command will write latent embedding figures into the analysis directory.<br>
SOLVAR will also reconstruct volumes from cluster centers after running k-means on the latent embedding. Alternatively we can use this command to generate volumes from specified latent positions by using the `--latent-coords` argument and providing a path to pickle file containing a numpy array of the latent coordinates.<br>
Additionally, this command can also generate trajectories (with the reconstructed volumes along the trajectory) by using `--num-trajectories` flag.<br>
The `--override-particles` argument is very useful for two cases:
* If we downsampled the particle stack and then ran SOLVAR, we can reconstruct from the original particle size by using `--override-particles og_size_stack.star same_ctf.pkl same_poses.pkl`.
* If we want to reconstruct from volumes with a different poses then what SOLVAR was given, for example if `--optimize-pose` was used we can reconstruct volumes with the new refined poses by using `--override-particles same_stack.star same_ctf.pkl /path/to/refined_poses.pkl`.

By default SOLVAR uses [RECOVAR's reconstruction algorithm](https://github.com/ma-gilles/recovar) and its implementation. We can use other reconstruction methods by specifying `--reconstruct-methods`.

### Interactive analysis viewer

We can view the analysis results in an interactive way by using the `solvar analysis-viewer` command. This opens an interactive GUI, which lets us:
* View the latent space in different UMAP/principal components axis.
* Select interactively latent coordinates and export them (to be used for reconstruction again with `solvar analyze --latent-coords coords.pkl`)
* Select interactively a subset of particles and export them to star/pkl file (to be used in subsequent runs in your pipeline).

```
solvar analysis-viewer analysis_output_dir/data.pkl
```

<details><summary><code>$ solvar analysis-viewer -h</code></summary>
    Usage: solvar analysis-viewer [OPTIONS] [PATH]

    Launch the interactive analysis viewer GUI.

    Opens an interactive GUI for visualizing and analyzing coordinate data with
    UMAP and PCA projections. Users can select cluster coordinates and save/load
    them.

    PATH: Path to analyze_coordinates pkl file (optional, will prompt if not
    provided)

    Options:
    -n, --max-points NUMBER_WITH_SUFFIX
                                    Max points to display (e.g., 10k, 2M, 500).
                                    Used for performance optimization with large
                                    datasets.
    -f, --format [solvar|cryodrgn]  Input data format (default: solvar)
    -h, --help                      Show this message and exit.

</details>
