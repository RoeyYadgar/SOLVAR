import os
import pickle

import click
import numpy as np
import torch
from aspire.volume import Volume

from solvar.covar_sgd import cosineSimilarity, frobeniusNorm, frobeniusNormDiff


@click.command()
@click.option("--recovar-results", type=click.Path(exists=True))
@click.option("--solvar-results", type=click.Path(exists=True))
def compare_recovar_results(recovar_results, solvar_results):
    solvar_results = torch.load(os.path.join(solvar_results, "training_results.bin"))
    solvar_eigenvectors = solvar_results["vectors"]
    gd_eigenvectors = solvar_results["vectorsGT"]
    device = solvar_eigenvectors.device
    rank = solvar_eigenvectors.shape[0]
    L = solvar_eigenvectors.shape[1]

    with open(os.path.join(recovar_results, "recorded_data.pkl"), "rb") as f:
        recovar_data = pickle.load(f)
        print(recovar_results)
        recovar_eigenvalues = recovar_data["eigenval_est"]

    recovar_eigenvectors = Volume(np.zeros((rank, L, L, L)))
    for i in range(rank):
        volume_path = f"output/volumes/eigen_pos{i:04d}.mrc"
        recovar_eigenvectors[i] = Volume.load(os.path.join(recovar_results, volume_path)) * np.sqrt(
            recovar_eigenvalues[i]
        )

    recovar_eigenvectors = torch.tensor(recovar_eigenvectors.asnumpy(), device=device, dtype=solvar_eigenvectors.dtype)
    recovar_eigenvectors = recovar_eigenvectors.reshape(rank, -1)
    solvar_eigenvectors = solvar_eigenvectors.reshape(rank, -1)
    frobenius_norm_err_recovar = frobeniusNormDiff(recovar_eigenvectors * L, gd_eigenvectors) / frobeniusNorm(
        gd_eigenvectors
    )
    frobenius_norm_err_solvar = frobeniusNormDiff(solvar_eigenvectors, gd_eigenvectors) / frobeniusNorm(gd_eigenvectors)

    cosine_sim_recovar = cosineSimilarity(recovar_eigenvectors, gd_eigenvectors)
    cosine_sim_solvar = cosineSimilarity(solvar_eigenvectors, gd_eigenvectors)

    cosine_sim_recovar = np.mean(np.sqrt(np.sum(cosine_sim_recovar**2, axis=0)))
    cosine_sim_solvar = np.mean(np.sqrt(np.sum(cosine_sim_solvar**2, axis=0)))

    print(f"Frobenius norm error for Recovar: {frobenius_norm_err_recovar}")
    print(f"Frobenius norm error for solvar: {frobenius_norm_err_solvar}")
    print(f"Cosine similarity for Recovar: {cosine_sim_recovar}")
    print(f"Cosine similarity for solvar: {cosine_sim_solvar}")


if __name__ == "__main__":
    compare_recovar_results()
