import os
import sys
import time
from typing import List

from matplotlib import offsetbox
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from rdkit import Chem
from rdkit.Chem import Draw
from sklearn.manifold import TSNE
from tap import Tap
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

from chemprop.data import get_smiles
from chemprop.features import get_features_generator
from chemprop.utils import makedirs


class Args(Tap):
    smiles_paths_1: List[str]  # Path to .csv files containing smiles strings (with header)
    smiles_paths_2: List[str]  # Path to .csv files containing smiles strings (with header)
    # smiles_paths_3: List[str]  # Path to .csv files containing smiles strings (with header)
    smiles_column: str = None  # Name of the column containing SMILES strings for the first data. By default, uses the first column.
    colors_1: List[str] = ['red', 'green', 'orange', 'purple', 'blue']  # Colors of the points associated with each dataset
    colors_2: List[str] = ['green', 'orange', 'purple', 'blue', 'red']  # Colors of the points associated with each dataset
    # colors_3: List[str] = ['orange', 'purple', 'blue', 'red', 'green']  # Colors of the points associated with each dataset
    sizes_1: List[float] = [1, 1, 1, 1, 1]  # Sizes of the points associated with each molecule
    sizes_2: List[float] = [1, 1, 1, 1, 1]  # Sizes of the points associated with each molecule
    # sizes_3: List[float] = [1, 1, 1, 1, 1]  # Sizes of the points associated with each molecule
    scale: int = 1  # Scale of figure
    plot_molecules: bool = False  # Whether to plot images of molecules instead of points
    max_per_dataset: int = 10000  # Maximum number of molecules per dataset; larger datasets will be subsampled to this size
    save_path: str  # Path to a .png file where the t-SNE plot will be saved
    cluster: bool = False  # Whether to create new clusters from all smiles, ignoring original csv groupings


def compare_datasets_tsne(args: Args):
    if len(args.smiles_paths_1) > len(args.colors_1) or len(args.smiles_paths_1) > len(args.sizes_1):
        raise ValueError('Must have at least as many colors and sizes as datasets')

    if len(args.smiles_paths_2) > len(args.colors_2) or len(args.smiles_paths_2) > len(args.sizes_2):
        raise ValueError('Must have at least as many colors and sizes as datasets')    
    """
    if len(args.smiles_paths_3) > len(args.colors_3) or len(args.smiles_paths_3) > len(args.sizes_3):
        raise ValueError('Must have at least as many colors and sizes as datasets')    
    """    

    # Random seed for random subsampling
    np.random.seed(0)

    # Load the smiles datasets
    print('Loading data')
    smiles_1, slices_1, labels_1 = [], [], []
    number_slice_1 = 0
    for smiles_path_1 in args.smiles_paths_1:
        # Get label
        label_1 = os.path.basename(smiles_path_1).replace('.csv', '')

        # Get SMILES
        new_smiles_1 = get_smiles(path=smiles_path_1, smiles_columns=args.smiles_column, flatten=True)
        print(f'{label_1}: {len(new_smiles_1):,}')

        # Subsample if dataset is too large
        if len(new_smiles_1) > args.max_per_dataset:
            print(f'Subsampling to {args.max_per_dataset:,} molecules')
            new_smiles_1 = np.random.choice(new_smiles_1, size=args.max_per_dataset, replace=False).tolist()

        slices_1.append(slice(len(smiles_1), len(smiles_1) + len(new_smiles_1)))
        labels_1.append(label_1)
        smiles_1 += new_smiles_1
        number_slice_1 = len(new_smiles_1)

    smiles_2, slices_2, labels_2 = [], [], []   

    for smiles_path_2 in args.smiles_paths_2:
        # Get label
        label_2 = os.path.basename(smiles_path_2).replace('.csv', '')

        # Get SMILES
        new_smiles_2 = get_smiles(path=smiles_path_2, smiles_columns=args.smiles_column, flatten=True)
        print(f'{label_2}: {len(new_smiles_2):,}')

        # Subsample if dataset is too large
        if len(new_smiles_2) > args.max_per_dataset:
            print(f'Subsampling to {args.max_per_dataset:,} molecules')
            new_smiles_2 = np.random.choice(new_smiles_2, size=args.max_per_dataset, replace=False).tolist()

        slices_2.append(slice(len(smiles_2), len(smiles_2) + len(new_smiles_2)))
        labels_2.append(label_2)
        smiles_2 += new_smiles_2
        # number_slice_2 = len(new_smiles_2) for third file

    """
    ** For third file
    smiles_3, slices_3, labels_3 = [], [], []
    number_slice_3 = 0
    for smiles_path_3 in args.smiles_paths_3:
        # Get label
        label_3 = os.path.basename(smiles_path_3).replace('.csv', '')

        # Get SMILES
        new_smiles_3 = get_smiles(path=smiles_path_3, smiles_columns=args.smiles_column, flatten=True)
        print(f'{label_3}: {len(new_smiles_3):,}')

        # Subsample if dataset is too large
        if len(new_smiles_3) > args.max_per_dataset:
            print(f'Subsampling to {args.max_per_dataset:,} molecules')
            new_smiles_3 = np.random.choice(new_smiles_3, size=args.max_per_dataset, replace=False).tolist()

        slices_3.append(slice(len(smiles_3), len(smiles_3) + len(new_smiles_3)))
        labels_3.append(label_3)
        smiles_3 += new_smiles_3
    """

    smiles = []
    smiles = smiles_1 + smiles_2 # + smiles_3

    # Compute Morgan fingerprints
    print('Computing Morgan fingerprints')
    morgan_generator = get_features_generator('morgan')
    morgans = np.array([morgan_generator(smile) for smile in tqdm(smiles, total=len(smiles))])

    print('Running t-SNE')
    start = time.time()
    tsne = TSNE(n_components=2, init='pca', random_state=0, metric='jaccard')
    X = tsne.fit_transform(morgans)
    print(f'time = {time.time() - start:.2f} seconds')

    if args.cluster:
        import hdbscan  # pip install hdbscan
        print('Running HDBSCAN')
        start = time.time()
        clusterer = hdbscan.HDBSCAN(min_cluster_size=5, gen_min_span_tree=True)
        colors = clusterer.fit_predict(X)
        print(f'time = {time.time() - start:.2f} seconds') 

    print('Plotting t-SNE')

    x_min, x_max = np.min(X, axis=0), np.max(X, axis=0)
    X = (X - x_min) / (x_max - x_min)

    X_1 = X[:number_slice_1]
    print("1st list")

    # Erase if you want to use third file
    print(X_1)
    X_2 = X[number_slice_1:]
    print("2nd list")
    print(X_2)


    """
    ** For third file 
    X_2 = X[number_slice_1:number_slice_2]
    print("2nd list")
    print(X_2)
    X_3 = X[number_slice_2:]
    print("3rd list")
    print(X_3)
    """

    makedirs(args.save_path, isfile=True)

    plt.clf()
    fontsize = 50 * args.scale
    fig = plt.figure(figsize=(64 * args.scale, 48 * args.scale))
    plt.title('t-SNE using Morgan fingerprint with Jaccard similarity', fontsize=2 * fontsize)
    ax = fig.gca()
    handles = []
    legend_kwargs = dict(loc='upper right', fontsize=fontsize)

    if args.cluster:
        plt.scatter(X_1[:, 0], X_1[:, 1], s=150 * np.mean(args.sizes_1), c=color_1, cmap='nipy_spectral')

    else:
        for slc_1, color_1, label_1, size_1 in zip(slices_1, args.colors_1, labels_1, args.sizes_1):
            if args.plot_molecules:
                # Plots molecules
                handles.append(mpatches.Patch(color=color_1, label=label_1))

                for smile, (x, y) in zip(smiles_1[slc_1], X_1[slc_1]):
                    img = Draw.MolsToGridImage([Chem.MolFromSmiles(smile)], molsPerRow=1, subImgSize=(200, 200))
                    imagebox = offsetbox.AnnotationBbox(offsetbox.OffsetImage(img), (x, y), bboxprops=dict(color=color_1))
                    ax.add_artist(imagebox)

            else:
                # Plots points
                plt.scatter(X_1[slc_1, 0], X_1[slc_1, 1], s=150 * size_1, color=color_1, label=label_1)


        if args.plot_molecules:
            legend_kwargs['handles'] = handles


    if args.cluster:
        plt.scatter(X_2[:, 0], X_2[:, 1], s=150 * np.mean(args.sizes_2), c=color_2, cmap='nipy_spectral')

    else:
        for slc_2, color_2, label_2, size_2 in zip(slices_2, args.colors_2, labels_2, args.sizes_2):
            if args.plot_molecules:
                # Plots molecules
                handles.append(mpatches.Patch(color=color_2, label=label_2))

                for smile, (x, y) in zip(smiles[slc_2], X[slc_2]):
                    img = Draw.MolsToGridImage([Chem.MolFromSmiles(smile)], molsPerRow=1, subImgSize=(200, 200))
                    imagebox = offsetbox.AnnotationBbox(offsetbox.OffsetImage(img), (x, y), bboxprops=dict(color=color_2))
                    ax.add_artist(imagebox)
            else:
                # Plots points
                plt.scatter(X_2[slc_2, 0], X_2[slc_2, 1], s=150 * size_2, color=color_2, label=label_2)

        if args.plot_molecules:
            legend_kwargs['handles'] = handles

    plt.legend(**legend_kwargs)
    plt.xticks([]), plt.yticks([])

    print('Saving t-SNE')
    plt.savefig(args.save_path)

"""
** For third file
    smile = []

    if args.cluster:
        plt.scatter(X_3[:, 0], X_3[:, 1], s=150 * np.mean(args.sizes_3), c=color_3, cmap='nipy_spectral')

    else:
        for slc_3, color_3, label_3, size_3 in zip(slices_3, args.colors_3, labels_3, args.sizes_3):
            if args.plot_molecules:
                # Plots molecules
                handles.append(mpatches.Patch(color=color_3, label=label_3))

                for smile, (x, y) in zip(smiles[slc_3], X[slc_3]):
                    img = Draw.MolsToGridImage([Chem.MolFromSmiles(smile)], molsPerRow=1, subImgSize=(200, 200))
                    imagebox = offsetbox.AnnotationBbox(offsetbox.OffsetImage(img), (x, y), bboxprops=dict(color=color_3))
                    ax.add_artist(imagebox)
            else:
                # Plots points
                plt.scatter(X_3[slc_3, 0], X_3[slc_3, 1], s=150 * size_2, color=color_3, label=label_3)

        if args.plot_molecules:
            legend_kwargs['handles'] = handles

"""


if __name__ == '__main__':
    compare_datasets_tsne(Args().parse_args())

