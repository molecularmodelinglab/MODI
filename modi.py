import numpy as np
import pandas as pd
from rdkit.Chem.PandasTools import LoadSDF
from scipy import spatial as sp
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.DataStructs import ConvertToNumpyArray


NBITS = 2048
RADIUS = 3
CHIRAL = False


def nearest_neighbors(reference, query, k=1, self_query=False, return_distance=False):
    """
    Gets the k nearest neighbors of reference set for each row of the query set

    Parameters
    ----------
        reference : array_like
            An array of points to where nearest neighbors are pulled.
        query : array_like
            An array of points to query nearest neighbors for
        k : int > 0, optional
            the number of nearest neighbors to return
        self_query : bool, optional
            if reference and query are same set of points, set to True
            to avoid each query returning itself as its own nearest neighbor
        return_distance : bool, optional
            if True, return distances of nearest neighbors
    Returns
    -------
        i : integer or array of integers
            The index of each neighbor in reference
            has shape [q, k] where q is number of rows in query
        d : float or array of floats, optional
            if return_distance set to true, returns associated
            euclidean distance of each nearest neighbor
            d is element matched to i, ei the distance of i[a,b] is d[a,b]
            The distances to the nearest neighbors.
    """

    tree = sp.KDTree(reference)

    if self_query:
        k = [x+2 for x in range(k)]
    else:
        k = [x+1 for x in range(k)]

    d, i = tree.query(query, k=k, workers=-1)

    if return_distance:
        return i, d
    else:
        return i


def modi(data, labels, return_class_contribution=False):
    """
    Gets the MODI from the given data and label set

    Parameters
    ----------
        data : array_like
            An array chemical descriptors (rows are chemicals and columns are descriptors).
        labels : array_like
            An array labels that are row matched to the data array
        return_class_contribution : bool, optional
            if True, return the normalized MODI for each class. Useful for imbalanced datasets
    Returns
    -------
        modi : float
            the calculated MODI for the given data and label
        class_contrib : list of tuples of length 2 (str, float), optional
            if return_class_contribution set to true, returns associated
            MODI score for each class in the data as a tuple of (class, MODI)
    """
    # get all the classes present in the dataset
    classes = np.unique(labels)
    k = classes.shape[0]

    # get the labels of the nearest neighbors
    nn_idx = nearest_neighbors(data, data, k=1, self_query=True)
    nn_labels = labels[nn_idx]

    # calculate the modi
    modi_value = 0
    class_contrib = []

    # loop through each class
    for c in classes:
        c_arr = np.where(labels == c)[0]
        c_labels = labels[c_arr]
        c_nn_labels = nn_labels[c_arr].flatten()

        modi_value += np.sum(c_labels == c_nn_labels) / c_arr.shape[0]
        class_contrib.append((c, np.sum(c_labels == c_nn_labels) / c_arr.shape[0]))

    if not return_class_contribution:
        return (k ** -1) * modi_value
    else:
        return (k ** -1) * modi_value, class_contrib


def get_morgan_finger(mol):
    """
    Get morgan fingerprint from RDKit Mol object
    """
    fp = np.zeros(NBITS)
    ConvertToNumpyArray(AllChem.GetMorganFingerprintAsBitVect(mol, nBits=NBITS, radius=RADIUS, useChirality=CHIRAL), fp)
    return fp


def modi_from_sdf(sdf_file, label_col, return_class_contribution=False):
    """
    Wrapper to get MODI from a .sdf file

    Parameters
    ----------
        sdf_file : str
            file path to sdf file
        label_col : str or int
            the column name/index of the column containing the binary labels
        return_class_contribution : bool, optional
            if True, return the normalized MODI for each class. Useful for imbalanced datasets
    Returns
    -------
        modi : float
            the calculated MODI for the given data and label
        class_contrib : list of tuples of length 2 (str, float), optional
            if return_class_contribution set to true, returns associated
            MODI score for each class in the data as a tuple of (class, MODI)
    """
    df = LoadSDF(sdf_file)
    return modi(np.vstack(df["ROMol"].apply(get_morgan_finger).to_numpy()), df[label_col].to_numpy(), return_class_contribution)


def modi_from_csv(csv_file, label_col, smiles_col="SMILES", return_class_contribution=False):
    """
    Wrapper to get MODI from a .csv file

    Parameters
    ----------
        csv_file : str
            file path to csv file
        label_col : str or int
            the column name/index of the column containing the binary labels
        smiles_col : str or int, optional
            the column name/index of the column containing the smiles strings. Defaults to "SMILES"
        return_class_contribution : bool, optional
            if True, return the normalized MODI for each class. Useful for imbalanced datasets
    Returns
    -------
        modi : float
            the calculated MODI for the given data and label
        class_contrib : list of tuples of length 2 (str, float), optional
            if return_class_contribution set to true, returns associated
            MODI score for each class in the data as a tuple of (class, MODI)
    """
    df = pd.read_csv(csv_file)
    df["ROMol"] = df[smiles_col].apply(Chem.MolFromSmiles)
    df.dropna(inplace=True)
    return modi(np.vstack(df["ROMol"].apply(get_morgan_finger).to_numpy()), df[label_col].to_numpy(), return_class_contribution)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument("file", help="input file for modi calculation")
    parser.add_argument("-l", "--label", type=str, help="column name/index of binary label", required=True)
    parser.add_argument("-n", "--nbit", type=int, help="number of morgan fingerprint bits to use", default=NBITS)
    parser.add_argument("-r", "--radius", type=int, help="radius of morgan fingerprint to use", default=RADIUS)
    parser.add_argument("-c", "--chiral", action="store_true", help="use chiral in fingerprint generation")
    parser.add_argument("-s", "--smiles", help="column name/index of smiles (only if using csv file)", default="SMILES")
    parser.add_argument("-p", "--partial", action="store_true", help="return partial/un-normalized MODI contribution "
                                                                     "of each label class")

    args = parser.parse_args()

    if args.chiral:
        CHIRAL = True

    if ".csv" in args.file:
        m = modi_from_csv(args.file, args.label, args.smiles, args.partial)
    else:
        m = modi_from_sdf(args.file, args.label, args.partial)

    if args.partial:
        m, class_contribution = m
        print(f'MODI: {m}\nClass Contribution: {class_contribution}')
    else:
        print(f'MODI: {m}')
