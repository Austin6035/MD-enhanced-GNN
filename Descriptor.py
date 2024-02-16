import deepchem as dc
import numpy as np
import pandas as pd
from rdkit import Chem


def get_feature(input_filepath, output_filepath):
    df = pd.read_csv(input_filepath, header=0, index_col=0)
    SMILES = df.loc[:, 'smiles'].values
    mols = np.array([Chem.MolFromSmiles(smiles) for smiles in SMILES])

    featurizer = dc.feat.RDKitDescriptors()
    feature_names = np.array([f[0] for f in featurizer.descList])

    features = featurizer.featurize(mols)
    ind, features = zip(*[(i, feat) for i, feat in enumerate(features) if len(feat) != 0])
    ind = list(ind)
    features = np.array(features)

    ipc_idx = np.where(feature_names == 'Ipc')
    feature_names[ipc_idx] = 'log_Ipc'
    features[:, ipc_idx] = np.log(features[:, ipc_idx] + 1)

    nonzero_sd = np.where(~np.isclose(np.std(features, axis=0), 0))[0]
    features = features[:, nonzero_sd]
    feature_names = feature_names[nonzero_sd]
    feature_names_list = list(feature_names)

    df = pd.DataFrame(features[ind], columns=feature_names_list, index=ind)
    df.to_csv(output_filepath)


def correction(input_filepath, output_filepath):
    df = pd.read_csv(input_filepath, encoding='gb18030', header=0, index_col=0)
    df.corr()
    df.corr().to_csv(output_filepath)


def del_corr(input_filepath, output_filepath):
    df = pd.read_csv(input_filepath, header=0, index_col=0)
    df = df.drop(labels=['Chi0', 'Chi0n', 'Chi0v', 'Chi1', 'Chi1n', 'Chi1v', 'Chi2v', 'Chi3v', 'Chi4v',
                         'fr_ketone_Topliss', 'fr_COO2', 'HeavyAtomMolWt', 'NumValenceElectrons', 'LabuteASA',
                         'MaxAbsEStateIndex', 'ExactMolWt', 'NumAromaticHeterocycles', 'HeavyAtomCount',
                         'MolMR', 'fr_phenol_noOrthoHbond', 'fr_nitro_arom', 'NumAliphaticHeterocycles',
                         'fr_Nhpyrrole', 'Chi3n', 'NumHeteroatoms', 'fr_Ar_N', 'fr_C_O_noCOO'], axis=1)
    df.to_csv(output_filepath)
