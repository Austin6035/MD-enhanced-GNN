import torch
import numpy as np
import pandas as pd
from rdkit.Avalon import pyAvalonTools
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import Chem, DataStructs
from sklearn.preprocessing import OneHotEncoder
from mendeleev import element
from pymatgen.core.structure import IMolecule
from torch_geometric.data import Data, Dataset


def atom():
    feature_lists = {
        'number': [],
        'ionenergies': [],
        'atomic_radius': [],
        'group_id': [],
        'period_list': [],
        'nvalence': [],
        'en_pauling': [],
        'atomic_volume': [],
        'electron_affinity': []
    }

    for i in range(1, 10):
        Ele = element(i)
        feature_lists['number'].append(i)
        feature_lists['ionenergies'].append(Ele.ionenergies[1])
        feature_lists['atomic_radius'].append(Ele.atomic_radius)
        feature_lists['group_id'].append(Ele.group_id)
        feature_lists['period_list'].append(Ele.period)
        feature_lists['nvalence'].append(Ele.nvalence())
        feature_lists['en_pauling'].append(Ele.en_pauling)
        feature_lists['atomic_volume'].append(Ele.atomic_volume)
        feature_lists['electron_affinity'].append(Ele.electron_affinity)

    encoded_features = {}

    for feature_name, feature_values in feature_lists.items():
        feature_array = np.array(feature_values).reshape(-1, 1)
        OHE = OneHotEncoder()
        OHE.fit(feature_array)
        encoded_features[feature_name] = OHE.transform(feature_array).toarray()

    return tuple(encoded_features.values())


a = atom()


def pf_binary_features_generator(mol, pf_name):
    mol = Chem.MolFromSmiles(mol) if type(mol) == str else mol
    if pf_name == 'atom_pair':
        features_vec = AllChem.GetHashedAtomPairFingerprintAsBitVect(mol)
        features = np.zeros((1,))
        DataStructs.ConvertToNumpyArray(features_vec, features)
        return features
    elif pf_name == 'avalon':
        features_vec = pyAvalonTools.GetAvalonFP(mol)
        features = np.zeros((1,))
        DataStructs.ConvertToNumpyArray(features_vec, features)
        return features
    elif pf_name == 'MACCS':
        features_vec = AllChem.GetMACCSKeysFingerprint(mol)
        features = np.zeros((1,))
        DataStructs.ConvertToNumpyArray(features_vec, features)
        return features
    elif pf_name == 'Morgan':
        features_vec = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2)
        features = np.zeros((1,))
        DataStructs.ConvertToNumpyArray(features_vec, features)
        return features
    elif pf_name == 'RDK':
        features_vec = Chem.RDKFingerprint(mol)
        features = np.zeros((1,))
        DataStructs.ConvertToNumpyArray(features_vec, features)
        return features
    elif pf_name == 'TT':
        features_vec = AllChem.GetHashedTopologicalTorsionFingerprintAsBitVect(mol)
        features = np.zeros((1,))
        DataStructs.ConvertToNumpyArray(features_vec, features)
        return features


def get(mol):
    all_nbrs = []
    atom_fea_list = []

    for atom in mol:
        nbrs = mol.get_neighbors(atom, radius)
        all_nbrs.append(nbrs)
        number = atom.specie.number
        n = number - 1
        arrays_to_concatenate = [arr[n] for arr in a]
        result = np.concatenate(arrays_to_concatenate, axis=0)
        atom_fea_list.append(result)

    all_nbrs = [sorted(nbrs, key=lambda x: x[1]) for nbrs in all_nbrs]
    atom_fea = torch.Tensor(np.array(atom_fea_list))

    return all_nbrs, atom_fea


def format_adj_matrix(adj_matrix):
    size = len(adj_matrix)
    src_list = list(range(size))
    all_src_nodes = torch.tensor([[x] * adj_matrix.shape[1] for x in src_list]).view(-1).long().unsqueeze(0)
    all_dst_nodes = adj_matrix.view(-1).unsqueeze(0)
    return torch.cat((all_src_nodes, all_dst_nodes), dim=0)


class GaussianDistance(object):

    def __init__(self, dmin, dmax, step, var=None):

        assert dmin < dmax
        assert dmax - dmin > step
        self.filter = np.arange(dmin, dmax + step, step)
        if var is None:
            var = step
        self.var = var

    def expand(self, distances):

        return np.exp(-(distances[..., np.newaxis] - self.filter) ** 2 /
                      self.var ** 2)


radius = 8
max_num_nbr = 12
gdf = GaussianDistance(0, radius, 0.2)


class MyDataset(Dataset):
    def __init__(self, target_file, structure_dir, feature_file):
        super(MyDataset, self).__init__()
        target_df = pd.read_csv(target_file, header=0)
        feature_df = pd.read_csv(feature_file, header=0, index_col=0)
        self.structure_dir = structure_dir
        self.filename = target_df.loc[:, 'filename'].values
        self.smiles = target_df.loc[:, 'smiles'].values
        self.target_p = target_df.loc[:, 'p(GPa)'].values
        self.target_Q = target_df.loc[:, 'Q(cal/g)'].values
        self.target_D = target_df.loc[:, 'D(km/s)'].values
        self.feature = feature_df.iloc[:, :].values

    def __getitem__(self, index):
        filename = self.filename[index]
        p_target = self.target_p[index]
        smiles = self.smiles[index]
        Q_target = self.target_Q[index]
        D_target = self.target_D[index]
        feature = self.feature[index]

        mol = IMolecule.from_file(self.structure_dir + str(filename))
        MOL = Chem.MolFromSmiles(smiles)
        MOL = Chem.AddHs(MOL)

        Morgan_fp = torch.tensor(pf_binary_features_generator(MOL, pf_name='Morgan'))
        atom_pair_fp = torch.tensor(pf_binary_features_generator(MOL, pf_name='atom_pair'))
        avalon_fp = torch.tensor(pf_binary_features_generator(MOL, pf_name='avalon'))
        MACCS_fp = torch.tensor(pf_binary_features_generator(MOL, pf_name='MACCS'))
        TT_fp = torch.tensor(pf_binary_features_generator(MOL, pf_name='TT'))
        RDK_fp = torch.tensor(pf_binary_features_generator(MOL, pf_name='RDK'))

        all_nbrs, atom_fea = get(mol)
        nbr_fea_idx, nbr_fea = [], []
        for nbr in all_nbrs:
            if len(nbr) < max_num_nbr:
                nbr_fea_idx.append(list(map(lambda x: x[2], nbr)) +
                                   [0] * (max_num_nbr - len(nbr)))
                nbr_fea.append(list(map(lambda x: x[1], nbr)) +
                               [radius + 1.] * (max_num_nbr -
                                                len(nbr)))
            else:
                nbr_fea_idx.append(list(map(lambda x: x[2],
                                            nbr[:max_num_nbr])))
                nbr_fea.append(list(map(lambda x: x[1],
                                        nbr[:max_num_nbr])))
        nbr_fea_idx, nbr_fea = np.array(nbr_fea_idx), np.array(nbr_fea)
        nbr_fea = gdf.expand(nbr_fea)
        nbr_fea = torch.Tensor(nbr_fea)
        nbr_fea = nbr_fea.view(-1, 41)

        nbr_fea_idx = format_adj_matrix(torch.LongTensor(nbr_fea_idx))
        p_target = torch.Tensor([float(p_target)])
        Q_target = torch.Tensor([float(Q_target)])
        D_target = torch.Tensor([float(D_target)])
        feature = torch.Tensor(feature)

        graph = Data(x=atom_fea, edge_index=nbr_fea_idx, edge_attr=nbr_fea, p=p_target,
                     Q=Q_target, D=D_target, morgan_fp=Morgan_fp, atom_pair_fp=atom_pair_fp,
                     avalon_fp=avalon_fp, MACCS_fp=MACCS_fp, TT_fp=TT_fp, RDK_fp=RDK_fp,
                     feature=feature)
        return graph

    def __len__(self):
        return len(self.label)


dataset = MyDataset()
data = []
for item in tqdm(dataset):
    data.append(item)
torch.save(data, 'dataset.pt')
