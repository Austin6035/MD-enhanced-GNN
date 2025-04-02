import deepchem as dc
import numpy as np
import pandas as pd
from rdkit import Chem


def del_corr(input_filepath, output_filepath):
    df = pd.read_csv(input_filepath, header=0, index_col=0)
    df = df.drop(labels=['Chi0', 'Chi0n', 'Chi0v', 'Chi1', 'Chi1n', 'Chi1v', 'Chi2v', 'Chi3v', 'Chi4v',
                         'fr_ketone_Topliss', 'fr_COO2', 'HeavyAtomMolWt', 'NumValenceElectrons', 'LabuteASA',
                         'MaxAbsEStateIndex', 'ExactMolWt', 'NumAromaticHeterocycles', 'HeavyAtomCount',
                         'MolMR', 'fr_phenol_noOrthoHbond', 'fr_nitro_arom', 'NumAliphaticHeterocycles',
                         'fr_Nhpyrrole', 'Chi3n', 'NumHeteroatoms', 'fr_Ar_N', 'fr_C_O_noCOO'], axis=1) 
                         # 根据计算的相关性数值修改列表内容
    df.to_csv(output_filepath)


del_corr('origin_feature.csv', 'final_feature.csv')

