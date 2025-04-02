1. The target.csv file contains the SMILES of the dataset molecules and three detonation properties Q, D, and p.

2. get_original_features.py and get_final_features.py are used to obtain 152 molecular descriptor features of the dataset molecules.

3. The Dataset.py file is used to generate the dataset for graph neural networks.

4. The GATv2.py file is the molecular descriptor-enhanced graph neural network model.

# Use
python get_original_features.py
python get_final_features.py
python Dataset.py
python GATv2.py
