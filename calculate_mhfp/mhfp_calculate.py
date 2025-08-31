import pandas as pd
from mhfp.encoder import MHFPEncoder
from rdkit import Chem


def calculate_MHFP(smiles):
    mhfp_encoder = MHFPEncoder()
    mol = Chem.MolFromSmiles(smiles)
    if mol is not None:
        mhfps = mhfp_encoder.encode_mol(mol)
        return mhfps
    return [-1]

def List_SMI(list):
    list_MHFP = []
    if len(list) != 0:
        for i in list:
            MHFPS = calculate_MHFP(i)
            list_MHFP.append([i, MHFPS])
        return list_MHFP
    return -1



'''
if __name__ == "__main__":
    smi = 'c1c2=C(C3=[N]4C(=C(c5ccc6C(=C7C=CC8=[N]7[Ti]74(n2c(=C8c2ccccc2)c1)(n56)Oc1ccc(cc1O7)C=O)C#Cc1ccc(cc1)C(=O)O)c1ccccc1)C=C3)c1c(cccc1OCOC'
    mol = Chem.MolFromSmiles(smi)
    print(str(type(mol)))
'''