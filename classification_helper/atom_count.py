from rdkit import Chem
from rdkit_helper import mol_from_smiles, rdkit_smiles_from_mol

def standard_mol(smiles):
    x = mol_from_smiles(smiles)
    return rdkit_smiles_from_mol(x)


def count_Atom(smiles):
    y = 0
    if len(smiles) != 0:
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            atoms = mol.GetAtoms()
            list = []
            count_dict = {'Cl': 0, 'F': 0, 'C': 0, 'O': 0, 'N': 0, 'S': 0, 'I': 0, 'P': 0, 'Br': 0}
            count_dict['nums'] = mol.GetNumAtoms()
            for at in atoms:
                x = at.GetSymbol()
                list.append(x)
                y = at.GetTotalNumHs() + y
            for k in list:
                count_dict[k] = list.count(k)
                count_dict.update({'H': y})
            count_dict['num'] = y + len(list)
            count_dict['nums'] = y + mol.GetNumAtoms()
            return count_dict
        return -1
    return -1

'''
if __name__ == "__main__":
    smi = 'FC(F)(F)C(F)(F)p1p(C(F)(F)C(F)(F)F)p1C(F)(F)C(F)(F)F'
    x = count_Atom(smi)
    print(x)
'''






