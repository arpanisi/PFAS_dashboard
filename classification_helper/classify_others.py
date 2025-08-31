from .atom_count import count_Atom
from rdkit import Chem



def classify_perfluoroalkylethers(smiles):
    patt = Chem.MolFromSmiles('FC(F)(OC(F)(C)F)C')
    m = Chem.MolFromSmiles(smiles)
    flag = m.HasSubstructMatch(patt)
    x = count_Atom(smiles)
    if x != -1:
        num_f = x['F']
        num_c = x['C']
        num_o = x['O']
        num_atoms_auto = x['nums']
        if num_c + num_f + num_o == num_atoms_auto \
                and num_c * 2 + 2 == num_f \
                and num_o >= 1 \
                and flag \
                and num_c + 1 - num_f / 2 == 0:
            return True
    return False


def classify_ptfes(smiles):
    patt1 = Chem.MolFromSmiles('FC(C(F)(C)F)(F)C')
    m1 = Chem.MolFromSmiles(smiles)
    flag1 = m1.HasSubstructMatch(patt1)
    if flag1:
        return True
    return False


def classify_pvdf(smiles):
    patt1 = Chem.MolFromSmiles('CCC(F)(C)F')
    m = Chem.MolFromSmiles(smiles)
    flag = m.HasSubstructMatch(patt1)
    if flag:
        return True
    return False


def classify_fep(smiles):
    patt1 = Chem.MolFromSmiles('FC(C(F)(C)F)(F)C')
    patt2 = Chem.MolFromSmiles('FC(F)(F)C(C)(C(F)(C)F)F')
    m1 = Chem.MolFromSmiles(smiles)
    flag1 = m1.HasSubstructMatch(patt1)
    flag2 = m1.HasSubstructMatch(patt2)
    if flag1 and flag2:
        return True
    return False

def classify_pfa(smiles):
    patt1 = Chem.MolFromSmiles('CC(F)(OC(F)(C(F)(C(F)(F)F)F)F)C(F)(C)F')
    patt2 = Chem.MolFromSmiles('FC(C(F)(C)F)(F)C')
    m = Chem.MolFromSmiles(smiles)
    flag1 = m.HasSubstructMatch(patt1)
    flag2 = m.HasSubstructMatch(patt2)
    if flag1 and flag2:
        return True
    return False

def classify_pfpes(smiles):
    patt1 = Chem.MolFromSmiles('FC(F)(C)OC')
    m1 = Chem.MolFromSmiles(smiles)
    flag1 = m1.HasSubstructMatch(patt1)
    patt2 = Chem.MolFromSmiles('FC(F)(OC)C')
    flag2 = m1.HasSubstructMatch(patt2)
    if flag1 and flag2:
        return True
    return False

def classify_cnf(smiles):  # 103  104
    x = count_Atom(smiles)
    if x != -1:
        num_c = x['C']
        num_n = x['N']
        num_f = x['F']
        num_atoms_auto = x['nums']
        if num_c + num_n + num_f == num_atoms_auto and (num_c + 1 - (num_f - num_n) / 2 == 0
                                                        or ('N1' in smiles and num_c + 1 - (num_f - num_n) / 2 == 1)):
            return True
        return False




def classify_nco(smiles):
    patt1 = Chem.MolFromSmiles('CN=C=O')
    m = Chem.MolFromSmiles(smiles)
    flag1 = m.HasSubstructMatch(patt1)
    patt2 = Chem.MolFromSmiles('CC#N')
    flag2 = m.HasSubstructMatch(patt2)
    patt3 = Chem.MolFromSmiles('C[N+](=O)[O-]')
    flag3 = m.HasSubstructMatch(patt3)
    patt4 = Chem.MolFromSmiles('CNCC(O)CC(C)(F)F')
    flag4 = m.HasSubstructMatch(patt4)
    if flag1 or flag2 or flag3 or flag4:
        return True
    return False

def perfluoropolyethers(smiles):
    patt1 = Chem.MolFromSmiles('FC(C)(F)OC')
    m = Chem.MolFromSmiles(smiles)
    flag1 = m.HasSubstructMatch(patt1)
    if flag1:
        return True
    return False


def classifying_others(smiles):
    if classify_cnf(smiles):
        return ['Other PFASs', 'Perfluoroalkyl-tert-amines']
    #if classify_n(smiles):
     #   return ['other PFASs', ' derivatives']
    #if classify_nco(smiles):
     #   return ['other PFASs', 'PFASs containing N']     # Isocys
    if classify_pfa(smiles):
        return ['Other PFASs', 'others']
    if classify_fep(smiles):
        return ['Other PFASs', 'others']
    if perfluoropolyethers(smiles):
        return ['Other PFASs', 'PFPEs']
    if classify_pvdf(smiles):
        return ['Other PFASs', 'others']
    if classify_ptfes(smiles):
        return ['Other PFASs', 'others']






