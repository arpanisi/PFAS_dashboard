from rdkit import Chem
from .atom_count import count_Atom, standard_mol
import re

#z = Chem.MolFromSmarts()
#z = Chem.MolToSmarts()
# Perfluorinated alkanes (PFAs)

def classify_pfas(smiles):
    x = count_Atom(smiles)
    if x != -1:
        num_f = x['F']
        num_c = x['C']
        num_atoms_auto = x['nums']
        if num_atoms_auto == num_c + num_f and num_c * 2 + 2 == num_f:
            return True
    return False


def classify_pfaenes(smiles):
    x = count_Atom(smiles)
    if x != -1:
        num_f = x['F']
        num_c = x['C']
        num_atoms_auto = x['nums']
        if num_atoms_auto == num_c + num_f \
                and num_c * 2 == num_f \
                and '=' in smiles:
            return True
    return False

# Perfluoroalkyl alcohols (PFACs)



def classify_pfacs(smiles):
    x = count_Atom(smiles)
    if x != -1:
        num_o = x['O']
        num_h = x['H']
        num_f = x['F']
        num_c = x['C']
        num_atoms_auto = x['nums']
        if num_atoms_auto == num_c + num_f + num_o + num_h \
                and '=' not in smiles \
                and num_h == 1 \
                and num_o == 1 \
                and smiles.startswith('OC'):
            return True
    return False


# Perfluoroalkyl ketones (PFAKs)


def classify_pfaks(smiles):
    x = count_Atom(smiles)
    if x != -1:
        num_f = x['F']
        num_c = x['C']
        num_o = x['O']
        num_atoms_auto = x['nums']
        #pattern = re.compile(r'O=C\((C\(F\)\(F\))+F\)(C\(F\)\(F\))+F')
        if num_c * 2 == num_f \
            and num_c + 1 - num_f / 2 == 1 \
            and num_atoms_auto == num_c + num_f + num_o \
            and num_o == 1 \
            and smiles.startswith('O=C'):
            return True
    return False

def classify_occco(smiles):
    pattern1 = re.compile('C\(=O\)CC\(=O\)(C\(F\)\(F\))+')
    pattern2 = re.compile(r'O=C1C+C1C\(=O\)(C\(F\)\(F\))+F')
    if (pattern1.search(smiles) is not None) or (pattern2.search(smiles) is not None):
        return True
    return False



def classify_nf(smiles):
    pattern = re.compile(r'.*NC+(C\(F\)\(F\))+F')
    if pattern.search(smiles) is not None:
        return True
    return False

def classify_ncc(smiles):
    pattern1 = re.compile(r'.*N.*CC(C\(F\)\(F\))+F')
    pattern2 = re.compile(r'C\[N\+\]\(C\)\(CCC\(F\)(C\(F\)\(F\))+F\)CC\(=O\)O')
    if (pattern1.search(smiles) is not None) or (pattern2.search(smiles) is not None):
        return True
    return False


def classify_fccoff(smiles):
    pattern1 = re.compile(r'O=C\(\/C\(F\)=C\(\/F\)C\(F\)\((C\(F\)\(F\)F\))+(C\(F\)\(F\))+F')
    pattern2 = re.compile(r'O=C\(C\(F\)=C\(F\)C\(F\)\((C\(F\)\(F\)F\))+(C\(F\)\(F\))+F')
    pattern3 = re.compile(r'O=C\(\/C\(F\)=C\(\\F\)C\(F\)\((C\(F\)\(F\)F\))+(C\(F\)\(F\))+F')
    if (pattern1.search(smiles) is not None) or (pattern2.search(smiles) is not None) or (
            pattern3.search(smiles) is not None):
        return True
    return False



def classify_con(smiles):   # quan F
    pattern = re.compile(r'(C\(F\)\(F\))+F')
    patt1 = Chem.MolFromSmiles('CC(F)(F)C(N)=O')
    patt2 = Chem.MolFromSmiles('COC')
    m = Chem.MolFromSmiles(smiles)
    flag1 = m.HasSubstructMatch(patt1)
    flag2 = m.HasSubstructMatch(patt2)
    if flag1 and not flag2 and (pattern.search(smiles) is not None):
        return True
    return False




def classify_ffcfo(smiles):
    pattern1 = re.compile(r'FC\(F\)=C\(F\)O(C\(F\)\(F\))+')
    pattern2 = re.compile(r'(C\(F\)\(F\))+OC\(F\)=C\(F\)F')
    pattern3 = re.compile(r'^C+O(C\(F\)\(F\))+F$')
    x = count_Atom(smiles)
    if x != -1:
        num_f = x['F']
        num_c = x['C']
        num_o = x['O']
        num_cl = x['Cl']
        num_br = x['Br']
        num_h = x['H']
        num_atoms_auto = x['nums']
        if num_c + num_f + num_h + num_o + num_br + num_cl == num_atoms_auto \
            and '=' in smiles \
            and ((pattern1.search(smiles) is not None) or (pattern2.search(smiles) is not None) or (pattern3.search(smiles) is not None)):
            return True
    return False

def classify_fof(smiles):
    pattern = re.compile(r'O=C\(CC\(=O\)(C\(F\)\(F\))+F\)(C\(F\)\(F\))+F')
    if pattern.search(smiles) is not None:
        return True
    return False


'''
def classifying_other_perfluoroalkyls(pfas):
    #if classify_pfaenes(pfas):
     #   return ['PFAA precursors', 'PFAenes']  # shuangjian =   perfluoroalkenes
    #if classify_pfacs(pfas):
     #   return ['PFAA precursors', 'PFACs']   # chun    perfluoroalkyl alcohols
    #if classify_pfaks(pfas):
     #   return ['PFAA precursors', 'PFAKs']  # tong
   # if classify_fof(pfas):
    #    return ['PFAA precursors', 'PFAKs']  # di tong
    if classify_ffcfo(pfas):
        return ['PFAA precursors', 'PFAenes-derivatives']  # xi ji O
    if classify_occco(pfas):
        return ['PFAA precursors', 'PFAK-derivatives']  # 2 tong
    if classify_fccoff(pfas):
        return ['PFAA precursors', 'PFAK-derivatives']   # tong = xiting
    if classify_con(pfas):
        return ['PFAA precursors', 'PFAK-derivatives']
    return None
'''