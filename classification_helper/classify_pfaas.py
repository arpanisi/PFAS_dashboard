from .atom_count import count_Atom, standard_mol
import re
from rdkit import Chem


def classify_pfcas(smiles):
    x = count_Atom(smiles)
    if x != -1:
        num_c = x['C']
        num_h = x['H']
        num_o = x['O']
        num_f = x['F']
        num_atoms_auto = x['nums']
        if (num_c - 1) * 2 + 1 == num_f \
                and num_h == 1 \
                and num_o == 2 \
                and num_c + num_h + num_f + num_o == num_atoms_auto \
                and smiles.startswith('O=C(O)'):
            return True
    return False


def classify_pfsas(smiles):
    x = count_Atom(smiles)
    if x != -1:
        num_f = x['F']
        num_c = x['C']
        num_o = x['O']
        num_h = x['H']
        num_s = x['S']
        num_atoms_auto = x['nums']
        if num_c * 2 + 1 == num_f \
                and num_h == 1 \
                and num_o == 3 \
                and num_s == 1 \
                and num_c + num_h + num_f + num_o + num_s == num_atoms_auto \
                and smiles.startswith('O=S(=O)(O)'):
            return True
    return False


def classify_pfsias(smiles):
    x = count_Atom(smiles)
    if x != -1:
        num_f = x['F']
        num_c = x['C']
        num_h = x['H']
        num_s = x['S']
        num_o = x['O']
        num_atoms_auto = x['nums']
        if num_c * 2 + 1 == num_f \
                and num_h == 1 \
                and num_o == 2 \
                and num_s == 1 \
                and num_c + num_h + num_f + num_o + num_s == num_atoms_auto \
                and smiles.startswith('O=S(O)'):
            return True
    return False


def classify_pfecas(smiles):
    x = count_Atom(smiles)
    if x != -1:
        num_f = x['F']
        num_c = x['C']
        num_h = x['H']
        num_o = x['O']
        num_atoms_auto = x['nums']
        if ((num_c - 1) * 2 + 1 == num_f
                and num_h == 1
                and num_o > 2
                and num_c + num_h + num_f + num_o == num_atoms_auto
                and smiles.startswith('O=C(O)')):
            return True
    return False


def classify_pfesas(smiles):
    x = count_Atom(smiles)
    if x != -1:
        num_f = x['F']
        num_c = x['C']
        num_h = x['H']
        num_o = x['O']
        num_s = x['S']
        num_atoms_auto = x['nums']
        if (num_c * 2 + 1 == num_f
                and num_h == 1
                and num_o > 3
                and num_s == 1
                and num_c + num_h + num_f + num_o + num_s == num_atoms_auto
                and smiles.startswith('O=S(=O)(O)')):
            return True
    return False


def classify_pfpas(smiles):
    x = count_Atom(smiles)
    if x != -1:
        num_f = x['F']
        num_c = x['C']
        num_h = x['H']
        num_o = x['O']
        num_p = x['P']
        num_atoms_auto = x['nums']
        if num_c * 2 + 1 == num_f \
                and num_h == 2 \
                and num_o == 3 \
                and num_p == 1 \
                and num_c + num_h + num_f + num_o + num_p == num_atoms_auto \
                and smiles.startswith("O=P(O)(O)"):
            return True
    return False


def classify_pfpias(smiles):
    x = count_Atom(smiles)
    if x != -1:
        num_f = x['F']
        num_c = x['C']
        num_o = x['O']
        num_h = x['H']
        num_p = x['P']
        num_atoms_auto = x['nums']
        if num_c * 2 + 2 == num_f \
                and num_h == 1 \
                and num_o == 2 \
                and num_p == 1 \
                and num_c + num_h + num_f + num_o + num_p == num_atoms_auto \
                and smiles.startswith('O=P(O)'):
            return True
    return False


# xin jia   (pfaas)
def classify_pfdicas(smiles):
    pattern = re.compile(r'O=C\(O\)(C\(F\)\(F\))+C\(=O\)O')
    if pattern.search(smiles) is not None:
        return True
    return False

def classify_pfdisas(smiles):
    pattern = re.compile((r'O=S\(=O\)\(O\)(C\(F\)\(F\))+S\(=O\)\(=O\)O'))
    if pattern.search(smiles) is not None:
        return True
    return False

def classify_oco(smiles):
    pattern = re.compile(r'O=C\(OC\(=O\)(C\(F\)\(F\))+F\)(C\(F\)\(F\))+F')
    if pattern.search(smiles) is not None:
        return True
    return False


def classify_coco(smiles):
    pattern = re.compile(r'COC\(=O\)(C\(F\)\(F\))+C\(=O\)OC')
    pattern1 = re.compile(r'.*COC\(=O\)(C\(F\)\(F\))+F')
    pattern2 = re.compile(r'.*OC\(=O\)(C\(F\)\(F\))+C\(=O\)O.*')
    pattern3 = re.compile(r'.*COC\(=O\)(C\(F\)\(F\))+C\(F\)F')
    pattern4 = re.compile(r'OC\(=O\)(C\(F\)\(F\))+F')
    patt1 = Chem.MolFromSmiles('COOC')
    patt2 = Chem.MolFromSmiles('C=C')
    m = Chem.MolFromSmiles(smiles)
    flag1 = m.HasSubstructMatch(patt1)
    flag2 = m.HasSubstructMatch(patt2)
    x = count_Atom(smiles)
    if x != -1:
        num_f = x['F']
        num_c = x['C']
        num_o = x['O']
        num_h = x['H']
        num_atoms_auto = x['nums']
        if num_c + num_h + num_f + num_o == num_atoms_auto \
            and not flag1 \
            and not flag2 \
            and not (smiles.endswith('C(F)F')) \
            and ((pattern1.search(smiles) is not None) or (pattern.search(smiles) is not None) or (pattern2.search(smiles) is not None) or (pattern3.search(smiles) is not None) or (pattern4.search(smiles) is not None)):
            return True
    return False



def classify_si(smiles):
    if ('Si' in smiles) or ('[Si]' in smiles):
        return True
    return False

def classify_metals(smiles):
    if ('Br' not in smiles) and (('Hg' in smiles) or ('Sb' in smiles) or ('Se' in smiles) or ('Ge' in smiles) or ('B' in smiles) or ('Sn' in smiles)
        or ('[Se]' in smiles) or ('[Hg]' in smiles) or ('[Sb]' in smiles) or ('[Ge]' in smiles) or ('[B]' in smiles) or ('[Sn]' in smiles)):
        return True
    return False

def classify_sohoh(smiles):
    patt1 = Chem.MolFromSmiles('O=S(OC)(C(F)(C)F)=O')
    patt2 = Chem.MolFromSmiles('FC(C)(S(=O)(O)=O)F')
    m = Chem.MolFromSmiles(smiles)
    flag1 = m.HasSubstructMatch(patt1)
    flag2 = m.HasSubstructMatch(patt2)
    x = count_Atom(smiles)
    if x != -1:
        num_f = x['F']
        num_c = x['C']
        num_o = x['O']
        num_h = x['H']
        num_s = x['S']
        num_atoms_auto = x['nums']
        if num_c + num_h + num_f + num_o + num_s == num_atoms_auto \
                and (flag1 or flag2):
            return True
    return False

def classify_polyfesas(smiles):
    patt = Chem.MolFromSmiles('COS(=O)(=O)O')
    m = Chem.MolFromSmiles(smiles)
    flag = m.HasSubstructMatch(patt)
    if smiles.startswith("O=S(=O)(O)"):
        smiles = smiles.replace('O=S(=O)(O)', 'F', 1)
        x = count_Atom(smiles)
        if x != -1:
            num_f = x['F']
            num_c = x['C']
            num_cl = x['Cl']
            num_br = x['Br']
            num_i = x['I']
            num_h = x['H']
            num_o = x['O']
            num_atoms_auto = x['nums']
            if (not flag) \
                and (num_f + num_h + num_c + num_br + num_cl + num_br + num_i + num_o == num_atoms_auto) \
                and num_o >= 1 \
                and (num_c + 1 - (num_f + num_h + num_cl + num_br + num_i) / 2 == 0):
                return True
    return False

def classifying_pfaas(smiles):
    if classify_si(smiles):
        return ['Other PFASs', 'Si PFASs']
    if classify_metals(smiles):
        return ['Other PFASs', '(Hg,Sn,Ge,Sb,Se,B) PFASs']
    if classify_pfcas(smiles):
        return ['PFAAs', 'PFCAs']
    if classify_pfsas(smiles):
        return ['PFAAs', 'PFSAs']
    if classify_pfsias(smiles):
        return ['PFAAs', 'PFSIAs']
    if classify_pfecas(smiles):
        return ['PFAAs', 'PFECAs']
    if classify_pfesas(smiles):
        return ['PFAAs', 'PFESAs']
    if classify_pfpas(smiles):
        return ['PFAAs', 'PFPAs']
    if classify_pfpias(smiles):
        return ['PFAAs', 'PFPIAs']
    if classify_pfdicas(smiles):
        return ['PFAAs', 'PFdiCAs']   # shuang suan
    if classify_pfdisas(smiles):
        return ['PFAAs', 'PFdiSAs']   # shuang s SUAN


#  new_add
    if classify_oco(smiles):
        return ['PFAAs', 'PFCA-anhydrides']   #  suan gan
    if classify_coco(smiles):
        return ['PFAAs', 'PFCA-ester derivatives']  #  zhi
    if classify_sohoh(smiles):
        return ['PFAAs', 'PFSA derivatives']
    return None
