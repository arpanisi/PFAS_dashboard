import re
from .atom_count import count_Atom
from rdkit import Chem


def classify_ployfcas(smiles):   # PCAs OR ployfcas
    if smiles.startswith('O=C(O)'):
        smiles = smiles.replace("O=C(O)", "F", 1)
        x = count_Atom(smiles)
        if x != -1:
            num_f = x['F']
            num_c = x['C']
            num_cl = x['Cl']
            num_br = x['Br']
            num_h = x['H']
            num_atoms_auto = x['nums']
            if num_f + num_h + num_c + num_br + num_cl == num_atoms_auto \
                and num_c + 1 - (num_f + num_h + num_cl + num_br) / 2 == 0:
                return True
    return False

def classify_polyfecas(smiles):
    if smiles.startswith('O=C(O)'):
        smiles = smiles.replace('O=C(O)', 'F', 1)
        patt = Chem.MolFromSmiles('COC')
        m = Chem.MolFromSmiles(smiles)
        flag = m.HasSubstructMatch(patt)
        x = count_Atom(smiles)
        if x != -1:
            num_f = x['F']
            num_c = x['C']
            num_cl = x['Cl']
            num_br = x['Br']
            num_h = x['H']
            num_o = x['O']
            num_i = x['I']
            num_atoms_auto = x['nums']
            if num_f + num_h + num_c + num_br + num_cl + num_o + num_i == num_atoms_auto \
                and num_o >= 1 \
                and num_c + 1 - (num_f + num_h + num_cl + num_br + num_i) / 2 == 0 \
                and flag:
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




def classify_ccffhbr(smiles):
    patt = Chem.MolFromSmiles('C=C')
    m = Chem.MolFromSmiles(smiles)
    flag = m.HasSubstructMatch(patt)
    x = count_Atom(smiles)
    if x != -1:
        num_f = x['F']
        num_c = x['C']
        num_cl = x['Cl']
        num_br = x['Br']
        num_h = x['H']
        num_i = x['I']
        num_atoms_auto = x['nums']
        if num_c + num_f + num_cl + num_br + num_h + num_i == num_atoms_auto \
            and (num_br + num_cl + num_i + num_h >= 1) \
            and ('#' not in smiles) \
            and flag \
            and (num_c + 1 - (num_f + num_cl + num_br + num_h + num_i) / 2 == 1):
            return True
        return False





def classify_ooff(smiles):
    pattern1 = re.compile(r'.*C\(=O\)O(C\(F\)\(F\))+C+C\(F\)\(F\)F')
    pattern2 = re.compile(r'.*C\(=O\)O(C\(F\)\(F\))+C\(F\)CC\(F\)\(F\)F')
    pattern3 = re.compile(r'.*C\(=O\)OC\(F\)(C\(F\)\(F\))+CF')
    pattern4 = re.compile(r'.*C\(=O\)O(C\(F\)\(F\))+F.*')
    pattern5 = re.compile(r'C\(=O\)OC+(C\(F\)\(F\))+')
    patt = Chem.MolFromSmiles('CS(=O)(O)=O')
    m = Chem.MolFromSmiles(smiles)
    flag = m.HasSubstructMatch(patt)
    if ((pattern1.search(smiles) is not None) or (pattern2.search(smiles) is not None) or (pattern3.search(smiles) is not None) or (pattern4.search(smiles) is not None) or (pattern5.search(smiles) is not None)) \
        and not flag:
        return True
    return False




def classify_oclfff(smiles):
    if smiles.startswith('O=C(Cl)'):
        smiles = smiles.replace('O=C(Cl)', 'F')
        x = count_Atom(smiles)
        if x != -1:
            num_f = x['F']
            num_c = x['C']
            num_h = x['H']
            num_br = x['Br']
            num_cl = x['Cl']
            num_i = x['I']
            num_atoms_auto = x['nums']
            if num_c + num_f + num_h + num_cl + num_br + num_i == num_atoms_auto \
                and (num_c + 1 - (num_f + num_h + num_cl + num_br + num_i) / 2 == 0):
                return True
    return False

def classify_misuan(smiles):
    patt1 = Chem.MolFromSmiles('COC')
    patt2 = Chem.MolFromSmiles('CC(=O)O')
    patt3 = Chem.MolFromSmiles('CC(=O)OC(C)=O')
    patt4 = Chem.MolFromSmiles('O=C(C)C(OC)=O')
    m = Chem.MolFromSmiles(smiles)
    flag1 = m.HasSubstructMatch(patt1)
    flag2 = m.HasSubstructMatch(patt2)
    flag3 = m.HasSubstructMatch(patt3)
    flag4 = m.HasSubstructMatch(patt4)
    if flag1 and flag2 and not flag3 and not flag4:
        x = count_Atom(smiles)
        if x != -1:
            num_c = x['C']
            num_h = x['H']
            num_f = x['F']
            num_o = x['O']
            num_atoms_auto = x['nums']
            if num_c + num_f + num_h + num_o == num_atoms_auto \
                and num_o >= 3:
                return True
    return False

def classify_polydiether(smiles):
    patt = Chem.MolFromSmiles('COOC')
    m = Chem.MolFromSmiles(smiles)
    flag = m.HasSubstructMatch(patt)
    pattern = re.compile(r'(C(\((F|(Cl)|(I)|(Br))\)){0,2})+')
    if flag and pattern.search(smiles) is not None:
        return True
    return False

def classify_polyohderi(smiles):
    pattern = re.compile(r'^O(C(\((F|(Cl)|(I)|(Br))\)){0,2})+')
    if pattern.search(smiles) is not None:
        return True
    return False

def classify_polyoh(smiles):
    pattern = re.compile(r'^O(C(\((F|(Cl)|(I)|(Br))\)){0,2})+')
    if pattern.search(smiles) is not None \
        and '=' not in smiles \
        and 'S' not in smiles \
        and '#' not in smiles \
        and 'N' not in smiles:
        return True
    return False

def classify_polyccco(smiles):
    patt = Chem.MolFromSmiles('CC(C)=O')
    m = Chem.MolFromSmiles(smiles)
    flag = m.HasSubstructMatch(patt)
    pattern = re.compile(r'(C(\((F|(Cl)|(I)|(Br))\)){0,2})+')
    if flag and pattern.search(smiles) is not None:
        return True
    return False

def classify_polycooh(smiles):
    patt1 = Chem.MolFromSmiles('COC(C)=O')
    patt2 = Chem.MolFromSmiles('CC(=O)O')
    patt3 = Chem.MolFromSmiles('COC(=O)O')
    patt4 = Chem.MolFromSmiles('COOC')
    m = Chem.MolFromSmiles(smiles)
    flag1 = m.HasSubstructMatch(patt1)
    flag2 = m.HasSubstructMatch(patt2)
    flag3 = m.HasSubstructMatch(patt3)
    flag4 = m.HasSubstructMatch(patt4)
    pattern = re.compile(r'(C(\((F|(Cl)|(I)|(Br))\)){0,2})+')
    if (flag1 or flag2 or flag3 or flag4) and pattern.search(smiles) is not None:
        return True
    return False


def classify_polysoo(smiles):
    patt1 = Chem.MolFromSmiles('CS(=O)(=O)Cl')
    patt2 = Chem.MolFromSmiles('CS(=O)(=O)F')
    m = Chem.MolFromSmiles(smiles)
    flag1 = m.HasSubstructMatch(patt1)
    flag2 = m.HasSubstructMatch(patt2)
    pattern = re.compile(r'(C(\((F|(Cl)|(I)|(Br))\)){0,2})+')
    if (flag1 or flag2) and pattern.search(smiles) is not None:
        return True
    return False

def classify_polyosoo(smiles):
    patt = Chem.MolFromSmiles('COS(C)(=O)=O')
    m = Chem.MolFromSmiles(smiles)
    flag = m.HasSubstructMatch(patt)
    pattern = re.compile(r'(C(\((F|(Cl)|(I)|(Br))\)){0,2})+')
    if flag and pattern.search(smiles) is not None:
        return True
    return False


def classify_hfccc(smiles):
    x = count_Atom(smiles)
    if x != -1:
        num_f = x['F']
        num_c = x['C']
        num_h = x['H']
        num_cl = x['Cl']
        num_br = x['Br']
        num_i = x['I']
        num_atoms_auto = x['nums']
        if num_c + num_f + num_h + num_cl + num_br + num_i == num_atoms_auto \
            and num_c + 1 - (num_f + num_cl + num_br + num_i + num_h) / 2 == 0 \
            and (num_h >= 1 or num_cl >= 1 or num_br >= 1 or num_i >= 1) \
            and '#' not in smiles \
            and '=' not in smiles:
                return True
    return False

def classify_polyesas(smiles):
    x = count_Atom(smiles)
    if x != -1:
        num_f = x['F']
        num_c = x['C']
        num_h = x['H']
        num_o = x['O']
        num_s = x['S']
        num_br = x['Br']
        num_i = x['I']
        num_cl = x['Cl']
        num_atoms_auto = x['nums']
        if (num_c + num_f + num_h + num_cl + num_br + num_i + num_o + num_s == num_atoms_auto
                and num_h + num_cl + num_br + num_i + num_f == num_c + 1
                and num_o > 3
                and num_s == 1
                and smiles.startswith('O=S(=O)(O)')):
            return True
    return False

def classify_soclco(smiles):
    if 'O=S(=O)(Cl)' in smiles:
        smiles = smiles.replace('O=S(=O)(Cl)', 'F')
        x = count_Atom(smiles)
        if x != -1:
            num_f = x['F']
            num_c = x['C']
            num_h = x['H']
            num_br = x['Br']
            num_i = x['I']
            num_cl = x['Cl']
            num_atoms_auto = x['nums']
            if (num_c + num_f + num_h + num_cl + num_br + num_i == num_atoms_auto) \
                and (num_h + num_cl + num_br + num_i + num_f == 2 * num_c + 2) \
                and (num_c + 1 - (num_f + num_h + num_cl + num_br + num_i) / 2 == 0):
                return True
    return False







def classifying_polyfluoroakly(smiles):
    if classify_ccffhbr(smiles):
        return ['PFAA precursors', 'PolyFAenes']   # weichuxian
    if classify_ployfcas(smiles):
        return ['Polyfluoroalkyl acids', 'PolyFCAs']  # yi duan suan
    if classify_polyfecas(smiles):
        return ['Polyfluoroalkyl acids', 'PolyFECAs']  # mi + suan     PolyFECAs
    if classify_polyfesas(smiles):
        return ['Polyfluoroalkyl acids', 'PolyFESAs']   #
    if classify_polyosoo(smiles):
        return ['Polyfluoroalkyl acids', 'PolyFSA derivatives']   # huang suan zhi
    if classify_soclco(smiles):
        return ['PFAA precursors', 'Sulfonyl chloride']

    #if classify_polysoo(smiles):
     #   return ['Polyfluoroalkyl acids', 'Sulfonyl-halide derivatives']  # huang xian lu
    if classify_ooff(smiles):
        return ['Polyfluoroalkyl acids', 'PolyFCA derivatives']  #  zhi
    if classify_oclfff(smiles):
        return ['PFAA precursors', 'Acid chloride']  # xian cl  COCl
    if classify_misuan(smiles):
        return ['Polyfluoroalkyl acids', 'PolyFCA derivatives']
    if classify_polyoh(smiles):
        return ['PFAA precursors', 'PolyFACs']
    if classify_polyohderi(smiles):
        return ['PFAA precursors', 'PolyFAC derivatives']
    if classify_polycooh(smiles):
        return ['Polyfluoroalkyl acids', 'PolyFCA derivatives']     # zhi
    if classify_polyccco(smiles):
        return ['PFAA precursors', 'PFAK derivatives']   # tong
    return None