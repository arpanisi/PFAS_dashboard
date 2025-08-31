from rdkit import Chem
from .atom_count import count_Atom
#from .classify_other_perfluoroalkyls import classify_pfas
import re
from rdkit import Chem
from .atom_count import count_Atom, standard_mol
import re
# PASFs



def classify_pfas(smiles):
    x = count_Atom(smiles)
    if x != -1:
        num_f = x['F']
        num_c = x['C']
        num_atoms_auto = x['nums']
        if num_atoms_auto == num_c + num_f and num_c * 2 + 2 == num_f:
            return True
    return False


def classify_pasfs(smiles):
    patt2 = Chem.MolFromSmiles('O=S(F)(C)=O')
    m = Chem.MolFromSmiles(smiles)
    flag = m.HasSubstructMatch(patt2)
    x = count_Atom(smiles)
    if x != -1:
        num_f = x['F']
        num_c = x['C']
        num_o = x['O']
        num_s = x['S']
        num_atoms_auto = x['nums']
        if num_c * 2 + 2 == num_f \
                and 'S1' not in smiles \
                and num_o == 2 \
                and num_s == 1 \
                and flag \
                and num_c + num_f + num_s + num_o == num_atoms_auto \
                and not smiles.startswith('O=S1(=O)')  \
                and (smiles.startswith("O=S(=O)(F)") or smiles.startswith("O=S(F)(=O)")):
            return True
    return False


def classify_fasas(smiles):
    x = count_Atom(smiles)
    if x != -1:
        num_f = x['F']
        num_c = x['C']
        num_o = x['O']
        num_s = x['S']
        num_h = x['H']
        num_n = x['N']
        num_atoms_auto = x['nums']
        if num_c * 2 + 1 == num_f \
                and num_h == 2 \
                and num_o == 2 \
                and num_n == 1 \
                and num_s == 1 \
                and num_c + num_f + num_s + num_o + num_n + num_h == num_atoms_auto \
                and smiles.startswith('NS(=O)(=O)'):
            return True
    return False



def classify_pafs(smiles):
    patt2 = Chem.MolFromSmiles('O=C(C)F')
    m = Chem.MolFromSmiles(smiles)
    flag = m.HasSubstructMatch(patt2)
    x = count_Atom(smiles)
    if x != -1:
        num_f = x['F']
        num_c = x['C']
        num_o = x['O']
        num_atoms_auto = x['nums']
        if num_c * 2 == num_f \
                and num_o == 1 \
                and num_c + num_f + num_o == num_atoms_auto \
                and flag \
                and smiles.startswith('O=C(F)'):
            return True
    return False




def classify_pfais(smiles):
    x = count_Atom(smiles)
    if x != -1:
        num_f = x['F']
        num_c = x['C']
        num_i = x['I']
        num_atoms_auto = x['nums']
        if num_c * 2 + 1 == num_f \
                and num_i == 1 \
                and num_c + num_f + num_i == num_atoms_auto \
                and (smiles.startswith("I") or '(I)' in smiles or smiles.endswith("I")):
            return True
    return False



def classify_pfals(smiles):
    patt = Chem.MolFromSmiles('CC(C)=O')
    m = Chem.MolFromSmiles(smiles)
    flag = m.HasSubstructMatch(patt)
    x = count_Atom(smiles)
    if x != -1:
        num_f = x['F']
        num_c = x['C']
        num_h = x['H']
        num_o = x['O']
        num_atoms_auto = x['nums']
        if (num_c - 1) * 2 + 1 == num_f \
                and num_h == 1 \
                and num_o == 1 \
                and num_c + num_f + num_o + num_h == num_atoms_auto \
                and smiles.startswith("O=C") \
                and not (smiles.startswith("O=C(F)")) \
                and not flag:
            return True
    return False


# xinjia


def classify_sfas(smiles):
    pattern = re.compile(r'^C+(C\(F\)\(F\))+F$')
    x = count_Atom(smiles)
    if x != -1:
        num_f = x['F']
        num_c = x['C']
        num_h = x['H']
        num_atoms_auto = x['nums']
        if num_c + num_f + num_h == num_atoms_auto \
            and num_c + 1 - (num_f + num_h) / 2 == 0 \
            and num_h == num_f \
            and pattern.search(smiles) is not None:
            return True
    return False

def classify_hfcs(smiles):
    x = count_Atom(smiles)
    if x != -1:
        num_f = x['F']
        num_c = x['C']
        num_h = x['H']
        num_atoms_auto = x['nums']
        if num_c + num_f + num_h == num_atoms_auto \
            and num_h >= 1 \
            and num_c + 1 - (num_f + num_h) / 2 == 0:
            return True
    return False



def classify_hfes(smiles):
    patt1 = Chem.MolFromSmiles('COC')
    m = Chem.MolFromSmiles(smiles)
    flag1 = m.HasSubstructMatch(patt1)
    x = count_Atom(smiles)
    if x != -1:
        num_f = x['F']
        num_c = x['C']
        num_h = x['H']
        num_o = x['O']
        num_atoms_auto = x['nums']
        if flag1 and not (smiles.startswith('OC')) and not ('C(O)' in smiles) \
            and num_c + num_o + num_f + num_h == num_atoms_auto \
            and num_o >= 1 \
            and num_c + 1 - (num_f + num_h) / 2 == 0:
            return True
    return False

def classify_hfos(smiles):
    if smiles.startswith("C=CC"):
        smiles = smiles.replace("C=C", "F", 1)
        if classify_pfas(smiles):
            return True
    return False

def classify_semiketons(smiles):
    x = count_Atom(smiles)
    if x != -1:
        num_f = x['F']
        num_c = x['C']
        num_h = x['H']
        num_o = x['O']
        num_atoms_auto = x['nums']
        pattern1 = re.compile(r'^C+C\(=O\)(C\(F\)\(F\))+F')  # an f ban h
        pattern2 = re.compile(r'O=C\((C\(F\)\(F\))+F\)(C\(F\)\(F\))+F')
        if pattern1.search(smiles) is not None \
            and pattern2.search(smiles) is not None \
            and num_c + num_h + num_f + num_o == num_atoms_auto  \
            and num_o == 1:
            return True
    return False




def classify_soo(smiles):
    pattern1 = re.compile(r'.*N.*S\(=O\)\(=O\)(C\(F\)\(F\))+F')
    pattern2 = re.compile(r'O=S\(=O\).*N.*(C\(F\)\(F\))+F')
    if (pattern1.search(smiles) is not None) or (pattern2.search(smiles) is not None):
        return True
    return False





def classify_paecfs(smiles):
    if 'O=C(F)' in smiles:
        smiles = smiles.replace('O=C(F)', 'F')
        x = count_Atom(smiles)
        if x != -1:
            num_f = x['F']
            num_c = x['C']
            num_o = x['O']
            num_atoms_auto = x['nums']
            if num_f + num_c + num_o == num_atoms_auto \
                and num_c + 1 - num_f / 2 == 0:
                return True
    return False



def classify_fcooooc(smiles):
    pattern = re.compile(r'O=C1C+C1(C\(F\)\(F\))+F')
    if pattern.search(smiles) is not None:
        return True
    return False


def classify_ccffh2(smiles):
    x = count_Atom(smiles)
    if x != -1:
        num_f = x['F']
        num_c = x['C']
        num_h = x['H']
        num_atoms_auto = x['nums']
        if num_c + num_f + num_h == num_atoms_auto and "C1" in smiles \
            and num_c + 1 - (num_f + num_h) / 2 == 1:
            return True
        return False


def classify_duomi(smiles):
    patt1 = Chem.MolFromSmiles('COC')
    m = Chem.MolFromSmiles(smiles)
    flag1 = m.HasSubstructMatch(patt1)
    x = count_Atom(smiles)
    if x != -1:
        num_c = x['C']
        num_h = x['H']
        num_f = x['F']
        num_o = x['O']
        num_atoms_auto = x['nums']
        if flag1 \
            and 'Cl' not in smiles \
            and num_c + num_f + num_h + num_o == num_atoms_auto \
            and num_o >= 1 \
            and num_c + 1 - (num_f + num_h) / 2 == 0:
            return True
    return False

def classify_n1_ftohs(smiles):  # n:1 Fluorotelomer alcohols
    if smiles.startswith("OCC"):
        smiles = smiles.replace("OC", "I")
        if classify_pfais(smiles):
            return True
    return False




def classify_perfluoroalkenes(smiles):
    x = count_Atom(smiles)
    if x != -1:
        num_f = x['F']
        num_c = x['C']
        num_atoms_auto = x['nums']
        if num_c * 2 == num_f \
            and num_c + num_f == num_atoms_auto \
            and num_c > 2 \
            and '=' in smiles \
            and num_c + 1 - num_f / 2 == 1:
            return True
    return False

def classify_ccfcc(smiles):
    pattern = re.compile(r'C=C+(C\(F\)\(F\))+C+=C')
    if pattern.search(smiles) is not None:
        return True
    return False



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

def classify_pfaks(smiles):
    patt1 = Chem.MolFromSmiles('O=C(C)F')
    patt2 = Chem.MolFromSmiles('O=C(C)C')
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
        #pattern = re.compile(r'O=C\((C\(F\)\(F\))+F\)(C\(F\)\(F\))+F')
        if num_c * 2 == num_f + num_h \
            and num_c + 1 - (num_f + num_h) / 2 == 1 \
            and num_atoms_auto == num_c + num_f + num_o + num_h \
            and num_o == 1 \
            and not flag1 \
            and flag2 \
            and '=' in smiles \
            and smiles.startswith('O=C'):
            return True
    return False

def classify_fof(smiles):
    pattern = re.compile(r'O=C\(CC\(=O\)(C\(F\)\(F\))+F\)(C\(F\)\(F\))+F')
    if pattern.search(smiles) is not None:
        return True
    return False


def classify_ffcfo(smiles):
    patt1 = Chem.MolFromSmiles('O=C(C)O')
    patt2 = Chem.MolFromSmiles('O=C(C)OC')
    m = Chem.MolFromSmiles(smiles)
    flag1 = m.HasSubstructMatch(patt1)
    flag2 = m.HasSubstructMatch(patt2)
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
        num_i = x['I']
        num_atoms_auto = x['nums']
        if num_c + num_f + num_h + num_o + num_br + num_cl + num_i == num_atoms_auto \
            and '=' in smiles \
            and not flag1 \
            and not flag2 \
            and ((pattern1.search(smiles) is not None) or (pattern2.search(smiles) is not None) or (pattern3.search(smiles) is not None)):
            return True
    return False


def classify_occco(smiles):
    pattern1 = re.compile('C\(=O\)CC\(=O\)(C\(F\)\(F\))+')
    pattern2 = re.compile(r'O=C1C+C1C\(=O\)(C\(F\)\(F\))+F')
    if (pattern1.search(smiles) is not None) or (pattern2.search(smiles) is not None):
        return True
    return False

def classify_fccoff(smiles):
    pattern1 = re.compile('C\(=O\)CC\(=O\)(C\(F\)\(F\))+')   # di ketons
    pattern2 = re.compile(r'O=C1C+C1C\(=O\)(C\(F\)\(F\))+F')  # di ketons
    pattern3 = re.compile(r'O=C\(\/C\(F\)=C\(\/F\)C\(F\)\((C\(F\)\(F\)F\))+(C\(F\)\(F\))+F')
    pattern4 = re.compile(r'O=C\(C\(F\)=C\(F\)C\(F\)\((C\(F\)\(F\)F\))+(C\(F\)\(F\))+F')
    pattern5 = re.compile(r'O=C\(\/C\(F\)=C\(\\F\)C\(F\)\((C\(F\)\(F\)F\))+(C\(F\)\(F\))+F')
    if (pattern1.search(smiles) is not None) or (pattern2.search(smiles) is not None) or (
            pattern3.search(smiles) is not None) or (pattern4.search(smiles) is not None) or (pattern5.search(smiles) is not None):
        return True
    return False



def determine_alkyl_chain(smiles):
    c_set = set(list(smiles))
    return len(c_set) == 1 and 'C' in c_set


def classify_alkyl_fasas(smiles):  # N-alkyl perfluoroalkane sulfonamides
    if 'NS(=O)(=O)' in smiles:
        split_smiles = smiles.split("NS(=O)(=O)", 1)   # Separate alkyl chain from PASF chains
        alkyl_chain = split_smiles[0]
        fasa_chain = "NS(=O)(=O)" + split_smiles[1]
        if determine_alkyl_chain(alkyl_chain) and classify_fasas(fasa_chain):
            return True
    return False


def classify_alkyl_fases(smiles):  # (N-alkyl) perfluoroalkane sulfonamidoethanols
    if "N(CCO)S(=O)(=O)" in smiles:
        split_smiles = smiles.split("N(CCO)S(=O)(=O)", 1)
        alkyl_chain = split_smiles[0]
        fasa_chain = "NS(=O)(=O)" + split_smiles[1]
        if determine_alkyl_chain(alkyl_chain) and classify_fasas(fasa_chain):
            return True
    elif smiles.startswith("O=S(=O)(NCCO)"):
        split_smiles = smiles.split("O=S(=O)(NCCO)", 1)
        fasa_chain = "NS(=O)(=O)" + split_smiles[1]
        if classify_fasas(fasa_chain):
            return True
    return False


def classify_alkyl_fasacs(smiles):  # N-Alkyl perfluoroalkane sulfonamidoethyl acrylates
    if smiles.startswith("C=CC(=O)OCCN") and "S(=O)(=O)" in smiles:
        split_smiles = smiles.split("C=CC(=O)OCCN", 1)
        new_split_smiles = split_smiles[1].split("S(=O)(=O)", 1)
        alkyl_chain = new_split_smiles[0].strip("()")
        if alkyl_chain != new_split_smiles[0]:   # to prevent C=CC(=O)OCCNCCCS(=O)(=O)
            fasa_chain = "NS(=O)(=O)" + new_split_smiles[1]
            if determine_alkyl_chain(alkyl_chain) and classify_fasas(fasa_chain):
                return True
    return False


def classify_alkyl_fasmacs(smiles):  # N-Alkyl perfluoroalkane sulfonamidoethyl methacrylates
    if smiles.startswith("C=C(C)C(=O)OCCN") and "S(=O)(=O)" in smiles:
        split_smiles = smiles.split("C=C(C)C(=O)OCCN", 1)
        new_split_smiles = split_smiles[1].split("S(=O)(=O)", 1)
        alkyl_chain = new_split_smiles[0].strip("()")
        if alkyl_chain != new_split_smiles[0]:
            fasa_chain = "NS(=O)(=O)" + new_split_smiles[1]
            if determine_alkyl_chain(alkyl_chain) and classify_fasas(fasa_chain):
                return True
    return False


def classify_alkyl_fasaas(smiles):   # N-alkyl perfluoroalkane sulfonamido- acetic acids
    if "N(CC(=O)O)S(=O)(=O)" in smiles:
        split_smiles = smiles.split("N(CC(=O)O)S(=O)(=O)", 1)
        alkyl_chain = split_smiles[0]
        fasa_chain = "NS(=O)(=O)" + split_smiles[1]
        if determine_alkyl_chain(alkyl_chain) and classify_fasas(fasa_chain):
            return True
    elif smiles.startswith("O=C(O)CNS(=O)(=O)"):
        split_smiles = smiles.split("O=C(O)CNS(=O)(=O)", 1)
        fasa_chain = "NS(=O)(=O)" + split_smiles[1]
        if classify_fasas(fasa_chain):
            return True
    return False

def classify_soo(smiles):
    pattern1 = re.compile(r'.*N.*S\(=O\)\(=O\)(C\(F\)\(F\))+F')
    pattern2 = re.compile(r'O=S\(=O\).*N.*(C\(F\)\(F\))+F')
    if (pattern1.search(smiles) is not None) or (pattern2.search(smiles) is not None):
        return True
    return False


def classify_n4(smiles):
    patt2 = Chem.MolFromSmiles('NS(=O)(C)=O')
    m = Chem.MolFromSmiles(smiles)
    flag = m.HasSubstructMatch(patt2)
    if flag:
        return True
    return False

def classify_perfluoroalkanes(smiles):
    x = count_Atom(smiles)
    if x != -1:
        num_f = x['F']
        num_c = x['C']
        num_atoms_auto = x['nums']
        if num_c + num_f == num_atoms_auto \
                and num_c * 2 + 2 == num_f \
                and num_c + 1 - num_f / 2 == 0:
                return True
    return False



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

def classify_amidederi(smiles):
    patt2 = Chem.MolFromSmiles('O=C(N)C(F)(C)F')
    m = Chem.MolFromSmiles(smiles)
    flag = m.HasSubstructMatch(patt2)
    if flag:
        return True
    return False

def classify_cfclbri(smiles):
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


def classifying_pfaa_precursors(pfas):
    if classify_perfluoroalkylethers(pfas):
        return ['Other PFASs', 'Perfluoroalkylethers']
    if classify_alkyl_fasas(pfas):
        return ['PFAA precursors', 'PASF-based substances']
    if classify_alkyl_fases(pfas):
        return ['PFAA precursors', 'PASF-based substances']
    if classify_alkyl_fasacs(pfas):
        return ['PFAA precursors', 'PASF-based substances']
    if classify_alkyl_fasmacs(pfas):
        return ['PFAA precursors', 'PASF-based substances']
    if classify_alkyl_fasaas(pfas):
        return ['PFAA precursors', 'PASF-based substances']
    if classify_soo(pfas):
        return ['PFAA precursors', 'PASF-based substances']
    if classify_pasfs(pfas):
        return ['PFAA precursors', 'PASFs']              # SO2
    if classify_fasas(pfas):
        return ['PFAA precursors', 'PASF-based substances']              # SO2NH2
    if classify_pafs(pfas):
        return ['PFAA precursors', 'PACFs']              # COF xian F
    if classify_pfais(pfas):
        return ['PFAA precursors', 'PFAIs']              # qian F I
    if classify_pfals(pfas):
        return ['PFAA precursors', 'PFALs']              # CHO quan         add
    if classify_sfas(pfas):
        return ['PFAA precursors', 'SFAs']               # ban F ban H
    if classify_hfcs(pfas):
        return ['PFAA precursors', 'HFCs']       #  F H hun he
    if classify_hfes(pfas):
        return ['PFAA precursors', 'HFEs']       # mi + hun
    if classify_hfos(pfas):
        return ['PFAA precursors', 'HFOs']               # 一duan shuang jian
    if classify_ccfcc(pfas):
        return ['PFAA precursors', 'HFOs']
    if classify_perfluoroalkenes(pfas):
        return ['PFAA precursors', 'PFAenes']
    if classify_paecfs(pfas):
        return ['PFAA precursors', 'PAECFs']
    if classify_pfacs(pfas):
        return ['PFAA precursors', 'PFACs']   # chun    perfluoroalkyl alcohols
    if classify_pfaks(pfas):
        return ['PFAA precursors', 'PFAKs']  # tong
    if classify_ffcfo(pfas):
        return ['PFAA precursors', 'PolyFEAenes']  # xi ji O
    if classify_amidederi(pfas):
        return ['Other PFASs', 'Amide derivatives']



    #if classify_fccoff(pfas):
     #   return ['PFAA precursors', 'PFAK derivatives']   # tong = xiting
    if classify_duomi(pfas):
        return ['PFAA precursors', 'PolyFEACs']
    if classify_perfluoroalkanes(pfas):
        return ['Other PFASs', 'Perfluoroalkanes']
    if classify_cfclbri(pfas):
        return ['Other PFASs', 'Polyfluoroalkanes']
    return None