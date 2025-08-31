import re
#from .classify_others import classify_fcooh
from rdkit import Chem
from .atom_count import count_Atom

def classify_ffooc(smiles):
    patt = Chem.MolFromSmiles('CC(F)(F)C(=O)Oc1ccccc1')
    m = Chem.MolFromSmiles(smiles)
    flag = m.HasSubstructMatch(patt)
    pattern1 = re.compile(r'COC\(=O\)c1ccc\((C\(F\)\(F\))+F\)cc1')
    pattern2 = re.compile(r'COC\(=O\)c1cccc\((C\(F\)\(F\))+F\)c1')
    if flag or (pattern1.search(smiles) is not None) or (pattern2.search(smiles) is not None):
        return True
    return False


def classify_oso(smiles):
    patt = Chem.MolFromSmiles('CC(F)(F)S(=O)(=O)c1ccccc1')
    m = Chem.MolFromSmiles(smiles)
    flag = m.HasSubstructMatch(patt)
    if flag:
        return True
    return False


def classify_cscco(smiles):
    patt = Chem.MolFromSmiles('CSc1c(C)ccc(C(=O)O)c1C')
    m = Chem.MolFromSmiles(smiles)
    flag = m.HasSubstructMatch(patt)
    pattern = re.compile(r'c\(C\(F\)\(C\(F\)\(F\)F\)C\(F\)\(F\)F\)c')
    if flag and pattern.search(smiles) is not None:
        return True
    return False

def classify_cfcoo(smiles):
    pattern = re.compile(r'O=C\(O\)c1ccc\((C\(F\)\(F\))+F\)cc1')
    if pattern.search(smiles) is not None:
        return True
    return False


def classify_oosof(smiles):
    pattern1 = re.compile(r'Cc1ccc\(S\(=O\)\(=O\)OC+(C\(F\)\(F\))+F\)cc1')
    pattern2 = re.compile(r'Cc1ccc\(S\(=O\)\(=O\)OC(C\(F\)\(F\))+C\(F\)F\)cc1')
    pattern3 = re.compile(r'Cc1ccc\(S\(=O\)\(=O\)OC+(C\(F\)\(F\))+COS\(=O\)\(=O\)c2ccc\(C\)cc2\)cc1')
    if (pattern1.search(smiles) is not None) or (pattern2.search(smiles)is not None) or (pattern3.search(smiles) is not None):
        return True
    return False




def classify_ffoncc(smiles):
    patt = Chem.MolFromSmarts('CC(F)(F)C(=O)Nc1ccccc1')
    m = Chem.MolFromSmiles(smiles)
    flag = m.HasSubstructMatch(patt)
    pattern1 = re.compile(r'Cc1cc\(C\(F\)\((C\(F\)\(F\)F\))+cc\(C\)c1NC\(=O\)c1c+')
    pattern2 = re.compile(r'c2c+')
    if flag or ((pattern1.search(smiles) is not None) and (pattern2.search(smiles) is not None)):
        return True
    return False


def classify_coffoc(smiles):
    patt1 = Chem.MolFromSmarts('O=C(CF)c1ccccc1')
    patt2 = Chem.MolFromSmiles('CC(F)(F)C(=O)CC(=O)c1ccccc1')
    m = Chem.MolFromSmiles(smiles)
    flag1 = m.HasSubstructMatch(patt1)
    flag2 = m.HasSubstructMatch(patt2)
    if flag1 or flag2:
        return True
    return False


def classify_sooocff(smiles):
    pattern1 = re.compile(r'c1c(.*?)OS\(=O\)\(=O\)(C\(F\)\(F\))+F')
    pattern2 = re.compile(r'O=S\(=O\)\(Oc1c.*?\)(C\(F\)\(F\))+F')
    if (pattern1.search(smiles) is not None) or (pattern2.search(smiles) is not None):
        return True
    return False

def classify_ffccfc(smiles):
    patt = Chem.MolFromSmiles('c1ccc2c(c1)CCC2')
    m = Chem.MolFromSmiles(smiles)
    flag = m.HasSubstructMatch(patt)
    if flag:
        return True
    return False

def classify_fccfoc(smiles):
    patt = Chem.MolFromSmiles('C/C(F)=C(/F)Oc1ccccc1')
    m = Chem.MolFromSmiles(smiles)
    flag = m.HasSubstructMatch(patt)
    pattern1 = re.compile(r'=C\(F\)(C\(F\)\(F\))+F')
    pattern2 = re.compile(r'=C\(\\F\)(C\(F\)\(F\))+F')
    if flag and ((pattern1.search(smiles) is not None) or (pattern2.search(smiles) is not None)):
        return True
    return False




def classify_hoofs(smiles):
    patt = Chem.MolFromSmiles('CC(=O)CC(=O)c1cccs1')
    m = Chem.MolFromSmiles(smiles)
    flag = m.HasSubstructMatch(patt)
    pattern = re.compile(r'C\(=O\)(C\(F\)\(F\))+')
    if flag and pattern.search(smiles) is not None:
        return True
    return False

def classify_osonc(smiles):
    patt = Chem.MolFromSmiles('CNC(=O)C(F)F')
    m = Chem.MolFromSmiles(smiles)
    flag = m.HasSubstructMatch(patt)
    if flag:
        return True
    return False

def classify_cfccc(smiles):
    patt = Chem.MolFromSmiles('FC(F)(c1ccccc1)C(F)(F)c1ccccc1')
    m = Chem.MolFromSmiles(smiles)
    flag = m.HasSubstructMatch(patt)
    if flag:
        return True
    return False

def classify_occof(smiles):
    patt1 = Chem.MolFromSmiles('FC(F)COc1ccccc1')
    patt2 = Chem.MolFromSmiles('FC(F)Oc1ccccc1')
    patt3 = Chem.MolFromSmiles('CC(F)(F)CCCOc1ccccc1')
    m = Chem.MolFromSmiles(smiles)
    flag1 = m.HasSubstructMatch(patt1)
    flag2 = m.HasSubstructMatch(patt2)
    flag3 = m.HasSubstructMatch(patt3)
    if flag1 or flag2 or flag3:
        return True
    return False

def classify_cnff(smiles):
    patt = Chem.MolFromSmiles('CC(F)c1ccccn1')
    m = Chem.MolFromSmiles(smiles)
    flag = m.HasSubstructMatch(patt)
    if flag:
        return True
    return False


def classify_fnnh(smiles):
    patt = Chem.MolFromSmiles('CC(F)(F)c1cc[nH]n1')
    m = Chem.MolFromSmiles(smiles)
    flag = m.HasSubstructMatch(patt)
    if flag:
        return True
    return False


def classify_ppfpff(smiles):
    patt = Chem.MolFromSmiles('c1ccc(P(c2ccccc2)c2ccccc2)cc1')
    m = Chem.MolFromSmiles(smiles)
    flag = m.HasSubstructMatch(patt)
    if flag:
        return True
    return False




def classify_nff(smiles):
    patt = Chem.MolFromSmiles('CC(F)(F)c1ccc[nH]1')
    m = Chem.MolFromSmiles(smiles)
    flag = m.HasSubstructMatch(patt)
    if flag:
        return True
    return False

def classify_cccsf(smiles):
    patt = Chem.MolFromSmiles('CC(F)(F)c1cccs1')
    m = Chem.MolFromSmiles(smiles)
    flag = m.HasSubstructMatch(patt)
    if flag:
        return True
    return False

def classify_nncff(smiles):
    patt = Chem.MolFromSmiles('CC(F)(F)c1cnc[nH]1')
    m = Chem.MolFromSmiles(smiles)
    flag = m.HasSubstructMatch(patt)
    if flag:
        return True
    return False

def classify_cffcn(smiles):
    patt = Chem.MolFromSmiles('CC(F)(F)CNc1ccccc1')
    m = Chem.MolFromSmiles(smiles)
    flag = m.HasSubstructMatch(patt)
    if flag:
        return True
    return False

def classify_cccnnf(smiles):
    patt1 = Chem.MolFromSmiles('CC(F)(F)c1nc(C(C)(F)F)nc(C(C)(F)F)n1')
    patt2 = Chem.MolFromSmiles('CC(F)(F)COc1ncncn1')
    m = Chem.MolFromSmiles(smiles)
    flag1 = m.HasSubstructMatch(patt1)
    flag2 = m.HasSubstructMatch(patt2)
    if flag1 or flag2:
        return True
    return False


def classify_fsoon(smiles):
    patt1 = Chem.MolFromSmiles('CC(F)(F)S(=O)(=O)Nc1ccccc1')
    patt2 = Chem.MolFromSmiles('CN(CCOC(N)=O)S(=O)(=O)C(C)(F)F')
    patt3 = Chem.MolFromSmiles('CO/C(O)=N/c1cccc(/N=C(\O)OCCCCN(C)S(=O)(=O)C(C)(F)F)c1')
    patt4 = Chem.MolFromSmarts('CC(F)(F)S(=O)(=O)NCc1ccccc1')
    m = Chem.MolFromSmiles(smiles)
    flag1 = m.HasSubstructMatch(patt1)
    flag2 = m.HasSubstructMatch(patt2)
    flag3 = m.HasSubstructMatch(patt3)
    flag4 = m.HasSubstructMatch(patt4)
    if flag1 or flag2 or flag3 or flag4:
        return True
    return False


def classify_fnnfc(smiles):
    patt = Chem.MolFromSmiles('CC(F)(F)c1nc2ccccc2[nH]1')
    m = Chem.MolFromSmiles(smiles)
    flag = m.HasSubstructMatch(patt)
    if flag:
        return True
    return False



def classify_fonon(smiles):
    patt = Chem.MolFromSmiles('CC(F)(F)c1c[nH]c(=O)[nH]c1=O')
    m = Chem.MolFromSmiles(smiles)
    flag = m.HasSubstructMatch(patt)
    if flag:
        return True
    return False

def classify_scf(smiles):
    patt = Chem.MolFromSmiles('CC(F)(F)Sc1ccccc1')
    m = Chem.MolFromSmiles(smiles)
    flag = m.HasSubstructMatch(patt)
    if flag:
        return True
    return False




def classify_fffc(smiles):
    x = count_Atom(smiles)
    if x != -1:
        num_c = x['C']
        num_h = x['H']
        num_f = x['F']
        num_atoms_auto = x['nums']
        patt = Chem.MolFromSmiles('FC(F)(F)C(F)(c1ccccc1)C(F)(F)F')
        m = Chem.MolFromSmiles(smiles)
        flag = m.HasSubstructMatch(patt)
        if flag or (num_c + num_h + num_f == num_atoms_auto and num_c + 1 - (num_f + num_h) / 2 == 4):
            return True
    return False


def classify_nnoffc(smiles):
    patt = Chem.MolFromSmiles('CC(F)(F)C(=O)N/N=C/c1ccccc1')
    m = Chem.MolFromSmiles(smiles)
    flag = m.HasSubstructMatch(patt)
    if flag:
        return True
    return False




def classify_cnof(smiles):
    patt = Chem.MolFromSmiles('O=C(Nc1ccccc1)C1(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C1(F)F')
    m = Chem.MolFromSmiles(smiles)
    flag = m.HasSubstructMatch(patt)
    if flag:
        return True
    return False

def classify_sscf(smiles):
    patt = Chem.MolFromSmiles('FC1(F)C(c2ccsc2)=C(c2ccsc2)C(F)(F)C1(F)F')
    m = Chem.MolFromSmiles(smiles)
    flag = m.HasSubstructMatch(patt)
    if flag:
        return True
    return False

def classify_foccnno(smiles):
    patt = Chem.MolFromSmiles('C/C(=C\C(=O)C(C)(F)F)NNC(=O)c1ccccc1')
    m = Chem.MolFromSmiles(smiles)
    flag = m.HasSubstructMatch(patt)
    if flag:
        return True
    return False


def classifying_side_chain(smiles):
    if classify_ppfpff(smiles):
        return ['other PFASs', 'side-chain fluorinated aromatics']      # Trip-phos
    if classify_oso(smiles):
        return ['other PFASs', 'side-chain fluorinated aromatics']    # PF-sulf-benz
    if classify_nncff(smiles):
        return ['other PFASs', 'side-chain fluorinated aromatics']         # Imidazole
    if classify_hoofs(smiles):
        return ['other PFASs', 'side-chain fluorinated aromatics']      # Thiop-dione


    if classify_cccsf(smiles):
        return ['other PFASs', 'side-chain fluorinated aromatics']     # PF-thiop
    if classify_sooocff(smiles):
        return ['other PFASs', 'side-chain fluorinated aromatics']    # Sulf
    if classify_osonc(smiles):
        return ['other PFASs', 'side-chain fluorinated aromatics']    # Amido-pyra
    if classify_coffoc(smiles):
        return ['other PFASs', 'side-chain fluorinated aromatics']     # Benz-one
    if classify_cscco(smiles):
        return ['other PFASs', 'side-chain fluorinated aromatics']   # Phenyl-thioethen

    if classify_cfcoo(smiles):
        return ['other PFASs', 'side-chain fluorinated aromatics']    # Benz-acid
    if classify_oosof(smiles):
        return ['other PFASs', 'side-chain fluorinated aromatics']    # Benz-sulf
    if classify_ffoncc(smiles):
        return ['other PFASs', 'side-chain fluorinated aromatics']   # Phenyl-amide
    if classify_ffooc(smiles):
        return ['other PFASs', 'side-chain fluorinated aromatics']  # Phenyl-ester

    if classify_ffccfc(smiles):
        return ['other PFASs', 'side-chain fluorinated aromatics']  # Flu-dihy-indene
    if classify_fccfoc(smiles):
        return ['other PFASs', 'side-chain fluorinated aromatics']   # Eoxy-benz
    if classify_cfccc(smiles):
        return ['other PFASs', 'side-chain fluorinated aromatics']  # Diphenylethane
    if classify_cnff(smiles):
        return ['other PFASs', 'side-chain fluorinated aromatics']  # 吡啶  PF-pyridine


    if classify_fnnh(smiles):
        return ['other PFASs', 'side-chain fluorinated aromatics']   # 吡唑  PF-pyrazole
    if classify_cffcn(smiles):
        return ['other PFASs', 'side-chain fluorinated aromatics']  # PF N-benze
    if classify_nff(smiles):
        return ['other PFASs', 'side-chain fluorinated aromatics']  # 吡咯 PF-pyrrole
    if classify_cccnnf(smiles):
        return ['other PFASs', 'side-chain fluorinated aromatics']  # 三嗪  PF-triazine

    if classify_fsoon(smiles):
        return ['other PFASs', 'side-chain fluorinated aromatics']  # 苄基磺酰胺 Benzyl-sulfamide
    if classify_fnnfc(smiles):
        return ['other PFASs', 'side-chain fluorinated aromatics']  # 苯并咪唑  Benzo-imida
    if classify_fonon(smiles):
        return ['other PFASs', 'side-chain fluorinated aromatics']  # mi ding  Pyrimidine
    if classify_nnoffc(smiles):
        return ['other PFASs', 'side-chain fluorinated aromatics']   # xian jing    Hydrazide


    if classify_cnof(smiles):
        return ['other PFASs', 'side-chain fluorinated aromatics']   # xian an   PF-carboxamide
    if classify_sscf(smiles):
        return ['other PFASs', 'side-chain fluorinated aromatics']   # shuang sai fen    Bis-thiophene
    if classify_foccnno(smiles):
        return ['other PFASs', 'side-chain fluorinated aromatics']  # ben bing xian jing    Benzo-hydrazide
    if classify_fffc(smiles):
        return ['other PFASs', 'side-chain fluorinated aromatics']     # PF-benz
    if classify_scf(smiles):
        return ['other PFASs', 'side-chain fluorinated aromatics']  # 磺烷  Phenyl-sulf
    if classify_occof(smiles):
        return ['other PFASs', 'side-chain fluorinated aromatics']  # mi   Pyenyl-eth