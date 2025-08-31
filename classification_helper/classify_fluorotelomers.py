from rdkit import Chem
from .classify_pfaa_precursors import classify_pfais
from .atom_count import count_Atom
import re

def determine_alkyl_chain(smiles):
    c_set = set(list(smiles))
    return len(c_set) == 1 and 'C' in c_set


def classify_pfas(smiles):
    x = count_Atom(smiles)
    if x != -1:
        num_f = x['F']
        num_c = x['C']
        num_atoms_auto = x['nums']
        if num_atoms_auto == num_c + num_f and num_c * 2 + 2 == num_f:
            return True
    return False

# Fluorotelomer PFAA precursors
def classify_ftis(smiles):  # n:2 Fluorotelomer iodides
    if "CCI" in smiles:
        smiles = smiles.replace("CCI", "F", 1)   # Remove the CH2CH2
        if classify_pfas(smiles):
            return True
    return False

def classify_ftohs(smiles):  # n:2 Fluorotelomer alcohols
    if smiles.startswith("OC"):
        smiles = smiles.replace("O", "F", 1)
        if classify_pfas(smiles):
            return True
    return False




def classify_ftacs(smiles):  # n:2 Fluorotelomer acrylates
    if smiles.startswith("C=CC(=O)OCCC"):
        smiles = smiles.replace("C=CC(=O)OCC", "F", 1)
        if classify_pfas(smiles):
            return True
    return False



def classify_ftmacs(smiles):  # n:2 Fluorotelomer methacrylates
    if smiles.startswith("C=C(C)C(=O)OCCC"):
        smiles = smiles.replace("C=C(C)C(=O)OCC", "F", 1)
        if classify_pfas(smiles):
            return True
    return False



def classify_monoesters(smiles):  # n:2 Polyfluoroalkyl phosphoric acid esters, monoester 单脂
    if smiles.startswith("O=P(O)(O)OCCC"):
        smiles = smiles.replace("O=P(O)(O)OCC", "F", 1)
        if classify_pfas(smiles):
            return True
    return False

def classify_diesters(smiles):  # n:2 Polyfluoroalkyl phosphoric acid esters, diester
    if smiles.startswith("O=P(O)(OCCC"):
        smiles = smiles.replace("O=P(O)(OCC", "", 1)
        split_smiles = smiles.split(")OCC", 1)
        if len(set(split_smiles)) == 1:
            new_smiles = "F" + split_smiles[0]
            if classify_pfas(new_smiles):
                return True
    return False


def classify_ftals(smiles):  # n:2 Fluorotelomer aldehydes
    if smiles.startswith("O=CCC"):
        smiles = smiles.replace("O=CC", "F", 1)
        if classify_pfas(smiles):
            return True
    return False


def classify_ftuals(smiles):  # n:2 Fluorotelomer unsaturated aldehydes
    if smiles.startswith("O=C/C=C(\F)C"):
        smiles = smiles.replace("O=C/C=C(\F)", "F", 1)
        if classify_pfas(smiles):
            return True
    elif smiles.startswith("O=CC=C(F)C"):
        smiles = smiles.replace("O=CC=C(F)", "F", 1)
        if classify_pfas(smiles):
            return True
    return False


def classify_ftcas(smiles):  # n:2 Fluorotelomer carboxylic acids
    if smiles.startswith("O=C(O)CC"):
        smiles = smiles.replace("O=C(O)C", "F", 1)
        if classify_pfas(smiles):
            return True
    return False



def classify_ftucas(smiles):  # n:2 Fluorotelomer unsaturated carboxylic acids
    if smiles.startswith("O=C(O)C=C(F)C"):
        smiles = smiles.replace("O=C(O)C=C(F)", "F", 1)
        if classify_pfas(smiles):
            return True
    return False


def classify_ftsas(smiles):  # n:2 Fluorotelomer sulfonic acids
    if smiles.startswith("O=S(=O)(O)CCC"):
        smiles = smiles.replace("O=S(=O)(O)CC", "F", 1)
        if classify_pfas(smiles):
            return True
    return False



def classify_sfaenes(smiles):
    x = count_Atom(smiles)
    if x != -1:
        num_c = x['C']
        num_f = x['F']
        num_h = x['H']
        num_atoms_auto = x['nums']
        if num_c + num_f + num_h == num_atoms_auto and num_h != 0:
            if ("C=CC") in smiles:
                split_smiles = smiles.split("C=C", 1)
                if len(split_smiles) == 2:
                    alkyl_chain = split_smiles[0]
                    pfai_chain = "F" + split_smiles[1]
                    if determine_alkyl_chain(alkyl_chain) and classify_pfas(pfai_chain):
                        return True
            elif ("/C=C/C") in smiles:
                split_smiles = smiles.split("/C=C/", 1)
                if len(split_smiles) == 2:
                    alkyl_chain = split_smiles[0]
                    pfai_chain = "F" + split_smiles[1]
                    if determine_alkyl_chain(alkyl_chain) and classify_pfas(pfai_chain):
                        return True
    return False


def classify_three_acids(smiles):  # n:3 Fluorotelomer carboxylic acids
    if smiles.startswith("O=C(O)CCC"):
        smiles = smiles.replace("O=C(O)CC", "F", 1)
        if classify_pfas(smiles):
            return True
    return False





def classify_n1_ftohs(smiles):  # n:1 Fluorotelomer alcohols
    if smiles.startswith("OCC"):
        smiles = smiles.replace("OC", "I")
        if classify_pfais(smiles):
            return True
    return False



def classify_scfff(smiles):
    pattern = re.compile(r'.*SCC(C\(F\)\(F\))+F')
    if pattern.search(smiles) is not None:
        return True
    return False

def classify_fiif(smiles):
    pattern1 = re.compile(r'F(C\(F\)\(F\))+CC\(I\)C+\(I\)C(C\(F\)\(F\))+F')
    pattern2 = re.compile(r'F(C\(F\)\(F\))+C+(C\(F\)\(F\))+F')
    if (pattern1.search(smiles) is not None) or (pattern2.search(smiles) is not None):
        return True
    return False

def classify_ifcl(smiles):
    pattern1 = re.compile(r'.*C+\(I\)C(C\(F\)\(F\))+Cl')
    pattern2 = re.compile(r'FC\(F\)\(Cl\)(C\(F\)\(F\))+C+I')
    if (pattern1.search(smiles) is not None) or (pattern2.search(smiles) is not None):
        return True
    return False


def classify_poooh(smiles):
    patt = Chem.MolFromSmiles('CC(F)(F)CC(O)COP(=O)(O)O')
    m = Chem.MolFromSmiles(smiles)
    flag = m.HasSubstructMatch(patt)
    if flag:
        return True
    return False




def classify_chun(smiles):
    patt1 = Chem.MolFromSmiles('CC(F)(F)CC(O)COC=O')
    patt2 = Chem.MolFromSmiles('COC(=O)O')
    m = Chem.MolFromSmiles(smiles)
    flag1 = m.HasSubstructMatch(patt1)
    flag2 = m.HasSubstructMatch(patt2)
    if flag1 and not flag2:
        return True
    return False

def classify_n2oh(smiles):
    pattern = re.compile(r'^OC+C\(I\)C(C\(F\)\(F\))+F$')
    if pattern.search(smiles) is not None:
        return True
    return False

def classify_cocco(smiles):
    pattern = re.compile('.*C\(=O\)OCC\(O\)C(C\(F\)\(F\))+C\(F\)\(C\(F\)\(F\)F\)C\(F\)\(F\)F')
    if pattern.search(smiles) is not None:
        return True
    return False

def classify_ohi(smiles):
    pattern = re.compile(r'^C+\(I\)C(C\(F\)\(F\))+F$')
    x = count_Atom(smiles)
    if x != -1:
        num_c = x['C']
        num_h = x['H']
        num_f = x['F']
        num_i = x['I']
        num_atoms_auto = x['nums']
        if pattern.search(smiles) is not None \
            and (num_i + num_f + num_c + num_h == num_atoms_auto) \
            and (num_c + 1 - (num_f + num_h + num_i) / 2 == 0):
            return True
    return False


def classifying_fluorotelomers(pfas):
    if classify_ftis(pfas):
        return ['PFAA precursors', 'n:2 fluorotelomer-based substances']
    if classify_ftohs(pfas):
        return ['PFAA precursors', 'n:2 fluorotelomer-based substances']
    if classify_ftacs(pfas):
        #  return ['Fluorotelomer PFAA precursors', 'n:2 FTACs']
        return ['PFAA precursors', 'n:2 fluorotelomer-based substances']
    if classify_ftmacs(pfas):
        #  return ['Fluorotelomer PFAA precursors', 'n:2 FTMACs']
        return ['PFAA precursors', 'n:2 fluorotelomer-based substances']
    if classify_monoesters(pfas):
        # return ['Fluorotelomer PFAA precursors', 'n:2 monoPAPs']
        return ['PFAA precursors', 'n:2 fluorotelomer-based substances']
    if classify_diesters(pfas):
        # return ['Fluorotelomer PFAA precursors', 'n:2 diPAPs']
        return ['PFAA precursors', 'n:2 fluorotelomer-based substances']
    if classify_ftals(pfas):
        # return ['Fluorotelomer PFAA precursors', 'n:2 FTALs']
        return ['PFAA precursors', 'n:2 fluorotelomer-based substances']
    if classify_ftcas(pfas):
        # return ['Fluorotelomer PFAA precursors', 'n:2 FTCAs']
        return ['Polyfluoroalkyl acids', 'PolyFCAs']
    if classify_ftsas(pfas):
        # return ['Fluorotelomer PFAA precursors', 'n:2 FTSAs']
        return ['Polyfluoroalkyl acids', 'PolyFSAs']
    if classify_sfaenes(pfas):
        return ['PFAA precursors', 'SFAenes']
    if classify_three_acids(pfas):
        # return ['Fluorotelomer PFAA precursors', 'n:3 Acids']
        return ['Polyfluoroalkyl acids', 'PolyFCAs']
    if classify_n1_ftohs(pfas):
        return ['PFAA precursors', 'n:1 FTOHs']      # n:1 FTOHs
    if classify_ohi(pfas):
        return ['PFAA precursors', 'n:2 fluorotelomer-based substances']   # n:2 FTIsD-I
    if classify_scfff(pfas):
        return ['PFAA precursors', 'n:2 fluorotelomer-based substances']  # ccs
    if classify_fiif(pfas):
        return ['PFAA precursors', 'n:2 fluorotelomer-based substances']  # shuang I
    if classify_ifcl(pfas):
        return ['PFAA precursors', 'n:2 fluorotelomer-based substances']  # han I derivatives  CCI
    if classify_poooh(pfas):
        return ['PFAA precursors', 'n:2 fluorotelomer-based substances']     #  p suan + chun  linsuanerqingyan
    if classify_chun(pfas):
        return ['PFAA precursors', 'n:2 fluorotelomer-based substances']
    if classify_n2oh(pfas):
        return ['PFAA precursors', 'n:2 fluorotelomer-based substances']
    return None