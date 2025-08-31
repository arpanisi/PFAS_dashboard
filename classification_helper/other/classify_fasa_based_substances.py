#from pfas_object import PFAS
from .classify_pfaa_precursors import classify_fasas
from .atom_count import count_Atom, standard_mol
import re


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




'''
def classifying_fasa_precursors(pfas):
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
    return None

'''