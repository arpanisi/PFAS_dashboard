from rdkit import Chem
from rdkit.Chem import Draw
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from deepchem.feat import Mol2VecFingerprint
from .create_network import umap_network
import networkx as nx


def find_similar(smiles):
    fingerprints_df = pd.read_csv('data/pfas_epa_mol2vec_descriptors.csv', index_col=0)
    fp = Mol2VecFingerprint()
    mol2vec = fp.featurize(smiles)

    df2 = np.vstack([fingerprints_df.values, mol2vec[0]])

    # Construct Graph G=(V,E)
    G, _, _ = umap_network(df2, n_components=3, n_neighbors=100)

    smiles_counter = list(G.nodes())[-1]
    reachable_nodes = nx.single_source_shortest_path_length(G, smiles_counter)

    reachable_nodes_list = [key for key, value in sorted(reachable_nodes.items(),
                                                         key=lambda item: item[1])]
    reachable_nodes_list = reachable_nodes_list[1:]

    pfas_complete_report = load_pfas_report()
    alternative_df = pfas_complete_report.iloc[reachable_nodes_list]

    return alternative_df

def draw_mol(smiles, padding=200):

    # Convert SMILES to molecule
    mol = Chem.MolFromSmiles(smiles)

    # Generate 2D coordinates
    # Chem.rdDepictor.Compute2DCoords(mol)

    fig, ax = plt.subplots(figsize=(3, 3))

    # Draw molecule as image
    img = Draw.MolToImage(mol, size=(800, 800))  # Draw image directly to PIL format
    # Convert to numpy array for processing
    img_np = np.array(img)

    # Detect the non-white pixels (assuming white is [255, 255, 255])
    non_white_pixels = np.where(np.all(img_np[:, :, :3] != [255, 255, 255], axis=-1))

    # Get bounding box of non-white pixels
    x_min, x_max = non_white_pixels[1].min(), non_white_pixels[1].max()
    y_min, y_max = non_white_pixels[0].min(), non_white_pixels[0].max()

    # Define padding

    # Apply padding while ensuring it doesn’t go out of bounds
    x_min = max(x_min - padding, 0)
    x_max = min(x_max + padding, img.width)
    y_min = max(y_min - padding, 0)
    y_max = min(y_max + padding, img.height)

    # Crop the image to this padded bounding box
    cropped_img = img.crop((x_min, y_min, x_max, y_max))
    ax.imshow(cropped_img)
    ax.axis("off")

    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)

    return fig


def load_pfas_report():
    complete_data = pd.read_csv('data/PFASSTRUCTV5-2024-07-12_wClass_filtered.csv', index_col=0)

    return complete_data