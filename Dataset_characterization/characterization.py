import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Patch
from tqdm import tqdm
import umap
from sklearn.preprocessing import StandardScaler
from rdkit import Chem
from rdkit.Chem import Descriptors, AllChem
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit.ML.Descriptors import MoleculeDescriptors


def get_morgan2(mol):
    return list(AllChem.GetMorganFingerprintAsBitVect(mol,2,nBits=2048))

def data_to_visualize_umap (df_1, df_2):
    dataframe = pd.concat([df_1, df_2], axis=0,ignore_index=True)
    property_data = dataframe.drop(columns=['SmilesForDropDu', 'db','Molecule'], axis=1)
    # Physicochemical properties
    scaled_property_data = StandardScaler().fit_transform(property_data)
    dataframe.Molecule = dataframe.SmilesForDropDu.apply(Chem.MolFromSmiles)
    # Morgan2 fingerprints
    dataframe['fp'] = dataframe.Molecule.apply(get_morgan2)
    morgan2_cols_list = ['morgan2_b'+ str(i) for i in list(range(2048))]
    morgan2_df = pd.DataFrame(
        dataframe['fp'].to_list(), 
        columns=morgan2_cols_list, 
        index=dataframe.index
    )
    dataframe = pd.concat([dataframe, morgan2_df], axis=1)
    morgan2_cols = dataframe[morgan2_cols_list].to_numpy()
    return dataframe, scaled_property_data, morgan2_cols

def plot_umap_benignAndtoxic(np_data_toplot, dataframe,):
    colors={"benign_drugs": "#ff7f00",
            "toxic_drugs": "#984ea3"}
    reducer   = umap.UMAP(random_state=42)
    embedding = reducer.fit_transform(np_data_toplot)
    color_array = dataframe.db.map(colors).values
    plt.figure(figsize=(8, 6))
    plt.scatter(
        embedding[:, 0],
        embedding[:, 1],
        c=color_array,
        s=8,
        alpha=0.8
    )
    for label, hexcol in [("Benign compounds", colors["benign_drugs"]),("Toxic compounds", colors["toxic_drugs"])]:
        plt.scatter([], [], c=hexcol, alpha=1, s=8,
                    label=label)
    plt.gca().set_aspect("equal", "datalim")
    plt.legend(title="Compound type", fontsize=12)
    plt.tight_layout()
    plt.show()
    
def plot_umap_trainingAndtext(np_data_toplot, dataframe):
    reducer   = umap.UMAP(random_state=42)
    embedding = reducer.fit_transform(np_data_toplot)
    alpha_values = {
        "Training set": 1,
        "Test set":1
    }
    colors = {
        "Training set": "#508AB2",  
        "Test set":"#B36A6F"   
    }
    plt.figure(figsize=(8, 6))
    for category in colors:
        mask = (dataframe['db'] == category).values
        plt.scatter(
            embedding[mask, 0],
            embedding[mask, 1],
            c=colors[category], 
            alpha=alpha_values[category],
            label=category,
            s=8
        )
    plt.gca().set_aspect('equal', 'datalim')
    plt.legend(title="Compound group", fontsize=12)
    plt.tight_layout()
    plt.show()

def pairwise_tanimoto_similarity(smi_list, query_set, fpe):
    default_value = -1
    results = []
    for my_smi in tqdm(query_set['SmilesForDropDu'], desc="Calculating similarities"):
        sim_results = fpe.similarity(my_smi, 0, n_workers=31)
        sim_dict = dict.fromkeys(smi_list, default_value)
        for idx, value in sim_results:
            p = smi_list[idx]
            if value > sim_dict[p]:
                sim_dict[p]=value
        results.append({'smi':my_smi,'dict':sim_dict})
    temp_save = pd.DataFrame(results)
    temp_save[smi_list] = [list(value_1.values()) for value_1 in temp_save['dict']]
    return temp_save.drop(columns=['dict'])

def cal_mw(mol):
    return round(Descriptors.MolWt(mol), 3)

def plot_umap_trainingAndtext_morgan2(np_data_toplot, dataframe):
    reducer = umap.UMAP(random_state=42)
    embedding = reducer.fit_transform(np_data_toplot)

    outlier_indices_1 = np.where((embedding[:, 1] < 10) & (embedding[:, 0] < 5))[0]
    outlier_indices_2 = np.where((embedding[:, 1] < 0) & (embedding[:, 0] < 15))[0]
    outlier_indices_3 = np.where((embedding[:, 1] < 7) & (embedding[:, 0] < 8))[0]
    outlier_indices_4 = np.where((embedding[:, 1] >12.5) & (embedding[:, 0] < 9))[0]
    outlier_indices_5 = np.where((embedding[:, 1] >12.5) & (embedding[:, 0] >10))[0]

    alpha_values = {
        "Training set": 0.8,
        "Test set":0.8
    }
    colors = {
        "Training set": "#508AB2",  
        "Test set": "orangered"   
    }

    fig, ax = plt.subplots(figsize=(8, 6))
    
    for category in colors:
        mask = (dataframe['db'] == category).values
        ax.scatter(
            embedding[mask, 0],
            embedding[mask, 1],
            c=colors[category], 
            alpha=alpha_values[category],
            label=category,
            s=8
        )				
    circle_patches = []
    for idx, (outlier_indices, label, color) in enumerate([
        (outlier_indices_1, "cluster A", "green"),
        (outlier_indices_2, "cluster B", "orange"),
        (outlier_indices_3, "cluster C", "purple"),
        (outlier_indices_4, "cluster D", "brown"),
        (outlier_indices_5, "cluster E", "olive")

    ]):
        if len(outlier_indices) > 0:
            cluster_coords = embedding[outlier_indices]
            center_x, center_y = cluster_coords.mean(axis=0)
            distances = np.linalg.norm(cluster_coords - [center_x, center_y], axis=1)
            radius = distances.max() + 0.2
            circle = Circle((center_x, center_y), radius=radius, edgecolor=color, facecolor='none', lw=1)
            ax.add_patch(circle)
            circle_patches.append(Patch(edgecolor=color, facecolor='none', label=label))

    handles, labels = ax.get_legend_handles_labels()
    handles += circle_patches
    labels += [patch.get_label() for patch in circle_patches]
    ax.legend(handles=handles, labels=labels, title="Compound group", loc='upper right',fontsize=12)
    ax.set_aspect('equal', 'datalim')
    plt.xlabel("UMAP 1", fontsize=12)
    plt.ylabel("UMAP 2", fontsize=12)
    plt.tight_layout()
    plt.show()

def plot_umap_trainingAndtext_physichem(np_data_toplot, dataframe, ):
    reducer = umap.UMAP(random_state=42)
    embedding = reducer.fit_transform(np_data_toplot)
    alpha_values = {"Training set": 1, "Test set": 1}
    colors = {"Training set": "#508AB2", "Test set": "orangered"}

    fig, ax = plt.subplots(figsize=(8, 6))

    for category in colors:
        mask = (dataframe['db'] == category).values
        ax.scatter(
            embedding[mask, 0],
            embedding[mask, 1],
            c=colors[category],
            alpha=alpha_values[category],
            label=category,
            s=10
        )	
    circle_patches = []
    handles, labels = ax.get_legend_handles_labels()
    handles += circle_patches
    labels += [patch.get_label() for patch in circle_patches]
    ax.legend(handles=handles, labels=labels, title="Compound group", fontsize=12)

    ax.set_aspect('equal', 'datalim')
    plt.xlabel("UMAP 1", fontsize=12)
    plt.ylabel("UMAP 2", fontsize=12)
    plt.tight_layout()
    plt.show()

def compute_2Drdkit(mol):
    calc = MoleculeDescriptors.MolecularDescriptorCalculator([x[0] for x in Descriptors._descList])
    ds = calc.CalcDescriptors(mol)
    return list(ds)

def desc_calculation(dataset, desc_cols):
    dataset.Molecule = dataset.SmilesForDropDu.apply(Chem.MolFromSmiles)
    dataset['desc'] = dataset.Molecule.apply(compute_2Drdkit)
    dataset[desc_cols] = dataset.desc.to_list()
    dataset.drop("desc",axis=1,inplace=True)
    dataset[desc_cols].dropna(inplace=True)
    return dataset

def plot_distribution_of_physichem_prop(Training_data, selected_desc, benign_set, toxic_set):
    fig, axes = plt.subplots(2, 3, figsize=(18,10), constrained_layout=True)
    axes = axes.flatten()
    for i, descriptor in enumerate(selected_desc):
        benign_np = benign_set[descriptor].to_numpy()
        toxic_np = toxic_set[descriptor].to_numpy()
        steps = np.linspace(0, Training_data[descriptor].max(), 20)
        ax = axes[i]
        sim_bins_benign = [((benign_np <= s).sum() / benign_np.shape[0])* 100 for s in steps]
        sim_bins_toxic = [((toxic_np <= s).sum() / toxic_np.shape[0])* 100 for s in steps]
        ax.set(
            title="Distribution of " + descriptor
        )
        ax.set_xlabel(descriptor)
        ax.set_ylabel('Percentage')
        ax.plot(
            steps,
            sim_bins_benign,
            color="#ff7f00",
            lw=1,
            alpha=1,
            linestyle = '-'
        )
        ax.plot(
            steps,
            sim_bins_toxic,
            color="#984ea3",
            lw=1,
            alpha=1,
            linestyle = '-'
        )
        ax.fill_between(
            steps,
            sim_bins_toxic,
            interpolate= True,
            color="#984ea3",
            alpha=0.7,
            label = "toxic"
        )
        ax.fill_between(
            steps,
            sim_bins_benign,
            interpolate= True,
            color="#ff7f00",
            alpha=0.7,
            label="benign"
        )
        ax.legend()

def add_murcko_scaffold(df):
    df['Molecule'] = df.SmilesForDropDu.apply(Chem.MolFromSmiles)
    df['MurckoScaffold'] = df.Molecule.apply(MurckoScaffold.GetScaffoldForMol)
    df['MurckoScaffold_smi'] = df.MurckoScaffold.apply(Chem.MolToSmiles)
    return df[df['MurckoScaffold_smi'] != '']

def count_shared_scaffold(base_set, query_set, img=False):
    query_scaffolds = set(query_set['MurckoScaffold_smi'])
    shared_scaffolds = base_set[base_set['MurckoScaffold_smi'].isin(query_scaffolds)].drop_duplicates(subset=['MurckoScaffold_smi'])
    print(f"Number of unique MurckoScaffolds in benign set: {len(set(base_set['MurckoScaffold_smi']))}")
    print(f"Number of unique MurckoScaffolds in toxic set: {len(query_scaffolds)}")
    print(f"Number of shared scaffolds between two sets: {len(shared_scaffolds)}")
    print(f"Percentage of shared scaffolds in query set: {(len(shared_scaffolds) / len(query_scaffolds)) * 100:.2f}%")
    max_cluster_size = query_set.groupby('MurckoScaffold_smi')['MurckoScaffold_smi'].count().max()
    print(f"Size of the largest MurckoScaffold cluster in query set: {max_cluster_size}")
    if img:
        return Chem.Draw.MolsToGridImage(list(shared_scaffolds['MurckoScaffold']), molsPerRow=7, subImgSize=(400, 200))
    
def functional_group_analysis(df, SMARTS_pattern):
    funtional_group = pd.DataFrame()
    benign_groups = []
    benign_pattern_name = []
    for query_mol in df.Molecule:
        for i, mol_funtionalgroup in enumerate(SMARTS_pattern.mol_pattern):
            if query_mol.HasSubstructMatch(mol_funtionalgroup):
                benign_groups.append(SMARTS_pattern.SMARTS_pattern[i])
                benign_pattern_name.append(SMARTS_pattern.Pattern_name[i])
    funtional_group['SMARTS_pattern'] = benign_groups
    funtional_group['Pattern_name'] = benign_pattern_name
    return funtional_group