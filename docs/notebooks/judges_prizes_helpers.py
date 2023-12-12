# basic data science
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.sparse import coo_matrix, csr_matrix
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedKFold
from statsmodels.stats.multitest import multipletests

# utils
import os
import itertools
import warnings
import pickle

# single cell data analysis libraries
import anndata
import decoupler as dc

# neural net and keras
import tensorflow as tf
import keras
from keras import backend as K
from keras.layers import Input, Dense, Dropout, BatchNormalization, Activation, GaussianNoise

# chemical libraries
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors

def plot_availability(de_df, cells_to_predict = ['Myeloid cells', 'B cells'], fig_height=5):
    """
    Plot the availability of cell-drug combinations in the dataset.
    
    Parameters
    ----------
    de_df : pd.DataFrame
        DE dataset
    cells_to_predict : list
        List of cell types to predict
    fig_height : int
        Height of the figure in inches
    """
    
    # prepare availability df
    available = de_df[['cell_type', 'sm_name']].drop_duplicates()
    available['cell_drug'] = available['cell_type'] + '__' + available['sm_name']
    all_combinations = pd.DataFrame(list(itertools.product(de_df['cell_type'].unique(), de_df['sm_name'].unique())), columns=['cell_type', 'sm_name'])
    all_combinations['cell_drug'] = all_combinations['cell_type'] + '__' + all_combinations['sm_name']
    all_combinations['status'] = np.where(
        all_combinations['cell_drug'].isin(available['cell_drug']), 'train',
        np.where(all_combinations['cell_type'].isin(cells_to_predict), 'to_predict', 'missing')
    )
    all_combinations['status'] = pd.Categorical(all_combinations['status'], categories=['train', 'to_predict', 'missing'])
    all_combinations = all_combinations.sort_values(['cell_type', 'status', 'sm_name'])

    # create plot
    plt.figure(figsize=(12, fig_height))
    sns.scatterplot(data=all_combinations, x='sm_name', y='cell_type', hue='status')
    # remove x ticks and labels
    plt.xticks([])
    plt.xlabel('Drugs')
    plt.ylabel('Cell types')
    plt.margins(x=0.005, y=0.1)
    plt.legend(bbox_to_anchor=(0.5, 1.15), loc='center', ncol=3, borderaxespad=0.)
    plt.tight_layout()
    plt.show()

def create_pseudobulk_profiles(data, metadata, gene_column = 'gene', multi = False):
    """
    Create pseudobulk profiles from single cell data.

    Parameters
    ----------
    data : pd.DataFrame
        Single cell data
    metadata : pd.DataFrame
        Metadata
    gene_column : str
        Name of the column that contains the gene names
    multi : bool
        Whether the data is coming from the ATAC-Seq multiome experiment or not
    
    Returns
    -------
    pseudobulk_profiles : pd.DataFrame
        Pseudobulk profiles
    pseudobulk_meta : pd.DataFrame
        Metadata of the pseudobulk profiles
    """

    # create two dictionaries: one for row ids and one for column ids that maps to integers
    genes = set(data[gene_column].values)
    cells = set(data['obs_id'].values)
    gene_dict = {gene: i for i, gene in enumerate(genes)}
    cell_dict = {cell: i for i, cell in enumerate(cells)}
    
    # in the data frame, map gene and cell columns to their integers
    data['gene'] = data[gene_column].map(gene_dict)
    data['obs_id'] = data['obs_id'].map(cell_dict)

    # create a sparse matrix from the data frame
    csr_matrix = coo_matrix((data['count'], (data['obs_id'], data['gene']))).tocsr()

    # create anndata object
    adata = anndata.AnnData(X=csr_matrix)
    cell_annotation = pd.DataFrame({'num_id': cell_dict.values(), 'obs_id': cell_dict.keys()}, index = cell_dict.keys()).sort_values(by=['num_id']).merge(metadata, on='obs_id', how='left')
    gene_annotation = pd.DataFrame({'num_id': gene_dict.values(), 'gene': gene_dict.keys()}, index = gene_dict.keys()).sort_values(by=['num_id'])

    # add annotations to the anndata object
    adata.obs = cell_annotation
    adata.var = gene_annotation
    
    # define pseudobulk ids
    if multi:
        adata.obs['pseudobulk_id'] = adata.obs['cell_type'] + '__' + adata.obs['donor_id']
    else:
        adata.obs['pseudobulk_id'] = adata.obs['plate_name'] + '__' + adata.obs['well'] + '__' + adata.obs['donor_id'] + '__' + adata.obs['cell_type'] + '__' + adata.obs['sm_name']

    # create psuedobulk profiles
    genes = adata.var['gene']
    pseudobulk_ids = adata.obs['pseudobulk_id'].unique()
    pseudobulk_profiles = pd.DataFrame(index=pseudobulk_ids, columns=genes)
    for pseudobulk_id in adata.obs['pseudobulk_id'].unique():
        pseudobulk = adata[adata.obs['pseudobulk_id'] == pseudobulk_id]
        pseudobulk_profile = pseudobulk.X.sum(axis=0)
        pseudobulk_profiles.loc[pseudobulk_id] = pseudobulk_profile

    # metadata
    if multi:
        pseudobulk_meta = adata.obs[['pseudobulk_id', 'cell_type', 'donor_id']].drop_duplicates().set_index('pseudobulk_id')
    else:
        pseudobulk_meta = adata.obs[['pseudobulk_id', 'cell_type', 'donor_id', 'library_id', 'plate_name', 'well', 'sm_name']].drop_duplicates().set_index('pseudobulk_id')
    
    return pseudobulk_profiles, pseudobulk_meta

def load_rna_pseudobulk(rna_file = 'data/my_data/pseudo_rna.pickle'):
    """
    Load RNA pseudobulk profiles if they exist, otherwise create them.

    Returns
    -------
    pseudo_rna : pd.DataFrame
        RNA pseudobulk profiles
    pseudo_rna_meta : pd.DataFrame
        Metadata of the RNA pseudobulk profiles
    """

    if os.path.exists(rna_file):
        with open(rna_file, 'rb') as f:
            pseudo_rna, pseudo_rna_meta = pickle.load(f)
    else:
        rna_data = pd.read_parquet('data/source_data/adata_train.parquet')
        rna_meta = pd.read_csv('data/source_data/adata_obs_meta.csv')
        rna_data.drop('normalized_count', axis=1, inplace=True)
        pseudo_rna, pseudo_rna_meta = create_pseudobulk_profiles(rna_data, rna_meta, multi=False)
        with open(rna_file, 'wb') as f:
            pickle.dump((pseudo_rna, pseudo_rna_meta), f)

    return pseudo_rna, pseudo_rna_meta

def load_atac_pseudobulk(atac_file = 'data/my_data/pseudo_atac.pickle'):
    """
    Load ATAC pseudobulk profiles if they exist, otherwise create them.

    Returns
    -------
    pseudo_atac : pd.DataFrame
        ATAC pseudobulk profiles
    pseudo_atac_meta : pd.DataFrame
        Metadata of the ATAC pseudobulk profiles
    """

    if os.path.exists(atac_file):
        with open(atac_file, 'rb') as f:
            pseudo_atac, pseudo_atac_meta = pickle.load(f)
    else:   
        atac_data = pd.read_parquet('data/source_data/multiome_train.parquet')
        atac_meta = pd.read_csv('data/source_data/multiome_obs_meta.csv')
        atac_data.drop('normalized_count', axis=1, inplace=True)
        pseudo_atac, pseudo_atac_meta = create_pseudobulk_profiles(atac_data, atac_meta, gene_column='location', multi=True)
        with open(atac_file, 'wb') as f:
            pickle.dump((pseudo_atac, pseudo_atac_meta), f)

    return pseudo_atac, pseudo_atac_meta

def pcaplot(data, meta, int_ax):
    """
    Plot PCA of data with cell type labels

    Parameters
    ----------
    data : pandas.DataFrame
        Data matrix with cells as rows and features as columns
    meta : pandas.DataFrame
        Metadata matrix with cells as rows and metadata as columns
    int_ax : matplotlib.axes.Axes
        Axis to plot PCA on
    """
    
    pca = PCA(n_components=2)
    log_data = np.log2(data.astype(float) + 1)
    pca.fit(log_data)
    pca_res = pca.transform(log_data)
    pca_df = pd.DataFrame(pca_res, columns=['PC1', 'PC2'], index=data.index)
    pca_df['cell_type'] = meta['cell_type']
    pc1_title = 'PC1 (' + str(round(pca.explained_variance_ratio_[0]*100, 2)) + '%)'
    pc2_title = 'PC2 (' + str(round(pca.explained_variance_ratio_[1]*100, 2)) + '%)'
    sns.scatterplot(data=pca_df, x='PC1', y='PC2', hue='cell_type', ax=int_ax)
    int_ax.set_xlabel(pc1_title)
    int_ax.set_ylabel(pc2_title)

def compute_lfc(ps_rna, ps_meta):
    """
    Compute "raw" log2 fold change for each cell type and drug.

    Parameters
    ----------
    ps_rna : pandas.DataFrame
        Pseudobulk RNA expression data.
    ps_meta : pandas.DataFrame
        Pseudobulk metadata.
    
    Returns
    -------
    lfc_df : pandas.DataFrame
        Log2 fold change for each cell type and drug.
    """

    # make pseudobulk index a multi-index by splitting current index with '__' and making it plate, well, donor, cell_type and sm_name
    rna_pseudo_mi = ps_rna.merge(ps_meta[['library_id', 'cell_type', 'sm_name']], left_index=True, right_index=True)
    rna_pseudo_mi.reset_index(drop=True, inplace=True)
    rna_pseudo_mi.set_index(['library_id', 'cell_type', 'sm_name'], inplace=True)

    # get the library_id and cell_type part of the index as a list of tuples
    rna_pseudo_mi_index = rna_pseudo_mi.index.tolist()
    rna_pseudo_mi_index = [i[:2] for i in rna_pseudo_mi_index]

    # get DMSO samples
    rna_dmso = rna_pseudo_mi.loc[rna_pseudo_mi.index.get_level_values('sm_name') == 'Dimethyl Sulfoxide']
    rna_dmso.reset_index(level='sm_name', drop=True, inplace=True)

    # compute mean log2 fold change for each cell type and drug
    lfc_df = np.log2(rna_pseudo_mi.astype(float) + 1) - np.log2(rna_dmso.astype(float) + 1)
    lfc_df = lfc_df.groupby(['cell_type', 'sm_name']).mean()

    return lfc_df

def input_processor(input_shape, output_units, dropout = 0.1, noise = 0.1, name = 'input'):
    """
    Create input processor for neural net. This includes a Gaussian noise layer, batch normalization, a dense layer and dropout.

    Parameters
    ----------
    input_shape : int
        Number of input features.
    output_units : int
        Number of output units.
    dropout : float
        Dropout rate.
    noise : float
        Noise standard deviation.
    name : str
        Name of input layer.

    Returns
    -------
    input_layer : keras layer
        Input layer.
    dense : keras layer
        Output layer.
    """

    input_layer = Input(shape = input_shape, name = name)
    dense = GaussianNoise(noise)(input_layer)
    dense = BatchNormalization()(dense)
    dense = Dense(output_units, activation='elu')(dense)
    dense = Dropout(dropout)(dense)
    return input_layer, dense

def neuralnet_regressor(
        drug_features_dict, cell_features_dict, output_size, 
        drug_dense_units = 64, cell_dense_units = 256,
        n_mid_layers = 2, n_mid_units = 128,  dropout=0.1
    ):
    """
    Creates a neural network regressor model.

    Parameters
    ----------
    drug_features_dict : dict
        Dictionary of drug features. Keys are feature names and values are dictionaries with the following keys:
            - n_inputs : int
                Number of input nodes.
            - n_units : int
                Number of units in the dense layer.
            - dropout : float
                Dropout rate.
            - noise : float
                Noise rate.

    cell_features_dict : dict
        Dictionary of cell features. Keys are feature names and values are dictionaries with the following keys:
            - n_inputs : int
                Number of input nodes.
            - n_units : int
                Number of units in the dense layer.
            - dropout : float
                Dropout rate.
            - noise : float
                Noise rate.

    output_size : int
        Number of output nodes.

    drug_dense_units : int, optional
        Number of units in the drug dense layer. The default is 64.

    cell_dense_units : int, optional
        Number of units in the cell dense layer. The default is 256.

    n_mid_layers : int, optional
        Number of middle layers. The default is 2.

    n_mid_units : int, optional
        Number of units in the middle layers. The default is 128.

    dropout : float, optional
        Dropout rate. The default is 0.1.

    Returns
    -------
    model : keras.Model
        Neural network regressor model.
    """
    
    # get drug inputs
    drug_inputs, drug_denses = [], []
    for drug_feature_name, drug_feature_dict in drug_features_dict.items():
        drug_input, drug_dense = input_processor(drug_feature_dict['n_inputs'], drug_feature_dict['n_units'], drug_feature_dict['dropout'], drug_feature_dict['noise'], name = drug_feature_name)
        drug_inputs.append(drug_input)
        drug_denses.append(drug_dense)

    cell_inputs, cell_denses = [], []
    for cell_feature_name, cell_feature_dict in cell_features_dict.items():
        cell_input, cell_dense = input_processor(cell_feature_dict['n_inputs'], cell_feature_dict['n_units'], cell_feature_dict['dropout'], cell_feature_dict['noise'], name = cell_feature_name)
        cell_inputs.append(cell_input)
        cell_denses.append(cell_dense)
    
    # create cell and drug dense layers
    concat_drug = keras.layers.concatenate(drug_denses)
    drug_dense = Dense(drug_dense_units, activation='elu')(concat_drug)
    drug_dense = Dropout(dropout)(drug_dense)
    concat_cell = keras.layers.concatenate(cell_denses)
    cell_dense = Dense(cell_dense_units, activation='elu')(concat_cell)
    cell_dense = Dropout(dropout)(cell_dense)

    # concat
    concat = keras.layers.concatenate([drug_dense, cell_dense])
    for i in range(n_mid_layers):
        concat = Dense(n_mid_units, activation='elu')(concat)
        concat = Dropout(dropout)(concat)

    # output
    output = Dense(output_size, activation='linear', name = 'output')(concat)

    # create model
    input_list = drug_inputs + cell_inputs
    model = keras.Model(inputs=input_list, outputs=output)

    return model

def mrrmse(y_true, y_pred):
    """
    Mean Rowwise Root Mean Squared Error function for keras.

    Parameters
    ----------
    y_true : keras tensor
        Ground truth values.

    y_pred : keras tensor
        The predicted values.

    Returns
    -------
    keras tensor
        Mean Rowwise Root Mean Squared Error.
    """
    return K.mean(K.sqrt(K.mean(K.square(y_true - y_pred), axis=1)))

def load_default_params():
    """
    Load default parameters for the neural network model.

    Returns:
    --------
    params: dict
        Dictionary with default parameters for the neural network model.
    """
    params = {
        'top_n_genes': 128,  
        'de_drug_units': 64, 'de_cell_units': 16,
        'fc_drug_units': 64, 'fc_cell_units': 16,
        'drug_dense_units': 128, 'cell_dense_units': 32, 
        'n_mid_layers': 2, 'n_mid_units': 64,
        'noise': 0.05, 'dropout': 0.3, 
        'learning_rate': 0.01,
        'reduce_lr_factor': 0.8, 'reduce_lr_patience': 10, 'reduce_lr_min_lr': 0.000001,
        'epochs': 300, 'batch_size': 128
    }
    return params

def prepare_inputs(drug_df_list, cell_df_list, train_df, test_df, int_features):
    """
    Prepare inputs for the neural network model.
    
    Parameters
    ----------
    drug_df_list : list
        List of drug indexed features data frames.
    cell_df_list : list
        List of cell indexed features data frames.
    train_df : pandas.DataFrame
        Training data frame.
    test_df : pandas.DataFrame 
        Test data frame.
    int_features : list
        List of input features to select from the data frames.
    
    Returns
    -------
    X_train : list
        List of matrices for training.
    X_test : list
        List of matrices for testing.
    y_train : np.array
        Training labels.
    y_test : np.array
        Test labels.
    """
    drug_X_train = [drug_df.loc[train_df.index.get_level_values(1).values, int_features].values for drug_df in drug_df_list]
    drug_X_test = [drug_df.loc[test_df.index.get_level_values(1).values, int_features].values for drug_df in drug_df_list]
    cell_X_train = [cell_df.loc[train_df.index.get_level_values(0).values, int_features].values for cell_df in cell_df_list]
    cell_X_test = [cell_df.loc[test_df.index.get_level_values(0).values, int_features].values for cell_df in cell_df_list]

    X_train = drug_X_train + cell_X_train
    X_test = drug_X_test + cell_X_test
    y_train = train_df.values
    y_test = test_df.values

    return X_train, X_test, y_train, y_test





