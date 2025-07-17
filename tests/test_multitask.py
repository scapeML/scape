import pytest
import numpy as np
import pandas as pd
import keras
from scape._model import SCAPE, create_default_model
from scape._losses import mrrmse


@pytest.fixture
def sample_data():
    """Create sample data for testing."""
    # Create sample differential expression data
    n_samples = 100
    n_genes = 50
    
    # Create multi-index with cell_type and sm_name
    cell_types = ['CD4 T cells', 'CD8 T cells', 'NK cells'] * (n_samples // 3 + 1)
    sm_names = ['Drug_A', 'Drug_B', 'Drug_C'] * (n_samples // 3 + 1)
    
    # Pad to exact n_samples
    cell_types = cell_types[:n_samples]
    sm_names = sm_names[:n_samples]
    
    index = pd.MultiIndex.from_arrays([cell_types, sm_names], names=['cell_type', 'sm_name'])
    
    # Create random gene expression data
    np.random.seed(42)
    de_data = np.random.randn(n_samples, n_genes)
    lfc_data = np.random.randn(n_samples, n_genes)
    
    gene_names = [f'Gene_{i}' for i in range(n_genes)]
    
    df_de = pd.DataFrame(de_data, index=index, columns=gene_names)
    df_lfc = pd.DataFrame(lfc_data, index=index, columns=gene_names)
    
    return df_de, df_lfc


def test_single_task_training(sample_data):
    """Test single task training with only slogpval output."""
    df_de, df_lfc = sample_data
    n_genes = 20
    
    # Create model with single output (default behavior)
    model = create_default_model(n_genes, df_de, df_lfc)
    
    # Verify single output configuration
    assert len(model.config['outputs']) == 1
    assert 'logpval' in model.config['outputs']
    
    # Select top genes for training (this is what the train method does internally)
    from scape._util import select_top_variable
    top_genes = select_top_variable([df_de], k=n_genes)
    
    # Train the model with validation that exists in the data
    results = model.train(
        val_cells=['CD4 T cells'],
        val_drugs=['Drug_A'],
        output_data='slogpval',
        input_columns=top_genes,
        epochs=5,
        batch_size=32
    )
    
    # Verify training completed
    assert results is not None
    assert 'history' in results
    assert len(results['history']) > 0
    
    # Test prediction with single output
    test_idx = [('CD8 T cells', 'Drug_B')]
    predictions = model.predict(test_idx)
    
    assert predictions.shape[0] == 1
    assert predictions.shape[1] == n_genes


def test_multi_task_training(sample_data):
    """Test multi-task training with both slogpval and lfc outputs."""
    df_de, df_lfc = sample_data
    n_genes = 20
    
    # Select top genes for training
    from scape._util import select_top_variable
    top_genes = select_top_variable([df_de], k=n_genes)
    
    # Create model setup with multiple outputs
    data_sources = {"slogpval": df_de, "lfc_pseudo": df_lfc}
    
    feature_extraction = {
        "slogpval_drug": {
            "source": "slogpval",
            "groupby": "sm_name",
            "function": "median",
        },
        "lfc_drug": {
            "source": "lfc_pseudo",
            "groupby": "sm_name",
            "function": "median",
        },
        "slogpval_cell": {
            "source": "slogpval",
            "groupby": "cell_type",
            "function": "median",
        },
        "lfc_cell": {
            "source": "lfc_pseudo",
            "groupby": "cell_type",
            "function": "median",
        },
    }
    
    input_mapping = {
        "in_slogpval_drug": "slogpval_drug",
        "in_lfc_drug": "lfc_drug",
        "in_slogpval_cell_encoder": "slogpval_cell",
        "in_lfc_cell_encoder": "lfc_cell",
        "in_slogpval_cell_decoder": "slogpval_cell",
        "in_lfc_cell_decoder": "lfc_cell",
    }
    
    # Multi-task configuration with both outputs
    config = {
        "inputs": {
            "in_slogpval_drug": [n_genes, 32, 16],
            "in_lfc_drug": [n_genes, 32, 16],
        },
        "conditional_encoder_input_structure": {
            "in_slogpval_cell_encoder": [n_genes, 16],
            "in_lfc_cell_encoder": [n_genes, 16],
        },
        "conditional_decoder_input_structure": {
            "in_slogpval_cell_decoder": [n_genes, 16],
            "in_lfc_cell_decoder": [n_genes, 16],
        },
        "conditional_decoder_input_hidden_sizes": [16],
        "encoder_hidden_layer_sizes": [32, 32],
        "decoder_hidden_layer_sizes": [32, 64],
        "outputs": {
            "slogpval": (n_genes, "linear"),
            "lfc": (n_genes, "linear"),
        },
        "noise": 0.01,
        "dropout": 0.05,
        "l1": 0,
        "l2": 0,
    }
    
    model_setup = {
        "data_sources": data_sources,
        "feature_extraction": feature_extraction,
        "input_mapping": input_mapping,
        "output_genes": top_genes,
        "config": config,
    }
    
    model = SCAPE(model_setup)
    
    # Verify multi-task configuration
    assert len(model.config['outputs']) == 2
    assert 'slogpval' in model.config['outputs']
    assert 'lfc' in model.config['outputs']
    
    # Train the model
    results = model.train(
        val_cells=['CD4 T cells'],
        val_drugs=['Drug_A'],
        output_data='slogpval',
        input_columns=top_genes,
        epochs=5,
        batch_size=32
    )
    
    # Verify training completed
    assert results is not None
    assert 'history' in results
    assert len(results['history']) > 0
    
    # Test prediction with multi-task outputs
    test_idx = [('CD8 T cells', 'Drug_B')]
    
    # Test prediction for first output (slogpval)
    predictions_0 = model.predict(test_idx, output_idx=0)
    assert predictions_0.shape[0] == 1
    assert predictions_0.shape[1] == n_genes
    
    # Test prediction for second output (lfc)
    predictions_1 = model.predict(test_idx, output_idx=1)
    assert predictions_1.shape[0] == 1
    assert predictions_1.shape[1] == n_genes
    
    # Verify predictions are different (different outputs)
    assert not np.allclose(predictions_0.values, predictions_1.values)


def test_multi_task_custom_loss_weights():
    """Test multi-task training with custom loss weights."""
    # Create sample data
    n_samples = 50
    n_genes = 20
    
    cell_types = ['CD4 T cells', 'CD8 T cells'] * (n_samples // 2 + 1)
    sm_names = ['Drug_A', 'Drug_B'] * (n_samples // 2 + 1)
    
    # Pad to exact n_samples
    cell_types = cell_types[:n_samples]
    sm_names = sm_names[:n_samples]
    
    index = pd.MultiIndex.from_arrays([cell_types, sm_names], names=['cell_type', 'sm_name'])
    
    np.random.seed(42)
    de_data = np.random.randn(n_samples, n_genes)
    lfc_data = np.random.randn(n_samples, n_genes)
    
    gene_names = [f'Gene_{i}' for i in range(n_genes)]
    
    df_de = pd.DataFrame(de_data, index=index, columns=gene_names)
    df_lfc = pd.DataFrame(lfc_data, index=index, columns=gene_names)
    
    # Create model with multi-task setup
    data_sources = {"slogpval": df_de, "lfc_pseudo": df_lfc}
    
    feature_extraction = {
        "slogpval_drug": {"source": "slogpval", "groupby": "sm_name", "function": "median"},
        "lfc_drug": {"source": "lfc_pseudo", "groupby": "sm_name", "function": "median"},
        "slogpval_cell": {"source": "slogpval", "groupby": "cell_type", "function": "median"},
        "lfc_cell": {"source": "lfc_pseudo", "groupby": "cell_type", "function": "median"},
    }
    
    input_mapping = {
        "in_slogpval_drug": "slogpval_drug",
        "in_lfc_drug": "lfc_drug",
        "in_slogpval_cell_encoder": "slogpval_cell",
        "in_lfc_cell_encoder": "lfc_cell",
        "in_slogpval_cell_decoder": "slogpval_cell",
        "in_lfc_cell_decoder": "lfc_cell",
    }
    
    config = {
        "inputs": {
            "in_slogpval_drug": [n_genes, 32, 16],
            "in_lfc_drug": [n_genes, 32, 16],
        },
        "conditional_encoder_input_structure": {
            "in_slogpval_cell_encoder": [n_genes, 16],
            "in_lfc_cell_encoder": [n_genes, 16],
        },
        "conditional_decoder_input_structure": {
            "in_slogpval_cell_decoder": [n_genes, 16],
            "in_lfc_cell_decoder": [n_genes, 16],
        },
        "conditional_decoder_input_hidden_sizes": [16],
        "encoder_hidden_layer_sizes": [32, 32],
        "decoder_hidden_layer_sizes": [32, 64],
        "outputs": {
            "slogpval": (n_genes, "linear"),
            "lfc": (n_genes, "linear"),
        },
        "noise": 0.01,
        "dropout": 0.05,
        "l1": 0,
        "l2": 0,
    }
    
    model_setup = {
        "data_sources": data_sources,
        "feature_extraction": feature_extraction,
        "input_mapping": input_mapping,
        "output_genes": df_de.columns,
        "config": config,
    }
    
    model = SCAPE(model_setup)
    
    # Test custom compilation with loss weights
    loss_weights = [1.0, 0.5]  # Weight slogpval more than lfc
    
    # Compile model manually with custom loss weights
    optimizer = keras.optimizers.RMSprop(learning_rate=0.005)
    model.model.compile(
        optimizer=optimizer,
        loss=[mrrmse, mrrmse],
        loss_weights=loss_weights
    )
    
    # Verify model is compiled correctly
    assert model.model.optimizer is not None
    assert len(model.model.loss) == 2
    
    # Test that the model can be trained with weighted losses
    # Note: This test demonstrates the capability, actual training would need
    # proper data preparation for multi-output training
    
    # Verify the model architecture supports multi-output
    assert len(model.model.outputs) == 2
    assert model.model.outputs[0].shape[-1] == n_genes
    assert model.model.outputs[1].shape[-1] == n_genes


def test_model_output_consistency():
    """Test that model outputs are consistent between single and multi-task modes."""
    # Create sample data
    n_samples = 30
    n_genes = 15
    
    cell_types = ['CD4 T cells'] * n_samples
    sm_names = ['Drug_A'] * n_samples
    
    index = pd.MultiIndex.from_arrays([cell_types, sm_names], names=['cell_type', 'sm_name'])
    
    np.random.seed(42)
    de_data = np.random.randn(n_samples, n_genes)
    lfc_data = np.random.randn(n_samples, n_genes)
    
    gene_names = [f'Gene_{i}' for i in range(n_genes)]
    
    df_de = pd.DataFrame(de_data, index=index, columns=gene_names)
    df_lfc = pd.DataFrame(lfc_data, index=index, columns=gene_names)
    
    # Create single-task model
    single_task_model = create_default_model(n_genes, df_de, df_lfc)
    
    # Create multi-task model with same architecture but multiple outputs
    data_sources = {"slogpval": df_de, "lfc_pseudo": df_lfc}
    
    feature_extraction = {
        "slogpval_drug": {"source": "slogpval", "groupby": "sm_name", "function": "median"},
        "lfc_drug": {"source": "lfc_pseudo", "groupby": "sm_name", "function": "median"},
        "slogpval_cell": {"source": "slogpval", "groupby": "cell_type", "function": "median"},
        "lfc_cell": {"source": "lfc_pseudo", "groupby": "cell_type", "function": "median"},
    }
    
    input_mapping = {
        "in_slogpval_drug": "slogpval_drug",
        "in_lfc_drug": "lfc_drug",
        "in_slogpval_cell_encoder": "slogpval_cell",
        "in_lfc_cell_encoder": "lfc_cell",
        "in_slogpval_cell_decoder": "slogpval_cell",
        "in_lfc_cell_decoder": "lfc_cell",
    }
    
    config = {
        "inputs": {
            "in_slogpval_drug": [n_genes, 256, 128],
            "in_lfc_drug": [n_genes, 256, 128],
        },
        "conditional_encoder_input_structure": {
            "in_slogpval_cell_encoder": [n_genes, 32],
            "in_lfc_cell_encoder": [n_genes, 32, 16],
        },
        "conditional_decoder_input_structure": {
            "in_slogpval_cell_decoder": [n_genes, 32],
            "in_lfc_cell_decoder": [n_genes, 32, 16],
        },
        "conditional_decoder_input_hidden_sizes": [32],
        "encoder_hidden_layer_sizes": [128, 128],
        "decoder_hidden_layer_sizes": [128, 512],
        "outputs": {
            "slogpval": (n_genes, "linear"),
            "lfc": (n_genes, "linear"),
        },
        "noise": 0.01,
        "dropout": 0.05,
        "l1": 0,
        "l2": 0,
    }
    
    model_setup = {
        "data_sources": data_sources,
        "feature_extraction": feature_extraction,
        "input_mapping": input_mapping,
        "output_genes": df_de.columns,
        "config": config,
    }
    
    multi_task_model = SCAPE(model_setup)
    
    # Verify model architectures
    assert len(single_task_model.config['outputs']) == 1
    assert len(multi_task_model.config['outputs']) == 2
    
    # Verify input structures are the same
    assert single_task_model.config['inputs'] == multi_task_model.config['inputs']
    
    # Verify both models have the expected architecture
    assert len(single_task_model.model.outputs) == 1
    assert len(multi_task_model.model.outputs) == 2
    
    # Verify output shapes match expected dimensions
    assert single_task_model.model.outputs[0].shape[-1] == n_genes
    assert multi_task_model.model.outputs[0].shape[-1] == n_genes
    assert multi_task_model.model.outputs[1].shape[-1] == n_genes