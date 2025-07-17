"""
Test suite for ScAPE quick-start functionality.

This test file validates all the capabilities demonstrated in the quick-start guide:
- Data loading and preprocessing
- Gene selection and alignment
- Model configuration and setup
- Model training with cross-validation
- Model prediction for different input formats
- Model saving and loading
- Result visualization and analysis
- Baseline comparison functionality

Tests use the existing data in _data/ directory and create temporary files
for model saving/loading tests.
"""

import pytest
import tempfile
import os
import shutil
import pandas as pd
import numpy as np
from pathlib import Path

import scape
import scape._io as io
import scape._util as util
import scape._losses as losses


class TestDataLoading:
    """Test data loading and preprocessing functionality."""
    
    def setup_method(self):
        """Set up test data paths."""
        self.data_dir = Path(__file__).parent.parent / "_data"
        self.de_path = self.data_dir / "de_train.parquet"
        self.lfc_path = self.data_dir / "lfc_train.parquet"
        
    def test_data_files_exist(self):
        """Test that required data files exist."""
        assert self.de_path.exists(), f"DE data file not found: {self.de_path}"
        assert self.lfc_path.exists(), f"LFC data file not found: {self.lfc_path}"
        
    def test_load_slogpvals(self):
        """Test loading signed log p-values data."""
        df_de = io.load_slogpvals(str(self.de_path))
        
        # Check basic properties
        assert isinstance(df_de, pd.DataFrame)
        assert df_de.shape[0] > 0, "DE data should not be empty"
        assert df_de.shape[1] > 0, "DE data should have genes"
        
        # Check index structure (should be MultiIndex with cell_type and sm_name)
        assert isinstance(df_de.index, pd.MultiIndex)
        assert df_de.index.names == ['cell_type', 'sm_name']
        
        # Check that we have expected dimensions (shape may vary with dataset)
        assert df_de.shape[0] == 614, f"Expected 614 rows, got {df_de.shape[0]}"
        assert df_de.shape[1] > 18000, f"Expected > 18000 genes, got {df_de.shape[1]}"
        
    def test_load_lfc(self):
        """Test loading log fold change data."""
        df_lfc = io.load_lfc(str(self.lfc_path))
        
        # Check basic properties
        assert isinstance(df_lfc, pd.DataFrame)
        assert df_lfc.shape[0] > 0, "LFC data should not be empty"
        assert df_lfc.shape[1] > 0, "LFC data should have genes"
        
        # Check index structure
        assert isinstance(df_lfc.index, pd.MultiIndex)
        assert df_lfc.index.names == ['cell_type', 'sm_name']
        
        # Check that we have expected dimensions (shape may vary with dataset)
        assert df_lfc.shape[0] == 614, f"Expected 614 rows, got {df_lfc.shape[0]}"
        assert df_lfc.shape[1] > 18000, f"Expected > 18000 genes, got {df_lfc.shape[1]}"
        
    def test_data_alignment(self):
        """Test that DE and LFC data can be aligned."""
        df_de = io.load_slogpvals(str(self.de_path))
        df_lfc = io.load_lfc(str(self.lfc_path))
        
        # Align as in quick-start
        df_lfc_aligned = df_lfc.loc[df_de.index, df_de.columns]
        
        # Check alignment
        assert df_lfc_aligned.shape == df_de.shape
        assert (df_lfc_aligned.index == df_de.index).all()
        assert (df_lfc_aligned.columns == df_de.columns).all()


class TestGeneSelection:
    """Test gene selection and data preprocessing."""
    
    def setup_method(self):
        """Set up test data."""
        self.data_dir = Path(__file__).parent.parent / "_data"
        self.df_de = io.load_slogpvals(str(self.data_dir / "de_train.parquet"))
        self.df_lfc = io.load_lfc(str(self.data_dir / "lfc_train.parquet"))
        self.df_lfc = self.df_lfc.loc[self.df_de.index, self.df_de.columns]
        
    def test_select_top_variable_genes(self):
        """Test gene selection functionality."""
        n_genes = 64
        top_genes = util.select_top_variable([self.df_de], k=n_genes)
        
        # Check that we get the expected number of genes
        assert len(top_genes) == n_genes
        
        # Check that all genes are present in the original data
        assert all(gene in self.df_de.columns for gene in top_genes)
        
        # Check that genes are unique
        assert len(set(top_genes)) == n_genes
        
    def test_gene_selection_different_k(self):
        """Test gene selection with different k values."""
        for k in [16, 32, 64, 128]:
            top_genes = util.select_top_variable([self.df_de], k=k)
            assert len(top_genes) == k
            assert all(gene in self.df_de.columns for gene in top_genes)


class TestModelConfiguration:
    """Test model configuration and setup."""
    
    def setup_method(self):
        """Set up test data and configuration."""
        self.data_dir = Path(__file__).parent.parent / "_data"
        self.df_de = io.load_slogpvals(str(self.data_dir / "de_train.parquet"))
        self.df_lfc = io.load_lfc(str(self.data_dir / "lfc_train.parquet"))
        self.df_lfc = self.df_lfc.loc[self.df_de.index, self.df_de.columns]
        
        self.n_genes = 64
        self.top_genes = util.select_top_variable([self.df_de], k=self.n_genes)
        
        # Create model configuration as in quick-start
        self.data_sources = {
            'slogpval': self.df_de,
            'lfc_pseudo': self.df_lfc
        }
        
        self.feature_extraction = {
            'slogpval_drug': {
                'source': 'slogpval',
                'groupby': 'sm_name',
                'function': 'median'
            },
            'lfc_drug': {
                'source': 'lfc_pseudo',
                'groupby': 'sm_name',
                'function': 'median'
            },
            'slogpval_cell': {
                'source': 'slogpval',
                'groupby': 'cell_type',
                'function': 'median'
            },
            'lfc_cell': {
                'source': 'lfc_pseudo',
                'groupby': 'cell_type',
                'function': 'median'
            }
        }
        
        self.input_mapping = {
            'in_slogpval_drug': 'slogpval_drug',
            'in_lfc_drug': 'lfc_drug',
            'in_slogpval_cell_encoder': 'slogpval_cell',
            'in_lfc_cell_encoder': 'lfc_cell',
            'in_slogpval_cell_decoder': 'slogpval_cell',
            'in_lfc_cell_decoder': 'lfc_cell',
        }
        
        self.config = {
            "inputs": {
                "in_slogpval_drug": [self.n_genes, 256, 128],
                "in_lfc_drug": [self.n_genes, 256, 128],
            },
            "conditional_encoder_input_structure": {
                "in_slogpval_cell_encoder": [self.n_genes, 32],
                "in_lfc_cell_encoder": [self.n_genes, 32, 16],
            },
            "conditional_decoder_input_structure": {
                "in_slogpval_cell_decoder": [self.n_genes, 32],
                "in_lfc_cell_decoder": [self.n_genes, 32, 16],
            },
            "conditional_decoder_input_hidden_sizes": [32],
            "encoder_hidden_layer_sizes": [128, 128],
            "decoder_hidden_layer_sizes": [128, 512],
            "outputs": {
                "logpval": (self.df_de.shape[1], "linear"), 
            },
            "noise": 0.01,
            "dropout": 0.05,
            "l1": 0,
            "l2": 0
        }
        
    def test_model_setup_creation(self):
        """Test model setup dictionary creation."""
        model_setup = {
            "data_sources": self.data_sources,
            "feature_extraction": self.feature_extraction,
            "input_mapping": self.input_mapping,
            "output_genes": self.df_de.columns,
            "config": self.config
        }
        
        # Check all required keys are present
        required_keys = ["data_sources", "feature_extraction", "input_mapping", "output_genes", "config"]
        for key in required_keys:
            assert key in model_setup, f"Missing key: {key}"
            
        # Check data sources
        assert "slogpval" in model_setup["data_sources"]
        assert "lfc_pseudo" in model_setup["data_sources"]
        
        # Check feature extraction
        expected_features = ["slogpval_drug", "lfc_drug", "slogpval_cell", "lfc_cell"]
        for feature in expected_features:
            assert feature in model_setup["feature_extraction"]
            
        # Check input mapping
        expected_inputs = [
            "in_slogpval_drug", "in_lfc_drug", 
            "in_slogpval_cell_encoder", "in_lfc_cell_encoder",
            "in_slogpval_cell_decoder", "in_lfc_cell_decoder"
        ]
        for input_name in expected_inputs:
            assert input_name in model_setup["input_mapping"]
            
    def test_model_creation(self):
        """Test SCAPE model creation."""
        model_setup = {
            "data_sources": self.data_sources,
            "feature_extraction": self.feature_extraction,
            "input_mapping": self.input_mapping,
            "output_genes": self.df_de.columns,
            "config": self.config
        }
        
        # Create model
        scm = scape.SCAPE(model_setup)
        
        # Check that model is created
        assert scm is not None
        assert hasattr(scm, 'model')
        assert scm.model is not None
        
        # Check model inputs
        expected_inputs = [
            "in_slogpval_drug", "in_lfc_drug",
            "in_slogpval_cell_encoder", "in_lfc_cell_encoder", 
            "in_slogpval_cell_decoder", "in_lfc_cell_decoder"
        ]
        model_input_names = [inp.name for inp in scm.model.inputs]
        for expected_input in expected_inputs:
            assert expected_input in model_input_names


class TestModelTraining:
    """Test model training functionality."""
    
    def setup_method(self):
        """Set up test data and model."""
        self.data_dir = Path(__file__).parent.parent / "_data"
        self.df_de = io.load_slogpvals(str(self.data_dir / "de_train.parquet"))
        self.df_lfc = io.load_lfc(str(self.data_dir / "lfc_train.parquet"))
        self.df_lfc = self.df_lfc.loc[self.df_de.index, self.df_de.columns]
        
        self.n_genes = 64
        self.top_genes = util.select_top_variable([self.df_de], k=self.n_genes)
        
        # Create model setup
        data_sources = {
            'slogpval': self.df_de,
            'lfc_pseudo': self.df_lfc
        }
        
        feature_extraction = {
            'slogpval_drug': {'source': 'slogpval', 'groupby': 'sm_name', 'function': 'median'},
            'lfc_drug': {'source': 'lfc_pseudo', 'groupby': 'sm_name', 'function': 'median'},
            'slogpval_cell': {'source': 'slogpval', 'groupby': 'cell_type', 'function': 'median'},
            'lfc_cell': {'source': 'lfc_pseudo', 'groupby': 'cell_type', 'function': 'median'}
        }
        
        input_mapping = {
            'in_slogpval_drug': 'slogpval_drug',
            'in_lfc_drug': 'lfc_drug',
            'in_slogpval_cell_encoder': 'slogpval_cell',
            'in_lfc_cell_encoder': 'lfc_cell',
            'in_slogpval_cell_decoder': 'slogpval_cell',
            'in_lfc_cell_decoder': 'lfc_cell',
        }
        
        config = {
            "inputs": {
                "in_slogpval_drug": [self.n_genes, 256, 128],
                "in_lfc_drug": [self.n_genes, 256, 128],
            },
            "conditional_encoder_input_structure": {
                "in_slogpval_cell_encoder": [self.n_genes, 32],
                "in_lfc_cell_encoder": [self.n_genes, 32, 16],
            },
            "conditional_decoder_input_structure": {
                "in_slogpval_cell_decoder": [self.n_genes, 32],
                "in_lfc_cell_decoder": [self.n_genes, 32, 16],
            },
            "conditional_decoder_input_hidden_sizes": [32],
            "encoder_hidden_layer_sizes": [128, 128],
            "decoder_hidden_layer_sizes": [128, 512],
            "outputs": {
                "logpval": (self.df_de.shape[1], "linear"), 
            },
            "noise": 0.01,
            "dropout": 0.05,
            "l1": 0,
            "l2": 0
        }
        
        self.model_setup = {
            "data_sources": data_sources,
            "feature_extraction": feature_extraction,
            "input_mapping": input_mapping,
            "output_genes": self.df_de.columns,
            "config": config
        }
        
    def test_model_training_basic(self):
        """Test basic model training functionality."""
        scm = scape.SCAPE(self.model_setup)
        
        # Train for just a few epochs to test functionality
        result = scm.train(
            val_cells=["NK cells"], 
            val_drugs=["Prednisolone"],
            input_columns=self.top_genes,
            epochs=3,
            baselines=["zero", "slogpval_drug"]
        )
        
        # Check that result is returned
        assert result is not None
        assert isinstance(result, dict)
        
        # Check that training metrics are present
        assert "history" in result
        assert "baselines" in result
        assert isinstance(result["history"], pd.DataFrame)
        
        # Check that history contains training information
        assert "loss" in result["history"].columns
        assert "val_loss" in result["history"].columns
        assert len(result["history"]) > 0
        
    def test_model_training_with_baselines(self):
        """Test model training with baseline comparisons."""
        scm = scape.SCAPE(self.model_setup)
        
        result = scm.train(
            val_cells=["NK cells"], 
            val_drugs=["Prednisolone"],
            input_columns=self.top_genes,
            epochs=3,
            baselines=["zero", "slogpval_drug"]
        )
        
        # Check that baseline information is present if baselines were requested
        if "baselines" in result:
            assert "zero" in result["baselines"]
            assert "slogpval_drug" in result["baselines"]
            
            # Check that baseline values are reasonable
            assert result["baselines"]["zero"] > 0
            assert result["baselines"]["slogpval_drug"] > 0
        
    def test_model_training_different_validation_sets(self):
        """Test training with different validation sets."""
        scm = scape.SCAPE(self.model_setup)
        
        # Test with different cell types
        result1 = scm.train(
            val_cells=["T cells CD4+"], 
            val_drugs=["Prednisolone"],
            input_columns=self.top_genes,
            epochs=2
        )
        
        scm2 = scape.SCAPE(self.model_setup)
        result2 = scm2.train(
            val_cells=["T cells CD8+"], 
            val_drugs=["Prednisolone"],
            input_columns=self.top_genes,
            epochs=2
        )
        
        # Results should be different due to different validation sets
        assert result1 is not None
        assert result2 is not None
        assert "history" in result1
        assert "history" in result2
        
        # Test with different drugs
        scm3 = scape.SCAPE(self.model_setup)
        result3 = scm3.train(
            val_cells=["NK cells"], 
            val_drugs=["Belinostat"],
            input_columns=self.top_genes,
            epochs=2
        )
        
        assert result3 is not None
        assert "history" in result3


class TestModelPrediction:
    """Test model prediction functionality."""
    
    def setup_method(self):
        """Set up trained model for prediction tests."""
        self.data_dir = Path(__file__).parent.parent / "_data"
        self.df_de = io.load_slogpvals(str(self.data_dir / "de_train.parquet"))
        self.df_lfc = io.load_lfc(str(self.data_dir / "lfc_train.parquet"))
        self.df_lfc = self.df_lfc.loc[self.df_de.index, self.df_de.columns]
        
        self.n_genes = 64
        self.top_genes = util.select_top_variable([self.df_de], k=self.n_genes)
        
        # Create and train model
        data_sources = {
            'slogpval': self.df_de,
            'lfc_pseudo': self.df_lfc
        }
        
        feature_extraction = {
            'slogpval_drug': {'source': 'slogpval', 'groupby': 'sm_name', 'function': 'median'},
            'lfc_drug': {'source': 'lfc_pseudo', 'groupby': 'sm_name', 'function': 'median'},
            'slogpval_cell': {'source': 'slogpval', 'groupby': 'cell_type', 'function': 'median'},
            'lfc_cell': {'source': 'lfc_pseudo', 'groupby': 'cell_type', 'function': 'median'}
        }
        
        input_mapping = {
            'in_slogpval_drug': 'slogpval_drug',
            'in_lfc_drug': 'lfc_drug',
            'in_slogpval_cell_encoder': 'slogpval_cell',
            'in_lfc_cell_encoder': 'lfc_cell',
            'in_slogpval_cell_decoder': 'slogpval_cell',
            'in_lfc_cell_decoder': 'lfc_cell',
        }
        
        config = {
            "inputs": {
                "in_slogpval_drug": [self.n_genes, 256, 128],
                "in_lfc_drug": [self.n_genes, 256, 128],
            },
            "conditional_encoder_input_structure": {
                "in_slogpval_cell_encoder": [self.n_genes, 32],
                "in_lfc_cell_encoder": [self.n_genes, 32, 16],
            },
            "conditional_decoder_input_structure": {
                "in_slogpval_cell_decoder": [self.n_genes, 32],
                "in_lfc_cell_decoder": [self.n_genes, 32, 16],
            },
            "conditional_decoder_input_hidden_sizes": [32],
            "encoder_hidden_layer_sizes": [128, 128],
            "decoder_hidden_layer_sizes": [128, 512],
            "outputs": {
                "logpval": (self.df_de.shape[1], "linear"), 
            },
            "noise": 0.01,
            "dropout": 0.05,
            "l1": 0,
            "l2": 0
        }
        
        model_setup = {
            "data_sources": data_sources,
            "feature_extraction": feature_extraction,
            "input_mapping": input_mapping,
            "output_genes": self.df_de.columns,
            "config": config
        }
        
        self.scm = scape.SCAPE(model_setup)
        # Train for minimal epochs just to get a working model
        self.scm.train(
            val_cells=["NK cells"], 
            val_drugs=["Prednisolone"],
            input_columns=self.top_genes,
            epochs=2
        )
        
    def test_predict_from_dataframe(self):
        """Test prediction from dataframe index."""
        # Predict using the full dataframe
        predictions = self.scm.predict(self.df_de)
        
        # Check prediction shape and type
        assert isinstance(predictions, pd.DataFrame)
        assert predictions.shape[0] == self.df_de.shape[0]
        assert predictions.shape[1] == self.df_de.shape[1]
        
        # Check that predictions have same index and columns
        assert (predictions.index == self.df_de.index).all()
        assert (predictions.columns == self.df_de.columns).all()
        
        # Check that predictions are numeric
        assert predictions.dtypes.apply(lambda x: pd.api.types.is_numeric_dtype(x)).all()
        
    def test_predict_from_multiindex(self):
        """Test prediction from MultiIndex."""
        # Create test MultiIndex
        test_index = pd.MultiIndex.from_tuples([
            ("NK cells", "Bosutinib"),
            ("NK cells", "Belinostat"),
            ("T cells CD4+", "Bosutinib"),
            ("T cells CD4+", "Belinostat"),
        ], names=["cell_type", "sm_name"])
        
        predictions = self.scm.predict(test_index)
        
        # Check prediction properties
        assert isinstance(predictions, pd.DataFrame)
        assert predictions.shape[0] == 4
        assert predictions.shape[1] == self.df_de.shape[1]
        
        # Check that predictions have correct index
        assert (predictions.index == test_index).all()
        assert predictions.index.names == ["cell_type", "sm_name"]
        
    def test_predict_from_tuples(self):
        """Test prediction from list of tuples."""
        # Test with tuple list
        test_tuples = [("NK cells", "Bosutinib"), ("NK cells", "Belinostat")]
        predictions = self.scm.predict(test_tuples)
        
        # Check prediction properties
        assert isinstance(predictions, pd.DataFrame)
        assert predictions.shape[0] == 2
        assert predictions.shape[1] == self.df_de.shape[1]
        
        # Check that predictions have correct index
        assert predictions.index.names == ["cell_type", "sm_name"]
        expected_index = pd.MultiIndex.from_tuples(test_tuples, names=["cell_type", "sm_name"])
        assert (predictions.index == expected_index).all()
        
    def test_predict_single_combination(self):
        """Test prediction for a single cell-drug combination."""
        # Test prediction for a single combination
        predictions = self.scm.predict([("NK cells", "Prednisolone")])
        
        assert isinstance(predictions, pd.DataFrame)
        assert predictions.shape[0] == 1
        assert predictions.shape[1] == self.df_de.shape[1]
        
        # Check that values are reasonable (not all zeros or NaN)
        assert not predictions.isna().all().all()
        assert not (predictions == 0).all().all()


class TestModelSaveLoad:
    """Test model saving and loading functionality."""
    
    def setup_method(self):
        """Set up model for save/load tests."""
        self.data_dir = Path(__file__).parent.parent / "_data"
        self.df_de = io.load_slogpvals(str(self.data_dir / "de_train.parquet"))
        self.df_lfc = io.load_lfc(str(self.data_dir / "lfc_train.parquet"))
        self.df_lfc = self.df_lfc.loc[self.df_de.index, self.df_de.columns]
        
        self.n_genes = 64
        self.top_genes = util.select_top_variable([self.df_de], k=self.n_genes)
        
        # Create model setup
        data_sources = {
            'slogpval': self.df_de,
            'lfc_pseudo': self.df_lfc
        }
        
        feature_extraction = {
            'slogpval_drug': {'source': 'slogpval', 'groupby': 'sm_name', 'function': 'median'},
            'lfc_drug': {'source': 'lfc_pseudo', 'groupby': 'sm_name', 'function': 'median'},
            'slogpval_cell': {'source': 'slogpval', 'groupby': 'cell_type', 'function': 'median'},
            'lfc_cell': {'source': 'lfc_pseudo', 'groupby': 'cell_type', 'function': 'median'}
        }
        
        input_mapping = {
            'in_slogpval_drug': 'slogpval_drug',
            'in_lfc_drug': 'lfc_drug',
            'in_slogpval_cell_encoder': 'slogpval_cell',
            'in_lfc_cell_encoder': 'lfc_cell',
            'in_slogpval_cell_decoder': 'slogpval_cell',
            'in_lfc_cell_decoder': 'lfc_cell',
        }
        
        config = {
            "inputs": {
                "in_slogpval_drug": [self.n_genes, 256, 128],
                "in_lfc_drug": [self.n_genes, 256, 128],
            },
            "conditional_encoder_input_structure": {
                "in_slogpval_cell_encoder": [self.n_genes, 32],
                "in_lfc_cell_encoder": [self.n_genes, 32, 16],
            },
            "conditional_decoder_input_structure": {
                "in_slogpval_cell_decoder": [self.n_genes, 32],
                "in_lfc_cell_decoder": [self.n_genes, 32, 16],
            },
            "conditional_decoder_input_hidden_sizes": [32],
            "encoder_hidden_layer_sizes": [128, 128],
            "decoder_hidden_layer_sizes": [128, 512],
            "outputs": {
                "logpval": (self.df_de.shape[1], "linear"), 
            },
            "noise": 0.01,
            "dropout": 0.05,
            "l1": 0,
            "l2": 0
        }
        
        self.model_setup = {
            "data_sources": data_sources,
            "feature_extraction": feature_extraction,
            "input_mapping": input_mapping,
            "output_genes": self.df_de.columns,
            "config": config
        }
        
        # Create temporary directory for test files
        self.temp_dir = tempfile.mkdtemp()
        
    def teardown_method(self):
        """Clean up temporary files."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
        
    def test_save_and_load_model(self):
        """Test saving and loading a trained model."""
        # Create and train model
        scm = scape.SCAPE(self.model_setup)
        result = scm.train(
            val_cells=["NK cells"], 
            val_drugs=["Prednisolone"],
            input_columns=self.top_genes,
            epochs=2
        )
        
        # Define file paths
        config_path = os.path.join(self.temp_dir, "config.pkl")
        weights_path = os.path.join(self.temp_dir, "model.weights.h5")
        result_path = os.path.join(self.temp_dir, "result.pkl")
        
        # Save model
        scm.save(config_path, weights_path, result_path)
        
        # Check that files were created
        assert os.path.exists(config_path)
        assert os.path.exists(weights_path)
        assert os.path.exists(result_path)
        
        # Load model
        loaded_scm = scape.SCAPE.load(config_path, weights_path, result_path)
        
        # Test that loaded model works
        assert loaded_scm is not None
        
        # Test prediction with loaded model
        original_pred = scm.predict([("NK cells", "Prednisolone")])
        loaded_pred = loaded_scm.predict([("NK cells", "Prednisolone")])
        
        # Predictions should be identical (or very close due to numerical precision)
        pd.testing.assert_frame_equal(original_pred, loaded_pred, rtol=1e-5)
        
    def test_save_during_training(self):
        """Test saving model during training process."""
        scm = scape.SCAPE(self.model_setup)
        
        # Define file paths
        config_path = os.path.join(self.temp_dir, "config.pkl")
        model_path = os.path.join(self.temp_dir, "model.keras")
        result_path = os.path.join(self.temp_dir, "result.pkl")
        
        # Train with automatic saving
        result = scm.train(
            val_cells=["NK cells"], 
            val_drugs=["Prednisolone"],
            input_columns=self.top_genes,
            epochs=2,
            output_folder=self.temp_dir,
            config_file_name="config.pkl",
            model_file_name="model.keras",
            result_file_name="result.pkl"
        )
        
        # Check that files were created during training
        assert os.path.exists(config_path)
        assert os.path.exists(result_path)
        # Note: model might be saved as .keras or .weights.h5 format
        
        # Check that result contains expected information
        assert result is not None
        assert "train_losses" in result
        assert "val_losses" in result


class TestResultAnalysis:
    """Test result analysis and visualization functionality."""
    
    def setup_method(self):
        """Set up trained model for analysis tests."""
        self.data_dir = Path(__file__).parent.parent / "_data"
        self.df_de = io.load_slogpvals(str(self.data_dir / "de_train.parquet"))
        self.df_lfc = io.load_lfc(str(self.data_dir / "lfc_train.parquet"))
        self.df_lfc = self.df_lfc.loc[self.df_de.index, self.df_de.columns]
        
        self.n_genes = 64
        self.top_genes = util.select_top_variable([self.df_de], k=self.n_genes)
        
        # Create and train model
        data_sources = {
            'slogpval': self.df_de,
            'lfc_pseudo': self.df_lfc
        }
        
        feature_extraction = {
            'slogpval_drug': {'source': 'slogpval', 'groupby': 'sm_name', 'function': 'median'},
            'lfc_drug': {'source': 'lfc_pseudo', 'groupby': 'sm_name', 'function': 'median'},
            'slogpval_cell': {'source': 'slogpval', 'groupby': 'cell_type', 'function': 'median'},
            'lfc_cell': {'source': 'lfc_pseudo', 'groupby': 'cell_type', 'function': 'median'}
        }
        
        input_mapping = {
            'in_slogpval_drug': 'slogpval_drug',
            'in_lfc_drug': 'lfc_drug',
            'in_slogpval_cell_encoder': 'slogpval_cell',
            'in_lfc_cell_encoder': 'lfc_cell',
            'in_slogpval_cell_decoder': 'slogpval_cell',
            'in_lfc_cell_decoder': 'lfc_cell',
        }
        
        config = {
            "inputs": {
                "in_slogpval_drug": [self.n_genes, 256, 128],
                "in_lfc_drug": [self.n_genes, 256, 128],
            },
            "conditional_encoder_input_structure": {
                "in_slogpval_cell_encoder": [self.n_genes, 32],
                "in_lfc_cell_encoder": [self.n_genes, 32, 16],
            },
            "conditional_decoder_input_structure": {
                "in_slogpval_cell_decoder": [self.n_genes, 32],
                "in_lfc_cell_decoder": [self.n_genes, 32, 16],
            },
            "conditional_decoder_input_hidden_sizes": [32],
            "encoder_hidden_layer_sizes": [128, 128],
            "decoder_hidden_layer_sizes": [128, 512],
            "outputs": {
                "logpval": (self.df_de.shape[1], "linear"), 
            },
            "noise": 0.01,
            "dropout": 0.05,
            "l1": 0,
            "l2": 0
        }
        
        model_setup = {
            "data_sources": data_sources,
            "feature_extraction": feature_extraction,
            "input_mapping": input_mapping,
            "output_genes": self.df_de.columns,
            "config": config
        }
        
        self.scm = scape.SCAPE(model_setup)
        self.result = self.scm.train(
            val_cells=["NK cells"], 
            val_drugs=["Prednisolone"],
            input_columns=self.top_genes,
            epochs=3,
            baselines=["zero", "slogpval_drug"]
        )
        
    def test_prediction_correlation_analysis(self):
        """Test correlation analysis between predicted and observed values."""
        # Get true values for a specific combination
        true_values = self.df_de.loc[("NK cells", "Prednisolone")]
        
        # Get predictions for the same combination
        pred_values = self.scm.predict([("NK cells", "Prednisolone")])
        
        # Check that both have same length
        assert len(true_values) == len(pred_values.iloc[0])
        
        # Create comparison dataframe
        df_cmp = pd.DataFrame({
            'y': true_values.values,
            'y_pred': pred_values.iloc[0].values
        })
        
        # Check that dataframe is created correctly
        assert isinstance(df_cmp, pd.DataFrame)
        assert 'y' in df_cmp.columns
        assert 'y_pred' in df_cmp.columns
        assert len(df_cmp) == len(true_values)
        
        # Check that neither column is all NaN
        assert not df_cmp['y'].isna().all()
        assert not df_cmp['y_pred'].isna().all()
        
        # Calculate correlation (should be positive for a reasonable model)
        correlation = df_cmp['y'].corr(df_cmp['y_pred'])
        assert not np.isnan(correlation), "Correlation should not be NaN"
        
    def test_mrrmse_calculation(self):
        """Test MRRMSE calculation functionality."""
        # Get true and predicted values
        true_values = self.df_de.loc[("NK cells", "Prednisolone")]
        pred_values = self.scm.predict([("NK cells", "Prednisolone")])
        
        # Calculate MRRMSE
        mrrmse = losses.np_mrrmse(true_values, pred_values)
        
        # Check that MRRMSE is a reasonable value
        assert isinstance(mrrmse, (float, np.floating))
        assert mrrmse > 0, "MRRMSE should be positive"
        assert not np.isnan(mrrmse), "MRRMSE should not be NaN"
        
        # MRRMSE should be finite
        assert np.isfinite(mrrmse), "MRRMSE should be finite"


class TestBaselineComparison:
    """Test baseline comparison functionality."""
    
    def setup_method(self):
        """Set up model for baseline testing."""
        self.data_dir = Path(__file__).parent.parent / "_data"
        self.df_de = io.load_slogpvals(str(self.data_dir / "de_train.parquet"))
        self.df_lfc = io.load_lfc(str(self.data_dir / "lfc_train.parquet"))
        self.df_lfc = self.df_lfc.loc[self.df_de.index, self.df_de.columns]
        
        self.n_genes = 64
        self.top_genes = util.select_top_variable([self.df_de], k=self.n_genes)
        
    def test_baseline_calculation(self):
        """Test that baselines are calculated correctly."""
        # Create model
        data_sources = {
            'slogpval': self.df_de,
            'lfc_pseudo': self.df_lfc
        }
        
        feature_extraction = {
            'slogpval_drug': {'source': 'slogpval', 'groupby': 'sm_name', 'function': 'median'},
            'lfc_drug': {'source': 'lfc_pseudo', 'groupby': 'sm_name', 'function': 'median'},
            'slogpval_cell': {'source': 'slogpval', 'groupby': 'cell_type', 'function': 'median'},
            'lfc_cell': {'source': 'lfc_pseudo', 'groupby': 'cell_type', 'function': 'median'}
        }
        
        input_mapping = {
            'in_slogpval_drug': 'slogpval_drug',
            'in_lfc_drug': 'lfc_drug',
            'in_slogpval_cell_encoder': 'slogpval_cell',
            'in_lfc_cell_encoder': 'lfc_cell',
            'in_slogpval_cell_decoder': 'slogpval_cell',
            'in_lfc_cell_decoder': 'lfc_cell',
        }
        
        config = {
            "inputs": {
                "in_slogpval_drug": [self.n_genes, 256, 128],
                "in_lfc_drug": [self.n_genes, 256, 128],
            },
            "conditional_encoder_input_structure": {
                "in_slogpval_cell_encoder": [self.n_genes, 32],
                "in_lfc_cell_encoder": [self.n_genes, 32, 16],
            },
            "conditional_decoder_input_structure": {
                "in_slogpval_cell_decoder": [self.n_genes, 32],
                "in_lfc_cell_decoder": [self.n_genes, 32, 16],
            },
            "conditional_decoder_input_hidden_sizes": [32],
            "encoder_hidden_layer_sizes": [128, 128],
            "decoder_hidden_layer_sizes": [128, 512],
            "outputs": {
                "logpval": (self.df_de.shape[1], "linear"), 
            },
            "noise": 0.01,
            "dropout": 0.05,
            "l1": 0,
            "l2": 0
        }
        
        model_setup = {
            "data_sources": data_sources,
            "feature_extraction": feature_extraction,
            "input_mapping": input_mapping,
            "output_genes": self.df_de.columns,
            "config": config
        }
        
        scm = scape.SCAPE(model_setup)
        
        # Train with baselines
        result = scm.train(
            val_cells=["NK cells"], 
            val_drugs=["Prednisolone"],
            input_columns=self.top_genes,
            epochs=2,
            baselines=["zero", "slogpval_drug"]
        )
        
        # Check that baselines are present and reasonable
        assert "baselines" in result
        assert "zero" in result["baselines"]
        assert "slogpval_drug" in result["baselines"]
        
        # Check baseline values
        zero_baseline = result["baselines"]["zero"]
        drug_baseline = result["baselines"]["slogpval_drug"]
        
        assert zero_baseline > 0
        assert drug_baseline > 0
        assert not np.isnan(zero_baseline)
        assert not np.isnan(drug_baseline)
        
        # Zero baseline should typically be worse than drug baseline
        # (higher MRRMSE means worse performance)
        assert zero_baseline >= drug_baseline
        
    def test_different_baselines(self):
        """Test training with different baseline configurations."""
        # Create model setup (using minimal config for speed)
        data_sources = {
            'slogpval': self.df_de,
            'lfc_pseudo': self.df_lfc
        }
        
        feature_extraction = {
            'slogpval_drug': {'source': 'slogpval', 'groupby': 'sm_name', 'function': 'median'},
            'lfc_drug': {'source': 'lfc_pseudo', 'groupby': 'sm_name', 'function': 'median'},
            'slogpval_cell': {'source': 'slogpval', 'groupby': 'cell_type', 'function': 'median'},
            'lfc_cell': {'source': 'lfc_pseudo', 'groupby': 'cell_type', 'function': 'median'}
        }
        
        input_mapping = {
            'in_slogpval_drug': 'slogpval_drug',
            'in_lfc_drug': 'lfc_drug',
            'in_slogpval_cell_encoder': 'slogpval_cell',
            'in_lfc_cell_encoder': 'lfc_cell',
            'in_slogpval_cell_decoder': 'slogpval_cell',
            'in_lfc_cell_decoder': 'lfc_cell',
        }
        
        config = {
            "inputs": {
                "in_slogpval_drug": [self.n_genes, 256, 128],
                "in_lfc_drug": [self.n_genes, 256, 128],
            },
            "conditional_encoder_input_structure": {
                "in_slogpval_cell_encoder": [self.n_genes, 32],
                "in_lfc_cell_encoder": [self.n_genes, 32, 16],
            },
            "conditional_decoder_input_structure": {
                "in_slogpval_cell_decoder": [self.n_genes, 32],
                "in_lfc_cell_decoder": [self.n_genes, 32, 16],
            },
            "conditional_decoder_input_hidden_sizes": [32],
            "encoder_hidden_layer_sizes": [128, 128],
            "decoder_hidden_layer_sizes": [128, 512],
            "outputs": {
                "logpval": (self.df_de.shape[1], "linear"), 
            },
            "noise": 0.01,
            "dropout": 0.05,
            "l1": 0,
            "l2": 0
        }
        
        model_setup = {
            "data_sources": data_sources,
            "feature_extraction": feature_extraction,
            "input_mapping": input_mapping,
            "output_genes": self.df_de.columns,
            "config": config
        }
        
        # Test with only zero baseline
        scm1 = scape.SCAPE(model_setup)
        result1 = scm1.train(
            val_cells=["NK cells"], 
            val_drugs=["Prednisolone"],
            input_columns=self.top_genes,
            epochs=2,
            baselines=["zero"]
        )
        
        assert "baselines" in result1
        assert "zero" in result1["baselines"]
        assert "slogpval_drug" not in result1["baselines"]
        
        # Test with only drug baseline
        scm2 = scape.SCAPE(model_setup)
        result2 = scm2.train(
            val_cells=["NK cells"], 
            val_drugs=["Prednisolone"],
            input_columns=self.top_genes,
            epochs=2,
            baselines=["slogpval_drug"]
        )
        
        assert "baselines" in result2
        assert "slogpval_drug" in result2["baselines"]
        assert "zero" not in result2["baselines"]
        
        # Test with no baselines
        scm3 = scape.SCAPE(model_setup)
        result3 = scm3.train(
            val_cells=["NK cells"], 
            val_drugs=["Prednisolone"],
            input_columns=self.top_genes,
            epochs=2
        )
        
        # Should still work without baselines
        assert result3 is not None
        assert "train_losses" in result3
        assert "val_losses" in result3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])