from tensorflow import keras
from scape._losses import mrrmse, np_mrrmse
from scape._util import extract_features, split_data
from scape._callbacks import MonitorCallback
from keras.layers import (
    Input,
    Dense,
    Concatenate,
    Dropout,
    BatchNormalization,
    GaussianNoise,
)
import os
import pandas as pd
import pickle


def _v(dfs):
    return [df.values for df in dfs]


class SCAPE:
    """
    A class to encapsulate functionalities for model training, evaluation, and prediction.

    Attributes
    ----------
    setup : dict
        A dictionary containing configuration and setup data.
    config : dict
        A subset of the `setup` dictionary, containing model-specific configurations.
    model : keras.Model
        The neural network model defined as per the given configuration.
    _last_train_results : dict, optional
        A dictionary storing the results from the last training session.

    Methods
    -------
    train(val_cells=None, val_drugs=None, input_columns=None, output_data="slogpval",
          callbacks="default", optimizer=None, epochs=600, batch_size=128,
          output_folder=None, model_file_name="model.keras", result_file_name=None,
          config_file_name=None, baselines=["zero"])
        Trains the model with the provided parameters and data.

    generate_inputs(idx_targets, names=["cell_type", "sm_name"], source_index=None,
                    source_columns=None)
        Generates input data for the model prediction from the given indices.

    load(config_file, weights_file, results_file)
        Static method to load a trained model from files.

    save(config_file, weights_file, results_file)
        Saves the model configuration, weights, and training results to files.

    predict(idx_targets, as_df=True, output_idx=0, idx_target_names=None)
        Predicts outputs using the model for the provided target indices.
    """

    def __init__(self, setup):
        self.setup = setup
        self.config = setup["config"]
        model, _, _ = create_model(setup["config"])
        self.model = model
        self._last_train_results = None

    def train(
        self,
        val_cells=None,
        val_drugs=None,
        input_columns=None,
        output_data="slogpval",
        callbacks="default",
        optimizer=None,
        epochs=600,
        batch_size=128,
        output_folder=None,
        model_file_name="model.keras",
        result_file_name=None,
        config_file_name=None,
        baselines=["zero"],
    ):
        """
        Trains the model based on the provided parameters and data.

        Parameters
        ----------
        val_cells : array-like, optional
            Validation data for cell lines.
        val_drugs : array-like, optional
            Validation data for drugs.
        input_columns : list of str, optional
            Specific columns to be used as input features.
        output_data : str, optional
            The key for the output data in the setup dictionary.
        callbacks : list of keras.callbacks.Callback or str, optional
            Callbacks to use during training. If 'default', uses predefined callbacks.
        optimizer : keras.optimizers.Optimizer, optional
            The optimizer to use for training.
        epochs : int, optional
            Number of epochs to train the model.
        batch_size : int, optional
            Size of the batches used in the training process.
        output_folder : str, optional
            Directory where to save the model and results.
        model_file_name : str, optional
            File name to save the trained model.
        result_file_name : str, optional
            File name to save training results.
        config_file_name : str, optional
            File name to save the model configuration.
        baselines : list of str, optional
            List of baseline methods to compare the model's performance against.

        Returns
        -------
        dict
            A dictionary containing training results and model information.
        """
        if isinstance(output_data, str):
            output_data = self.setup["data_sources"][output_data]
        input_mapping = self.setup["input_mapping"]
        ix_train, ix_val = split_data(output_data, values=[val_cells, val_drugs])
        data_source_index = ix_train
        data_source_columns = output_data.columns.tolist()
        if input_columns is not None:
            data_source_columns = input_columns

        # Compute features on the training set
        feats_train, feats_val = extract_features(
            self.setup["data_sources"],
            self.setup["feature_extraction"],
            data_source_index=data_source_index,
            join_on_index=[ix_train, ix_val],
        )

        Y_train = output_data.loc[ix_train, :]
        Y_val = output_data.loc[ix_val, :]

        # Compute baselines (0s and median)
        _baselines = {}
        for b in baselines:
            if b == "zero":
                _baselines[b] = np_mrrmse(Y_val, Y_val * 0.0)
            else:
                _baselines[b] = np_mrrmse(Y_val, feats_val[b].values)
            print(f"Baseline {b} MRRMSE: {_baselines[b]:.4f}")

        print(f"Model size (MB): {(self.model.count_params() * 4 / 1024**2):.2f}")

        if optimizer is None:
            optimizer = keras.optimizers.RMSprop(learning_rate=0.005)

        self.model.compile(optimizer=optimizer, loss=mrrmse)
        # Print model input names:
        print("Model inputs:", self.model.input_names)

        # Get train inputs from features, using the order and names of the inputs of the model
        # and the mapping between inputs and features
        train_inputs = [
            feats_train[input_mapping[k]].loc[:, data_source_columns]
            for k in self.model.input_names
        ]
        val_inputs = [
            feats_val[input_mapping[k]].loc[:, data_source_columns]
            for k in self.model.input_names
        ]
        train_outputs = [Y_train]
        val_outputs = [Y_val]

        callback_list = callbacks if not isinstance(callbacks, str) else []
        if callbacks == "default":
            from keras.callbacks import ReduceLROnPlateau

            reduce_lr = ReduceLROnPlateau(
                monitor="loss",
                factor=0.85,
                patience=12,
                min_lr=1e-5,
                verbose=0,
                mode="min",
            )
            callback_list.append(reduce_lr)
            if output_folder is not None:
                if not os.path.exists(output_folder):
                    os.makedirs(output_folder)
                model_file = os.path.join(output_folder, model_file_name)
                if os.path.exists(model_file):
                    raise ValueError(f"Model file {model_file} already exists.")
                model_filepath = os.path.join(output_folder, model_file_name)
                print(f"Model will be saved to {model_filepath}")
                callback_list.append(checkpoint(model_filepath))
            monitor_callback = MonitorCallback(
                monitor="val_loss", baseline_errors=_baselines
            )
            callback_list.append(monitor_callback)
            # Create Early Stop to restore weights
            early_stop = keras.callbacks.EarlyStopping(
                monitor="val_loss",
                mode="min",
                patience=epochs,
                restore_best_weights=True,
            )
            callback_list.append(early_stop)

        history = self.model.fit(
            _v(train_inputs),
            _v(train_outputs),
            epochs=epochs,
            batch_size=batch_size,
            shuffle=True,
            verbose=0,
            callbacks=callback_list,
            validation_data=(_v(val_inputs), _v(val_outputs)),
        )

        self._last_train_results = {
            "history": pd.DataFrame(history.history),
            "ix_train": ix_train,
            "ix_val": ix_val,
            "data_source_index": data_source_index,
            "data_source_columns": data_source_columns,
            "baselines": _baselines,
            "input_names": self.model.input_names,
            "model_file": model_file_name,
            "drugs": val_drugs,
            "cells": val_cells,
        }
        if output_folder is not None:
            model_file_name = model_file_name.split(".")[0]
            result_file = (
                f"{model_file_name}_result.pkl"
                if result_file_name is None
                else result_file_name
            )
            config_file = (
                f"{model_file_name}_config.pkl"
                if config_file_name is None
                else config_file_name
            )
            with open(os.path.join(output_folder, result_file), "wb") as f:
                pickle.dump(self._last_train_results, f)
            with open(os.path.join(output_folder, config_file), "wb") as f:
                pickle.dump(self.setup, f)
        return self._last_train_results

    def generate_inputs(
        self,
        idx_targets,
        names=["cell_type", "sm_name"],
        source_index=None,
        source_columns=None,
    ):
        """
        Generates input data for the model prediction from the given indices.

        This method prepares the input features for prediction based on specified indices,
        using data source configurations and mappings defined in the setup.

        Parameters
        ----------
        idx_targets : array-like, pandas.DataFrame, pandas.Series, or list
            The target indices for which to generate inputs. Can be an array, a DataFrame,
            a Series, or a list of tuples.
        names : list of str, optional
            Names for the indices, used when `idx_targets` is a list of tuples.
            Defaults to ["cell_type", "sm_name"].
        source_index : pandas.Index or None, optional
            The index of the data source to use. If None, uses the entire data source.
        source_columns : list of str or None, optional
            Specific columns to be used for generating inputs. If None, uses all columns.

        Returns
        -------
        list
            A list of numpy arrays corresponding to the inputs for the model, structured
            according to the model's input names and the mapping defined in the setup.
        """
        idx = idx_targets
        if isinstance(idx_targets, pd.DataFrame) or isinstance(idx_targets, pd.Series):
            idx = idx_targets.index
        elif isinstance(idx_targets, list):
            idx = pd.MultiIndex.from_tuples(idx_targets, names=names)

        feats = extract_features(
            self.setup["data_sources"],
            self.setup["feature_extraction"],
            data_source_index=source_index,
            data_source_columns=source_columns,
            join_on_index=idx,
        )
        inputs = [
            feats[self.setup["input_mapping"][name]] for name in self.model.input_names
        ]
        return inputs

    @staticmethod
    def load(config_file, weights_file, results_file):
        """
        Loads a trained model along with its configuration and results from specified files.

        This static method is used to instantiate a `SCAPE` object from a saved configuration,
        including the model's weights and training results. It is useful for resuming work
        with a previously trained model without needing to retrain from scratch.

        Parameters
        ----------
        config_file : str
            The path to the file containing the model's configuration data.
        weights_file : str
            The path to the file containing the model's trained weights.
        results_file : str
            The path to the file containing the results from the model's training process.

        Returns
        -------
        SCAPE
            A `SCAPE` object instantiated with the loaded configuration, weights, and training results.

        Raises
        ------
        IOError
            If any of the specified files cannot be read or are not found.
        """
        scm = None
        with open(config_file, "rb") as f:
            setup = pickle.load(f)
            scm = SCAPE(setup)
            scm.model.load_weights(weights_file)
            with open(results_file, "rb") as f:
                scm._last_train_results = pickle.load(f)
            return scm

    def save(self, config_file, weights_file, results_file):
        """
        Saves the model configuration, weights, and training results to files.

        This method enables the preservation of the current state of the model,
        including its configuration, trained weights, and results of the training.
        It facilitates reloading the model state for future use or analysis.

        Parameters
        ----------
        config_file : str
            The file path where the model's configuration will be saved.
        weights_file : str
            The file path where the model's weights will be saved.
        results_file : str
            The file path where the training results will be saved.

        Notes
        -----
        The method uses Python's pickle module to serialize the data into files.
        Ensure that the provided file paths are accessible and writable.
        """
        with open(config_file, "wb") as f:
            pickle.dump(self.setup, f)
        with open(results_file, "wb") as f:
            pickle.dump(self._last_train_results, f)
        self.model.save_weights(weights_file)

    def predict(
        self,
        idx_targets,
        as_df=True,
        output_idx=0,
        idx_target_names=None,
    ):
        """
        Predicts outputs using the model for the provided target indices.

        This method applies the trained model to generate predictions for specified target
        indices. It can return the results as a pandas DataFrame or as a numpy array, based
        on the `as_df` parameter.

        Parameters
        ----------
        idx_targets : array-like, pandas.DataFrame, pandas.Series, or list
            The indices for which predictions are required. Can be in various formats including
            an array, DataFrame, Series, or a list of tuples.
        as_df : bool, optional
            If True, returns the predictions as a pandas DataFrame. If False, returns predictions
            as a numpy array. Default is True.
        output_idx : int, optional
            The index of the output to return, in case the model has multiple outputs. Default is 0.
        idx_target_names : list of str, optional
            Names for the target indices, used when `idx_targets` is a list of tuples. If None,
            uses names from the training data's source index.

        Returns
        -------
        pandas.DataFrame or numpy.ndarray
            The predicted values. The format depends on the `as_df` parameter.

        Raises
        ------
        ValueError
            If no model has been trained yet, or if an invalid `output_idx` is provided.

        Notes
        -----
        The method assumes that the model and the necessary data processing steps are already
        properly set up and trained.
        """
        if self._last_train_results is None:
            raise ValueError("No model trained yet.")
        idx = idx_targets
        if isinstance(idx_targets, pd.DataFrame) or isinstance(idx_targets, pd.Series):
            idx = idx_targets.index
        elif isinstance(idx_targets, list):
            n = (
                self._last_train_results["data_source_index"].names
                if idx_target_names is None
                else idx_target_names
            )
            idx = pd.MultiIndex.from_tuples(idx_targets, names=n)

        feats = extract_features(
            self.setup["data_sources"],
            self.setup["feature_extraction"],
            data_source_index=self._last_train_results["data_source_index"],
            data_source_columns=self._last_train_results["data_source_columns"],
            join_on_index=idx,
        )
        inputs = [
            feats[self.setup["input_mapping"][name]] for name in self.model.input_names
        ]
        preds = self.model.predict(inputs)
        if isinstance(preds, list):
            preds = preds[output_idx]
        elif output_idx > 0:
            raise ValueError(f"Model has only one output.")
        if as_df:
            return pd.DataFrame(preds, index=idx, columns=self.setup["output_genes"])
        return preds


def checkpoint(filepath, monitor="val_loss"):
    checkpoint = keras.callbacks.ModelCheckpoint(
        filepath=filepath,
        monitor=monitor,
        verbose=0,
        save_best_only=True,
        save_weights_only=True,
        mode="min",
    )
    return checkpoint


def dense(
    x, units, act="elu", dropout=0.10, bnorm=True, noise=0.0, l1=0.0, l2=0.0, name=None
):
    if noise > 0:
        x = GaussianNoise(noise)(x)
    if dropout > 0:
        x = Dropout(dropout)(x)
    if bnorm:
        x = BatchNormalization()(x)
    reg = None
    if l1 > 0 or l2 > 0:
        reg = keras.regularizers.l1_l2(l1=l1, l2=l2)
    return Dense(units, name=name, activation=act, activity_regularizer=reg)(x)


def create_input_encoder(
    processing_layer_sizes: list,
    act: str = "elu",
    dropout: float = 0.10,
    input_noise: float = 0.0,
    name: str = None,
):
    input_layer = Input(shape=(processing_layer_sizes[0],), name=name)
    if name is None:
        name = "input"
    x = input_layer
    for i, size in enumerate(processing_layer_sizes[1:]):
        noise = input_noise if i == 0 else 0.0
        x = dense(x, size, act=act, dropout=dropout, noise=noise, name=f"{name}_{i}")
    # Create a model for the input processor
    input_processor = keras.Model(input_layer, x, name=f"{name}_processor")
    return input_processor


def encoder(
    input_structure: dict,
    hidden_layer_sizes: list,
    conditional_input_structure: dict = None,
    act: str = "elu",
    dropout: float = 0.10,
    input_noise: float = 0.0,
    l1: float = 0.0,
    l2: float = 0.0,
):
    # Create the input processors
    input_processors = {}
    for name, layer_sizes in input_structure.items():
        input_processors[name] = create_input_encoder(
            layer_sizes, act=act, dropout=dropout, input_noise=input_noise, name=name
        )
    # Create the conditional input processors
    conditional_input_processors = {}
    if conditional_input_structure is not None:
        for name, layer_sizes in conditional_input_structure.items():
            conditional_input_processors[name] = create_input_encoder(
                layer_sizes,
                act=act,
                dropout=dropout,
                input_noise=input_noise,
                name=name,
            )
    # Concatenate the processed inputs
    inputs = []
    for name, processor in input_processors.items():
        inputs.append(processor.output)
    x = Concatenate()(inputs)
    for size in hidden_layer_sizes:
        x = dense(x, size, act=act, dropout=dropout, l1=l1, l2=l2)

    # Use the conditional inputs to modify the hidden layer
    if conditional_input_processors:
        conditional_inputs = []
        for name, processor in conditional_input_processors.items():
            conditional_inputs.append(processor.output)
        x = Concatenate()(conditional_inputs + [x])
        for size in hidden_layer_sizes:
            x = dense(x, size, act=act, dropout=dropout, l1=l1, l2=l2)
    # Create the final model
    enc_inputs = [processor.input for processor in input_processors.values()]
    if conditional_input_processors:
        enc_inputs += [
            processor.input for processor in conditional_input_processors.values()
        ]
    encoder = keras.Model(
        enc_inputs,
        x,
        name="encoder",
    )
    return encoder


def decoder(
    enc_input_shape,
    outputs: dict = {"logpval": (18211, "linear")},
    hidden_layer_sizes: list = [256, 512],
    conditional_input_structure: dict = None,
    conditional_input_hidden_layer_sizes: list = [32],
    act: str = "elu",
    dropout: float = 0.10,
    input_noise: float = 0.0,
    enc_input_name: str = "enc_input",
    l1=0.0,
    l2=0.0,
):
    enc_input = Input(shape=enc_input_shape, name=enc_input_name)
    conditional_input_processors = {}
    if conditional_input_structure is not None:
        for name, layer_sizes in conditional_input_structure.items():
            conditional_input_processors[name] = create_input_encoder(
                layer_sizes,
                act=act,
                dropout=dropout,
                input_noise=input_noise,
                name=name,
            )
        # Concatenate and add the hidden layers
        conditional_inputs = []
        for name, processor in conditional_input_processors.items():
            conditional_inputs.append(processor.output)
        if len(conditional_inputs) > 1:
            x = Concatenate()(conditional_inputs)
            for size in conditional_input_hidden_layer_sizes:
                x = dense(x, size, act=act, dropout=dropout, l1=l1, l2=l2)
        else:
            x = conditional_inputs[0]
        # Concatenate with the encoder input
        x = Concatenate()([x, enc_input])
    else:
        x = enc_input
    # Add the hidden layers
    for size in hidden_layer_sizes:
        x = dense(x, size, act=act, dropout=dropout, l1=l1, l2=l2)
    # Add the outputs
    list_outputs = []
    for k, (size, act) in outputs.items():
        y = dense(x, size, act=act, name=k)
        list_outputs.append(y)
    # Create the final model
    dec_inputs = [enc_input]
    if conditional_input_processors:
        dec_inputs += [
            processor.input for processor in conditional_input_processors.values()
        ]
    decoder = keras.Model(
        inputs=dec_inputs,
        outputs=list_outputs,
        name="decoder",
    )
    return decoder


def aenn(enc, dec, enc_input_name="enc_input"):
    # Get the inputs of the encoder and decoder, except the encoder input to decoder
    inputs_enc = [inp for inp in enc.inputs]
    inputs_dec = [inp for inp in dec.inputs]
    inputs = [inp for inp in inputs_enc + inputs_dec if inp.name != enc_input_name]
    h = enc(inputs_enc)
    # Get the index of the input in inputs_dec which is enc_input_name
    idx = [i for i, inp in enumerate(inputs_dec) if inp.name == enc_input_name][0]
    # Replace the decoder input with the encoder output to link both models
    inputs_dec[idx] = h
    outputs = dec(inputs_dec)
    model = keras.Model(inputs=inputs, outputs=outputs, name="autoencoder")
    return model


def load_model(filepath, custom_objects={"mrrmse": mrrmse}):
    return keras.models.load_model(filepath, custom_objects=custom_objects)


def create_model(config):
    noise = config.get("noise", 0.0)
    dropout = config.get("dropout", 0.0)
    l1 = config.get("l1", 0.0)
    l2 = config.get("l2", 0.0)
    enc = encoder(
        config["inputs"],
        config["encoder_hidden_layer_sizes"],
        conditional_input_structure=config["conditional_encoder_input_structure"],
        input_noise=noise,
        dropout=dropout,
        l1=l1,
        l2=l2,
    )
    dec = decoder(
        enc.output.shape[1],
        outputs=config["outputs"],
        hidden_layer_sizes=config["decoder_hidden_layer_sizes"],
        conditional_input_structure=config["conditional_decoder_input_structure"],
        conditional_input_hidden_layer_sizes=config[
            "conditional_decoder_input_hidden_sizes"
        ],
        dropout=dropout,
        input_noise=noise,
    )
    model = aenn(enc, dec)
    return model, enc, dec


def predict(idx_targets, models, as_df=True, output_idx=0):
    # Map using data mapping
    predictions = []
    data = models["data"]
    for i, m in enumerate(models["models"]):
        res = models["results"][i]
        feats = extract_features(
            data["data_sources"],
            data["feature_extraction"],
            data_source_index=res["data_source_index"],
            data_source_columns=res["data_source_columns"],
            join_on_index=idx_targets.index,
        )
        inputs = [feats[data["input_mapping"][name]] for name in m.input_names]
        preds = m.predict(inputs)
        if isinstance(preds, list):
            preds = preds[output_idx]
        elif output_idx > 0:
            raise ValueError(f"Model {i} has only one output.")
        predictions.append(preds)
    if as_df:
        import pandas as pd

        return [
            pd.DataFrame(p, index=idx_targets.index, columns=data["output_genes"])
            for p in predictions
        ]
    return predictions


def create_default_model(n_genes, df_de, df_lfc):
    # Data sources point to the dataframes with the data from which the features are extracted
    data_sources = {"slogpval": df_de, "lfc_pseudo": df_lfc}

    # Define the way in which the features are extracted from the data sources
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

    # Names of the input layers -> feature extraction method
    input_mapping = {
        "in_slogpval_drug": "slogpval_drug",
        "in_lfc_drug": "lfc_drug",
        "in_slogpval_cell_encoder": "slogpval_cell",
        "in_lfc_cell_encoder": "lfc_cell",
        "in_slogpval_cell_decoder": "slogpval_cell",
        "in_lfc_cell_decoder": "lfc_cell",
    }

    # Configuration for the NN architecture
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
            "logpval": (df_de.shape[1], "linear"),
        },
        # Layer config
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

    # Create model
    return SCAPE(model_setup)
