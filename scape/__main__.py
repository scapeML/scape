import scape

def main():
    print(f"Running ScAPE version {scape.__version__}")
    # Show the help message
    import argparse

    parser = argparse.ArgumentParser(
        description="ScAPE - Single Cell Analysis of Perturbational Effects"
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"scape version {scape.__version__}",
        help="Show the version of scape",
    )
    # Add argument for training, it requires params like the model name, the data, etc.
    subparsers = parser.add_subparsers(
        dest="command", help="Subcommands for ScAPE"
    )
    # Training a model
    train_parser = subparsers.add_parser(
        "train", help="Train a model for a given dataset"
    )
    train_parser.add_argument(
        "slogpval",
        type=str,
        help="Parquet file containing the signed log-pvalues (de_train.parquet)",
    )
    train_parser.add_argument(
        "lfc",
        type=str,
        help="Parquet file containing the log-fold changes (lfc_train.parquet)",
    )
    train_parser.add_argument(
        "--n-genes",
        type=int,
        default=64,
        help="Number of top genes to use for training for the base model",
    )
    # Same but for the enhanced model
    train_parser.add_argument(
        "--n-genes-enhanced",
        type=int,
        default=256,
        help="Number of top genes to use for training for the enhanced model",
    )
    # Min number of drugs to use for the enhanced model
    train_parser.add_argument(
        "--min_drugs",
        type=int,
        default=50,
        help="Minimum number of drugs to use for training for the enhanced model",
    )
    
    # out folder, default current dir
    train_parser.add_argument(
        "--output-dir",
        type=str,
        required=False,
        help="Folder to save the trained models",
    )
    train_parser.add_argument(
        "--cv-cell",
        type=str,
        required=False,
        default="NK cells",
        help="Cell type used as validation during training and for model selection (default: NK cells)",
    )
    train_parser.add_argument(
        "--cv-drug",
        type=str,
        required=False,
        help="Drug used as validation during training and for model selection (default: None)",
    )
    # Epochs
    train_parser.add_argument(
        "--epochs",
        type=int,
        default=600,
        help="Number of epochs to train the model",
    )
    # Batch size
    train_parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
        help="Batch size to train the model",
    )
    # Config file (default config.pkl)
    train_parser.add_argument(
        "--config-name",
        type=str,
        default="config",
        help="Name of the config file (.pkl extension will be added)",
    )
    # Model file name (default model)
    train_parser.add_argument(
        "--model-name",
        type=str,
        default="model",
        help="Name of the model file to store weights (.keras extension will be added)",
    )
    args = parser.parse_args()

    # If the command was train, train the model
    if parser.parse_args().command == "train":
        train(args)


def train(args):
    # Read the files
    df_de = scape.io.load_slogpvals(args.slogpval)
    print(f"DE shape: {df_de.shape}")
    df_lfc = scape.io.load_lfc(args.lfc)
    print(f"LFC shape: {df_lfc.shape}")
    val_cells = [args.cv_cell] if args.cv_cell else None
    val_drugs = [args.cv_drug] if args.cv_drug else None
    print(f"Training model with {args.n_genes} genes")
    print(f"Validation cell(s): {val_cells}")
    print(f"Validation drug(s): {val_drugs}")
    # Create a default model
    model = scape.model.create_default_model(args.n_genes, df_de, df_lfc)
    top_genes = top_genes = scape.util.select_top_variable([df_de], k=args.n_genes)
    model.train(
        val_cells=val_cells,
        val_drugs=val_drugs,
        output_data="slogpval",
        callbacks="default",
        input_columns=top_genes,
        optimizer=None,
        epochs=args.epochs,
        batch_size=args.batch_size,
        output_folder=args.output_dir,
        config_file_name=f"{args.config_name}.pkl",
        model_file_name=f"{args.model_name}.keras",
        baselines=["zero", "slogpval_drug"]
    )


if __name__ == "__main__":
    main()