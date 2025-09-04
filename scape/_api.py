import scape

def train(
    de_file,
    lfc_file,
    n_genes=64,
    output_dir=None,
    cv_cell="NK cells",
    cv_drug=None,
    epochs=600,
    batch_size=128,
    config_name="config",
    model_name="model"
):
    # Read the files
    df_de = scape.io.load_slogpvals(de_file)
    print(f"DE shape: {df_de.shape}")
    df_lfc = scape.io.load_lfc(lfc_file)
    print(f"LFC shape: {df_lfc.shape}")
    val_cells = [cv_cell] if cv_cell else None
    val_drugs = [cv_drug] if cv_drug else None
    print(f"Training model with {n_genes} genes")
    print(f"Validation cell(s): {val_cells}")
    print(f"Validation drug(s): {val_drugs}")
    # Create a default model
    model = scape.model.create_default_model(n_genes, df_de, df_lfc)
    top_genes = scape.util.select_top_variable([df_de], k=n_genes)
    model.train(
        val_cells=val_cells,
        val_drugs=val_drugs,
        output_data="slogpval",
        callbacks="default",
        input_columns=top_genes,
        optimizer=None,
        epochs=epochs,
        batch_size=batch_size,
        output_folder=output_dir,
        config_file_name=f"{config_name}.pkl",
        model_file_name=f"{model_name}.keras",
        baselines=["zero", "slogpval_drug"]
    )
    return model