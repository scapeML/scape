import numpy as np
import pandas as pd


def select_top_variable(
    dfs,
    k=64,
    union=True,
    exclude_controls=False,
    controls=["Belinostat", "Dabrafenib"],
    axis=0,
):
    if exclude_controls:
        topn_list = [
            df.loc[~df.index.get_level_values("cell_type").isin(controls)]
            .std(axis=axis)
            .sort_values(ascending=False)
            .head(k)
            .index.tolist()
            for df in dfs
        ]
    else:
        topn_list = [
            df.std(axis=axis).sort_values(ascending=False).head(k).index.tolist()
            for df in dfs
        ]
    if union:
        topn_list = list(set().union(*topn_list))
    else:
        topn_list = list(set(topn_list[0]).intersection(*topn_list[1:]))
    return topn_list


def split_data(
    df,
    levels=["cell_type", "sm_name"],
    values=[
        ["B cells", "Myeloid cells"],
        ["Idelalisib", "Linagliptin", "CHIR-99021", "R428"],
    ],
):
    selected = None
    for l, v in zip(levels, values):
        if l is not None and v is not None:
            ix = df.index.get_level_values(l).isin(v)
            if selected is None:
                selected = ix
            else:
                selected = selected & ix
    if selected is None:
        return df.index, pd.Index([])
    return df[~selected].index, df[selected].index


def grouped_features(df, g="sm_name", index=None):
    f = df.groupby(g).median()
    if index is None:
        index = df.index
    return pd.DataFrame(index=index).join(f, on=g, how="left")


def extract_features(
    data_sources: dict,
    features: dict,
    data_source_index=None,
    data_source_columns=None,
    join_on_index=None,
):
    if not isinstance(join_on_index, list):
        join_on_index = [join_on_index]

    # feature_result = {}
    results = [dict() for _ in range(len(join_on_index))]
    for name, feature_attrs in features.items():
        ds = data_sources[feature_attrs["source"]]
        groupby = feature_attrs["groupby"]
        fn = feature_attrs["function"]
        if data_source_index is not None:
            ds = ds.loc[data_source_index]
        if data_source_columns is not None:
            ds = ds[data_source_columns]
        df_feature = ds.groupby(groupby).agg(fn)
        df_feature.name = name
        for feature_result, idx in zip(results, join_on_index):
            if idx is not None:
                joined_df = pd.DataFrame(index=idx).join(
                    df_feature, on=groupby, how="left"
                )
                joined_df.name = name
                feature_result[name] = joined_df
            else:
                feature_result[name] = df_feature
    if len(results) == 1:
        return results[0]
    return results


def to_pvalue(logpvals, threshold=None):
    # Convert from -log10(p-value) to p-value
    pvals = np.power(10, -np.abs(logpvals))
    if threshold is not None:
        below_threshold = pvals <= threshold
        pvals[below_threshold] = 1
        pvals[~below_threshold] = 0
    return pvals


def plot_result(
    result,
    target_metric="val_loss",
    metrics=["loss"],
    include_val=True,
    logy=True,
    interval_size=None,
    ax=None,
    legend=False,
):
    if include_val:
        val_metrics = [f"val_{m}" for m in metrics]
        metrics = metrics + val_metrics
    metrics = list(set(metrics + [target_metric]))
    min_val = result["history"][target_metric].min()
    r = result["history"][metrics]
    x_best = r[target_metric].idxmin()
    if interval_size:
        start = max(0, x_best - interval_size)
        end = min(x_best + interval_size, r.shape[0])
        r = r.iloc[start:end, :]
    if ax is None:
        ax = r.plot(logy=logy, legend=legend)
    else:
        r.plot(logy=logy, ax=ax, legend=legend)
    percentages = {}
    if "baselines" in result:
        for b_name, b_val in result["baselines"].items():
            ax.axhline(y=b_val, linestyle=":", color="k", label=b_name)
            percentages[b_name] = 100 * (1 - (min_val / b_val))
    
    ax.axhline(y=min_val, color="k", label="min val")
    ax.axvline(x=result["history"][target_metric].idxmin(), color="k", linestyle=":")
    # 
    # Add a title using cell/drug for validation
    v_drugs = result["drugs"]
    v_cells = result["cells"]
    val_drugs = f"{len(v_drugs)} drug(s): " + "/".join(v_drugs)
    val_cells = f"{len(v_cells)} cell(s): " + "/".join(v_cells)
    # If val_drugs is longer than 50 characters, truncate
    val_drugs = (val_drugs[:47] + '...') if len(val_drugs) > 50 else val_drugs
    val_cells = (val_cells[:47] + '...') if len(val_cells) > 50 else val_cells
    # Add a title using cell/drug for validation and also the percentages from baselines
    if percentages:
        ax.set_title(f"Validation: {val_cells} + {val_drugs}\n" + "\n".join([f"Baseline {k}: {v:.2f}%" for k, v in percentages.items()]))
    else:
        ax.set_title(f"Validation: {val_cells} + {val_drugs}")
    
    if legend:
        #ax.legend()
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')



def plot_results(results, logy=True, interval_size=None, ax=None, legend=False):
    import matplotlib.pyplot as plt

    # Determine the number of rows needed for subplots (2 plots per row).
    num_results = len(results)
    num_rows = (num_results + 1) // 2  # Adds an extra row if odd number of results.
    fig, axes = plt.subplots(nrows=num_rows, ncols=2, figsize=(10, num_rows * 3))
    axes = axes.flatten() if num_rows > 1 else [axes]
    for i, res in enumerate(results):
        plot_result(
            res, interval_size=interval_size, logy=logy, ax=axes[i], legend=legend
        )
    # If the number of results is odd, we should hide the last ax
    if num_results % 2 != 0:
        axes[-1].axis("off")
    # Add fold title
    for i, ax in enumerate(axes):
        ax.set_title(f"Fold {i}")
    plt.tight_layout()
    plt.show()


def plot_results2(results, logy=True, interval_size=None, ax=None, legend=False):
    import matplotlib.pyplot as plt

    # Determine the number of rows needed for subplots (2 plots per row).
    num_results = len(results)
    num_rows = (num_results + 1) // 2  # Adds an extra row if odd number of results.
    fig, axes = plt.subplots(nrows=num_rows, ncols=2, figsize=(10, num_rows * 3))
    axes = axes.flatten() if num_rows > 1 else [axes]

    handles, labels = [], []

    for i, res in enumerate(results):
        plot_result(
            res, interval_size=interval_size, logy=logy, ax=axes[i], legend=False
        )
        # Collect handles and labels for the legend
        if legend:
            handles_, labels_ = axes[i].get_legend_handles_labels()
            handles.extend(handles_)
            labels.extend(labels_)

    # If the number of results is odd, we should hide the last ax
    if num_results % 2 != 0:
        axes[-1].axis("off")

    # Add fold title
    for i, ax in enumerate(axes):
        ax.set_title(f"Fold {i}")

    # Remove duplicates in handles and labels
    by_label = dict(zip(labels, handles))

    if legend:
        # Create a single shared legend
        fig.legend(
            by_label.values(), by_label.keys(), loc="upper right", bbox_to_anchor=(1, 1)
        )

    plt.tight_layout()
    plt.show()


def plot(keras_model, **kwargs):
    from keras.utils import plot_model
    from IPython.display import Image, display
    import tempfile
    import os

    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as temp_file:
        temp_path = temp_file.name

    kwargs["to_file"] = temp_file.name
    plot_model(keras_model, **kwargs)
    # Display the image in the notebook
    display(Image(filename=temp_path))
    # Delete the temporary file
    os.remove(temp_path)
