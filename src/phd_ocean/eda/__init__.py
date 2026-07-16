"""
Exploratory Data Analysis tools for oceanographic datasets.

Main workflows
--------------
run_eda_1d
    Yearly statistical summary tables and violin plots.

run_eda_2d
    Full-period statistical maps and seasonal/monthly/annual maps.
"""

from .io import (
    expand_paths,
    infer_time_dim,
    standardize_time_dim,
    resample_time,
    load_dataset,
    load_many_datasets,
    transform_longitude,
    select_region,
    select_vertical_section_ij,
    make_section_indices,
    get_first_data_variable,
)

from .stats import (
    DEFAULT_1D_STATS,
    DEFAULT_2D_STATS,
    compute_1d_statistics,
    compute_1d_statistics_by_year,
    compute_2d_statistics,
    aggregate_periods,
    compute_1d_circular_statistics,
    compute_1d_vector_statistics,
    compute_1d_vector_statistics_by_year,
)

from .plotting import (
    get_lon_lat,
    add_map_features,
    plot_violin_by_year,
    plot_2d_map,
    plot_2d_subplots,
    plot_quiver,
    plot_scalar_with_vectors,
    plot_vector_map,
    plot_scalar_vector_subplots,
)

from .eda_1d import (
    prepare_1d_data,
    xarray_to_yearly_dataframe,
    dataframe_to_violin_data,
    save_summary_table,
    run_eda_1d,
    prepare_1d_vector_data,
    run_eda_1d_vector,
)

from .eda_2d import (
    prepare_2d_field,
    run_eda_2d_statistics,
    run_eda_2d_periods,
    run_eda_2d,
)


__all__ = [
    # io
    "expand_paths",
    "infer_time_dim",
    "standardize_time_dim",
    "resample_time",
    "load_dataset",
    "load_many_datasets",
    "transform_longitude",
    "select_region",
    "select_vertical_section_ij",
    "make_section_indices",
    "get_first_data_variable",

    # stats
    "DEFAULT_1D_STATS",
    "DEFAULT_2D_STATS",
    "compute_1d_statistics",
    "compute_1d_statistics_by_year",
    "compute_2d_statistics",
    "aggregate_periods",
    "compute_1d_circular_statistics",
    "compute_1d_vector_statistics",
    "compute_1d_vector_statistics_by_year",

    # plotting
    "get_lon_lat",
    "add_map_features",
    "plot_violin_by_year",
    "plot_2d_map",
    "plot_2d_subplots",
    "plot_quiver",
    "plot_scalar_with_vectors",
    "plot_vector_map",
    "plot_scalar_vector_subplots",

    # eda_1d
    "prepare_1d_data",
    "xarray_to_yearly_dataframe",
    "dataframe_to_violin_data",
    "save_summary_table",
    "run_eda_1d",
    "prepare_1d_vector_data",
    "run_eda_1d_vector",


    # eda_2d
    "prepare_2d_field",
    "run_eda_2d_statistics",
    "run_eda_2d_periods",
    "run_eda_2d",
]