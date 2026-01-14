import os
import re
import tempfile
from difflib import SequenceMatcher
from pathlib import Path
from textwrap import fill
from typing import Dict, List, Optional, Tuple

import datajoint as dj
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

logger = dj.logger


# Constants used for syllable filtering (shared between MotionSequence and TrajectoryPlot)
# Values match keypoint-moseq defaults: https://github.com/dattalab/keypoint-moseq
MIN_DURATION = 3  # Minimum duration in frames
MIN_FREQUENCY = 0.005  # Minimum frequency as fraction (0.5% of total instances)

# Supported video file extensions
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".wmv", ".mpeg", ".mpg", ".mkv", ".webm"}


def extract_base_video_key(video_key: str) -> str:
    """Extract base video name from video key (removes DLC suffix if present).

    Args:
        video_key: Video key from results.h5 (may include DLC suffix, e.g. '21_11_8_one_mouseDLC_resnet50_OFT_FPApr14shuffle1_100000' or '21_11_8_one_mouse.top.irDLC_resnet50_moseq_exampleAug21shuffle1_500000')

    Returns:
        Base video key (name before "DLC" or "irDLC" suffix if present, otherwise original key, e.g. '21_11_8_one_mouse')
    """
    # Handle various DLC suffix patterns
    if ".irDLC" in video_key:
        return video_key.split(".irDLC")[0]
    if "irDLC" in video_key:
        return video_key.split("irDLC")[0]
    if "DLC" in video_key:
        return video_key.split("DLC")[0]
    if video_key.endswith("-tracking"):
        return video_key[:-9]  # Remove "-tracking" suffix
    return video_key


def match_video_keys_to_file_paths(video_keys, file_ids, file_paths, video_only=False):
    """Match video keys from results.h5 to file paths from RecordingSet.File.

    Args:
        video_keys: List of video keys from results.h5 (e.g., ['video1', 'video2'])
        file_ids: List of file_ids corresponding to file_paths
        file_paths: List of file paths from RecordingSet.File
        video_only: If True, filter to only video file extensions

    Returns:
        dict: Mapping of video_key -> (file_id, file_path) for matched videos
    """
    if video_only:
        filtered_indices = [
            i
            for i, fp in enumerate(file_paths)
            if Path(fp).suffix.lower() in VIDEO_EXTENSIONS
        ]
        file_ids = [file_ids[i] for i in filtered_indices]
        file_paths = [file_paths[i] for i in filtered_indices]

    matched = {}
    for vid_key in video_keys:
        base_video_key = extract_base_video_key(vid_key)

        for file_id, file_path in zip(file_ids, file_paths):
            file_stem = Path(file_path).stem

            # Try multiple matching strategies in order of specificity:
            # 1. Exact match (case-insensitive) - most specific
            # 2. Base video key exact match (case-insensitive)
            # 3. Partial match: check if base_video_key starts with file_stem or vice versa
            #    (more reliable than "contains" to avoid false matches)
            if (
                vid_key == file_stem
                or vid_key.lower() == file_stem.lower()
                or base_video_key == file_stem
                or base_video_key.lower() == file_stem.lower()
            ):
                # Exact matches - highest priority
                matched[vid_key] = (file_id, file_path)
                break
            elif (
                # Partial matches - check if one starts with the other (more reliable than "contains")
                # This handles cases like:
                # - "21_11_8_one_mouse.top.ir" (file) vs "21_11_8_one_mouse.top.irDLC_..." (key)
                # - "21_11_8_one_mouse.top.irDLC_..." (file) vs "21_11_8_one_mouse.top.irDLC_..." (key)
                (
                    len(base_video_key) > 5
                    and len(file_stem) > 5
                    and (
                        file_stem.lower().startswith(base_video_key.lower())
                        or base_video_key.lower().startswith(file_stem.lower())
                    )
                )
                or (
                    len(vid_key) > 5
                    and len(file_stem) > 5
                    and (
                        file_stem.lower().startswith(vid_key.lower())
                        or vid_key.lower().startswith(file_stem.lower())
                    )
                )
            ):
                # Partial match found - one is a prefix of the other
                matched[vid_key] = (file_id, file_path)
                break

    return matched


# ---- Modified version of the viz function from the main branch of keypoint_moseq  ----
def plot_medoid_distance_outliers(
    project_dir: str,
    recording_name: str,
    original_coordinates: np.ndarray,
    interpolated_coordinates: np.ndarray,
    outlier_mask,
    outlier_thresholds,
    bodyparts: list[str],
    **kwargs,
):
    """Create and save a plot comparing distance-to-medoid for original vs. interpolated keypoints.

    Generates a multi-panel plot showing the distance from each keypoint to the medoid
    position for both original and interpolated coordinates. The plot includes threshold
    lines and shaded regions for outlier frames. Saves the figure to the QA plots
    directory.

    Parameters
    -------
    project_dir: str
        Path to the project directory where the plot will be saved.

    recording_name: str
        Name of the recording, used for the plot title and filename.

    original_coordinates: ndarray of shape (n_frames, n_keypoints, keypoint_dim)
        Original keypoint coordinates before interpolation.

    interpolated_coordinates: ndarray of shape (n_frames, n_keypoints, keypoint_dim)
        Keypoint coordinates after interpolation.

    outlier_mask: ndarray of shape (n_frames, n_keypoints)
        Boolean mask indicating outlier keypoints (True = outlier).

    outlier_thresholds: ndarray of shape (n_keypoints,)
        Distance thresholds for each keypoint above which points are considered outliers.

    bodyparts: list of str
        Names of bodyparts corresponding to each keypoint. Must have length equal to
        n_keypoints.

    **kwargs
        Additional keyword arguments (ignored), usually overflow from **config().

    Returns
    -------
    tuple of (fig, plot_path)
        fig: matplotlib.figure.Figure
            The generated figure object
        plot_path: str
            Path to the saved plot file: 'QA/plots/keypoint_distance_outliers/{recording_name}.png'
    """
    from keypoint_moseq.util import get_distance_to_medoid, plot_keypoint_traces

    # Use QA directory for outlier plots
    plot_path = os.path.join(
        project_dir,
        "QA",
        "plots",
        "keypoint_distance_outliers",
        f"{recording_name}.png",
    )
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(plot_path), exist_ok=True)

    original_distances = get_distance_to_medoid(
        original_coordinates
    )  # (n_frames, n_keypoints)
    interpolated_distances = get_distance_to_medoid(
        interpolated_coordinates
    )  # (n_frames, n_keypoints)

    fig = plot_keypoint_traces(
        traces=[original_distances, interpolated_distances],
        plot_title=recording_name,
        bodyparts=bodyparts,
        line_labels=["Original", "Interpolated"],
        thresholds=outlier_thresholds,
        shading_mask=outlier_mask,
    )

    fig.savefig(plot_path, dpi=300)

    plt.close()
    logger.info(
        f"Saved keypoint distance outlier plot for {recording_name} to {plot_path}."
    )
    return fig, plot_path


# ---- Modified version of the viz function from the main branch of keypoint_moseq  ----
def plot_pcs(
    pca,
    *,
    use_bodyparts,
    skeleton,
    keypoint_colormap="autumn",
    keypoint_colors=None,
    savefig=True,
    project_dir=None,
    scale=1,
    plot_n_pcs=10,
    axis_size=(2, 1.5),
    ncols=5,
    node_size=30.0,
    line_width=2.0,
    interactive=True,
    **kwargs,
):
    """
    Visualize the components of a fitted PCA model.

    For each PC, a subplot shows the mean pose (semi-transparent) along with a
    perturbation of the mean pose in the direction of the PC.

    Parameters
    ----------
    pca : :py:func:`sklearn.decomposition.PCA`
        Fitted PCA model

    use_bodyparts : list of str
        List of bodyparts to that are used in the model; used to index bodypart
        names in the skeleton.

    skeleton : list
        List of edges that define the skeleton, where each edge is a pair of
        bodypart names.

    keypoint_colormap : str
        Name of a matplotlib colormap to use for coloring the keypoints.

    keypoint_colors : array-like, shape=(num_keypoints,3), default=None
        Color for each keypoint. If None, `keypoint_colormap` is used. If the
        dtype is int, the values are assumed to be in the range 0-255,
        otherwise they are assumed to be in the range 0-1.

    savefig : bool, True
        Whether to save the figure to a file. If true, the figure is saved to
        `{project_dir}/pcs-{xy/xz/yz}.pdf` (`xz` and `yz` are only included
        for 3D data).

    project_dir : str, default=None
        Path to the project directory. Required if `savefig` is True.

    scale : float, default=0.5
        Scale factor for the perturbation of the mean pose.

    plot_n_pcs : int, default=10
        Number of PCs to plot.

    axis_size : tuple of float, default=(2,1.5)
        Size of each subplot in inches.

    ncols : int, default=5
        Number of columns in the figure.

    node_size : float, default=30.0
        Size of the keypoints in the figure.

    line_width: float, default=2.0
        Width of edges in skeleton

    interactive : bool, default=True
        For 3D data, whether to generate an interactive 3D plot.
    """
    from jax_moseq.models.keypoint_slds import center_embedding
    from keypoint_moseq.util import get_edges
    from keypoint_moseq.viz import plot_pcs_3D

    k = len(use_bodyparts)
    if k < 2:
        raise ValueError("use_bodyparts must contain at least 2 items for PCA plotting")
    d = len(pca.mean_) // (k - 1)

    if keypoint_colors is None:
        cmap = plt.cm.get_cmap(keypoint_colormap)
        keypoint_colors = cmap(np.linspace(0, 1, k))

    Gamma = np.array(center_embedding(k))
    edges = get_edges(use_bodyparts, skeleton)
    plot_n_pcs = min(plot_n_pcs, pca.components_.shape[0])

    magnitude = np.sqrt((pca.mean_**2).mean()) * scale
    ymean = Gamma @ pca.mean_.reshape(k - 1, d)
    ypcs = (pca.mean_ + magnitude * pca.components_).reshape(-1, k - 1, d)
    ypcs = Gamma[np.newaxis] @ ypcs[:plot_n_pcs]

    if d == 2:
        dims_list, names = [[0, 1]], ["xy"]
    if d == 3:
        dims_list, names = [[0, 1], [0, 2]], ["xy", "xz"]

    for dims, name in zip(dims_list, names):
        nrows = int(np.ceil(plot_n_pcs / ncols))
        fig, axs = plt.subplots(nrows, ncols, sharex=True, sharey=True)
        for i, ax in enumerate(axs.flat):
            if i >= plot_n_pcs:
                ax.axis("off")
                continue

            for e in edges:
                ax.plot(
                    *ymean[:, dims][e].T,
                    color=keypoint_colors[e[0]],
                    zorder=0,
                    alpha=0.25,
                    linewidth=line_width,
                )
                ax.plot(
                    *ypcs[i][:, dims][e].T,
                    color="k",
                    zorder=2,
                    linewidth=line_width + 0.2,
                )
                ax.plot(
                    *ypcs[i][:, dims][e].T,
                    color=keypoint_colors[e[0]],
                    zorder=3,
                    linewidth=line_width,
                )

            ax.scatter(
                *ymean[:, dims].T,
                c=keypoint_colors,
                s=node_size,
                zorder=1,
                alpha=0.25,
                linewidth=0,
            )
            ax.scatter(
                *ypcs[i][:, dims].T,
                c=keypoint_colors,
                s=node_size,
                zorder=4,
                edgecolor="k",
                linewidth=0.2,
            )

            ax.set_title(f"PC {i+1}", fontsize=10)
            ax.set_aspect("equal")
            ax.axis("off")

        fig.set_size_inches((axis_size[0] * ncols, axis_size[1] * nrows))
        plt.tight_layout()

        if savefig:
            if project_dir is None:
                raise ValueError(fill("The `savefig` option requires a `project_dir`"))
            plt.savefig(os.path.join(project_dir, f"pcs-{name}.pdf"))
        plt.show()

    if interactive and d == 3:
        plot_pcs_3D(
            ymean,
            ypcs,
            edges,
            keypoint_colormap,
            project_dir if savefig else None,
            node_size / 3,
            line_width * 2,
        )
    return fig


def copy_pdf_to_png(project_dir, model_name):
    """
    Convert PDF progress plot to PNG format using pdf2image.

    Args:
        project_dir (str or Path): Project directory path (must be absolute)
        model_name (str): Model name directory

    """
    from pdf2image import convert_from_path

    # Ensure project_dir is an absolute path
    project_dir_path = Path(project_dir)
    if not project_dir_path.is_absolute():
        raise ValueError(f"project_dir must be an absolute path, got: {project_dir}")
    model_dir = project_dir_path / model_name
    pdf_path = model_dir / "fitting_progress.pdf"
    png_path = model_dir / "fitting_progress.png"

    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF progress plot not found at {pdf_path}")

    # Convert PDF to images and save PNG to the model directory
    images = convert_from_path(str(pdf_path), dpi=300)

    if not images:
        raise ValueError(f"Could not convert PDF at {pdf_path} (no images returned)")

    # Save PNG to the model directory (full absolute path)
    images[0].save(str(png_path), "PNG")
    logger.info(f"Generated PNG progress plot at {png_path}")
    return png_path, pdf_path


# ---- Modified version of the viz function from the main branch of keypoint_moseq  ----
def plot_nan_breakdown(
    coordinates: Dict[str, np.ndarray], use_bodyparts: List[str]
) -> str:
    """Create a PNG visualization of NaN proportion breakdown by recording and bodypart.

    Replicates keypoint-moseq's `check_nan_proportions` logic

    Generates a color-coded table showing the proportion of NaN values for each
    recording and bodypart combination. The table uses a color gradient (RdYlBu_r)
    where red indicates high NaN proportions and blue indicates low NaN proportions.

    Parameters
    ----------
    coordinates : dict
        Dictionary mapping recording names to coordinate arrays of shape
        (n_frames, n_bodyparts, 2).
    use_bodyparts : list of str
        List of bodypart names corresponding to the columns in the coordinate arrays.

    Returns
    -------
    str
        Path to the temporary PNG file containing the visualization.
    """
    # Calculate NaN proportions breakdown for each recording and bodypart
    keys = sorted(coordinates.keys())
    nan_props = [np.isnan(coordinates[k]).any(-1).mean(0) for k in keys]

    # Create the DataFrame for visualization
    nan_df = pd.DataFrame(data=nan_props, index=keys, columns=use_bodyparts)

    # Create matplotlib figure with table
    fig, ax = plt.subplots(
        figsize=(max(12, len(use_bodyparts) * 1.5), max(8, len(keys) * 0.5))
    )
    ax.axis("tight")
    ax.axis("off")

    # Format values as percentages for display
    nan_df_display = nan_df.applymap(lambda x: f"{x:.1%}")

    # Create table with color gradient based on NaN proportions
    table = ax.table(
        cellText=nan_df_display.values,
        rowLabels=nan_df.index,
        colLabels=nan_df.columns,
        cellLoc="center",
        loc="center",
    )

    # Style the table
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2)

    # Apply color gradient to cells based on NaN proportions
    for i in range(len(nan_df.index)):
        for j in range(len(nan_df.columns)):
            value = nan_df.iloc[i, j]
            # Normalize value to [0, 1] for colormap
            normalized_value = min(max(value, 0), 1)
            # Use RdYlBu_r colormap (red for high NaN, blue for low NaN)
            color = plt.cm.RdYlBu_r(normalized_value)
            table[(i + 1, j)].set_facecolor(color)
            table[(i + 1, j)].set_text_props(weight="bold" if value > 0.5 else "normal")

    # Style header row
    for j in range(len(nan_df.columns)):
        table[(0, j)].set_facecolor("#F2F2F2")
        table[(0, j)].set_text_props(weight="bold", color="#222")

    # Style row labels
    for i in range(len(nan_df.index)):
        table[(i + 1, -1)].set_facecolor("#F2F2F2")
        table[(i + 1, -1)].set_text_props(weight="bold")

    plt.title("NaN Proportion Breakdown", fontsize=14, fontweight="bold", pad=20)

    # Save PNG to temporary file for DataJoint attach
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
        nan_png_path = f.name
    plt.savefig(nan_png_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    return nan_png_path


def extract_syllable_id_from_filename(filename):
    """Extract syllable ID from filename.

    Args:
        filename (Path): File path object.

    Returns:
        int or None: Syllable ID if extractable, None otherwise.
    """
    try:
        stem = filename.stem
        if "syllable" in stem.lower():
            return int(
                stem.lower().replace("syllable", "").replace("_", "").replace("-", "")
            )
        elif stem.isdigit():
            return int(stem)
    except (ValueError, AttributeError):
        pass
    return None


def find_trajectory_files(directory, extension, exclude_name=None):
    """Find trajectory plot files (GIF or PDF) in directory.

    Args:
        directory (Path): Directory to search.
        extension (str): File extension to search for ('.gif' or '.pdf').
        exclude_name (str): Filename to exclude from results.

    Returns:
        tuple: (set of syllable IDs, dict mapping syllable_id to file path)
    """
    directory = Path(directory)
    if not directory.exists():
        return set(), {}

    pattern = f"syllable*{extension}"
    files = list(directory.rglob(pattern))

    if not files:
        alt_pattern = f"*{extension}"
        alt_files = [
            f
            for f in directory.rglob(alt_pattern)
            if ("syllable" in f.name.lower() or f.stem.isdigit())
            and (exclude_name is None or f.name != exclude_name)
        ]
        if alt_files:
            logger.info(
                f"Using alternative {extension} pattern, found {len(alt_files)} files"
            )
            files = alt_files

    syllable_ids = set()
    file_paths = {}
    for f in files:
        syllable_id = extract_syllable_id_from_filename(f)
        if syllable_id is not None:
            syllable_ids.add(syllable_id)
            file_paths[syllable_id] = f

    return syllable_ids, file_paths


def find_grid_movie_files(directory):
    """Find grid movie MP4 files in directory.

    Args:
        directory (Path): Directory to search.

    Returns:
        tuple: (set of syllable IDs, dict mapping syllable_id to file path)
    """
    directory = Path(directory)
    files = list(directory.rglob("syllable*.mp4"))

    syllable_ids = {int(f.stem.replace("syllable", "")) for f in files}
    file_paths = {int(f.stem.replace("syllable", "")): f for f in files}

    return syllable_ids, file_paths
