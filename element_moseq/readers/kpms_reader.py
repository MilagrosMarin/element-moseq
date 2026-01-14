import os
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union

import cv2
import datajoint as dj
import numpy as np
import yaml

from ..plotting.viz_utils import VIDEO_EXTENSIONS

logger = dj.logger

KPMS_DJ_CONFIG = "kpms_dj_config.yml"
CONFIG_FILENAMES = [
    "config.yml",
    "config.yaml",
]  # Used for both pose estimation and KPMS base configs


def _pose_estimation_config_path(kpset_dir: Union[str, os.PathLike]) -> str:
    """
    Return the path to the pose estimation config file (e.g., DeepLabCut config.yaml) in the keypoint set directory.

    Args:
        kpset_dir: Keypoint set directory (where pose estimation files are located)

    Returns:
        Path to pose estimation config file (config.yml or config.yaml)
    """
    kpset_path = Path(kpset_dir)
    for filename in CONFIG_FILENAMES:
        config_path = kpset_path / filename
        if config_path.exists():
            return str(config_path)
    return str(kpset_path / CONFIG_FILENAMES[0])


def _kpms_base_config_path(kpms_project_dir: Union[str, os.PathLike]) -> str:
    """
    Return the path to the KPMS base config file (created by keypoint_moseq's setup_project) in the KPMS output directory.

    Args:
        kpms_project_dir: KPMS project output directory

    Returns:
        Path to KPMS base config file (config.yml or config.yaml)
    """
    project_path = Path(kpms_project_dir)
    for filename in CONFIG_FILENAMES:
        config_path = project_path / filename
        if config_path.exists():
            return str(config_path)
    return str(project_path / CONFIG_FILENAMES[0])


def _kpms_dj_config_path(kpms_project_dir: Union[str, os.PathLike]) -> str:
    """
    Return the path to the KPMS DJ config file (kpms_dj_config.yml) in the KPMS output directory.
    This is the DataJoint-specific config file that gets updated during the pipeline.

    Args:
        kpms_project_dir: KPMS project output directory

    Returns:
        Path to KPMS DJ config file (kpms_dj_config.yml)
    """
    return str(Path(kpms_project_dir) / KPMS_DJ_CONFIG)


# ---- Modified version of the function from the main branch of keypoint_moseq  ----
def _check_config_validity(config: Dict[str, Any]) -> bool:
    """
    Minimal mirror of keypoint_moseq.io.check_config_validity logic that matters
    for anatomy consistency (anterior/posterior must be subset of use_bodyparts).
    """
    errors = []
    for bp in config.get("anterior_bodyparts", []):
        if bp not in config.get("use_bodyparts", []):
            errors.append(
                f"ACTION REQUIRED: `anterior_bodyparts` contains {bp} "
                "which is not one of the options in `use_bodyparts`."
            )
    for bp in config.get("posterior_bodyparts", []):
        if bp not in config.get("use_bodyparts", []):
            errors.append(
                f"ACTION REQUIRED: `posterior_bodyparts` contains {bp} "
                "which is not one of the options in `use_bodyparts`."
            )

    if errors:
        for error in errors:
            logger.warning(error)
        return False
    return True


# ---- Modified version of the function from the main branch of keypoint_moseq  ----
def dj_generate_config(kpms_project_dir: str, **kwargs) -> tuple:
    """
    Generate or refresh `<kpms_project_dir>/kpms_dj_config.yml` from the KPMS base config.

    Behavior:
      - If the KPMS DJ config doesn't exist, start from the KPMS base `<kpms_project_dir>/config.yml`
        (created by keypoint_moseq's `setup_project`), then overlay kwargs and write KPMS DJ config.
      - If the KPMS DJ config exists, load it, overlay kwargs, and rewrite it.

    Args:
        kpms_project_dir: KPMS project output directory
        **kwargs: Key-value pairs to update in the config

    Returns:
        Tuple of (kpms_dj_config_path, kpms_dj_config_dict, kpms_base_config_path, kpms_base_config_dict)
    """
    kpms_project_dir = str(kpms_project_dir)
    kpms_base_config_path = _kpms_base_config_path(kpms_project_dir)
    kpms_dj_config_path = _kpms_dj_config_path(kpms_project_dir)

    # Load KPMS base config if it exists
    kpms_base_config_dict = None
    if Path(kpms_base_config_path).exists():
        with open(kpms_base_config_path, "r") as f:
            kpms_base_config_dict = yaml.safe_load(f)
        if kpms_base_config_dict is None:
            raise ValueError(
                f"Config file exists but is empty: {kpms_base_config_path}"
            )

    # Generate or update KPMS DJ config
    if Path(kpms_dj_config_path).exists():
        with open(kpms_dj_config_path, "r") as f:
            kpms_dj_config_dict = yaml.safe_load(f)
        if kpms_dj_config_dict is None:
            raise ValueError(f"Config file exists but is empty: {kpms_dj_config_path}")
    else:
        if not Path(kpms_base_config_path).exists():
            raise FileNotFoundError(
                f"Missing KPMS base config at {kpms_base_config_path}"
            )
        kpms_dj_config_dict = kpms_base_config_dict.copy()

    # Update bodyparts if provided
    if "bodyparts" in kwargs:
        kpms_dj_config_dict["bodyparts"] = list(kwargs["bodyparts"])

    if "use_bodyparts" in kwargs:
        use_bodyparts = list(kwargs["use_bodyparts"])
        kpms_dj_config_dict["use_bodyparts"] = use_bodyparts

        # Filter anterior/posterior to be subsets of use_bodyparts
        if "anterior_bodyparts" in kwargs:
            anterior = [
                bp for bp in kwargs["anterior_bodyparts"] if bp in use_bodyparts
            ]
            kwargs["anterior_bodyparts"] = anterior

        if "posterior_bodyparts" in kwargs:
            posterior = [
                bp for bp in kwargs["posterior_bodyparts"] if bp in use_bodyparts
            ]
            kwargs["posterior_bodyparts"] = posterior

    kpms_dj_config_dict.update(kwargs)

    if "skeleton" not in kpms_dj_config_dict:
        kpms_dj_config_dict["skeleton"] = []

    with open(kpms_dj_config_path, "w") as f:
        yaml.safe_dump(
            kpms_dj_config_dict,
            f,
            sort_keys=False,
            default_flow_style=False,
            allow_unicode=True,
        )

    return (
        kpms_dj_config_path,
        kpms_dj_config_dict,
        kpms_base_config_path,
        kpms_base_config_dict,
    )


# ---- Modified version of the viz function from the main branch of keypoint_moseq  ----
def load_kpms_dj_config(
    kpms_project_dir: str = None,
    config_path: str = None,
    check_if_valid: bool = True,
    build_indexes: bool = True,
) -> Dict[str, Any]:
    """
    Load kpms_dj_config.yml from either a KPMS project directory or a direct file path.

    Args:
        kpms_project_dir: KPMS project output directory containing kpms_dj_config.yml (optional)
        config_path: Direct path to kpms_dj_config.yml file (optional)
        check_if_valid: Check anatomy subset validity
        build_indexes: Add jax arrays 'anterior_idxs' and 'posterior_idxs'

    Returns:
        Configuration dictionary

    Raises:
        ValueError: If neither or both kpms_project_dir and config_path are provided
        FileNotFoundError: If the config file doesn't exist

    Mirrors keypoint_moseq.io.load_config behavior:
      - check_if_valid -> anatomy subset checks
      - build_indexes -> adds jax arrays 'anterior_idxs' and 'posterior_idxs'
        indexing into 'use_bodyparts' by order.
    """
    import jax.numpy as jnp

    if kpms_project_dir is None and config_path is None:
        raise ValueError("Either 'kpms_project_dir' or 'config_path' must be provided")
    if kpms_project_dir is not None and config_path is not None:
        raise ValueError("Cannot provide both 'kpms_project_dir' and 'config_path'")

    # Determine the config file path
    if config_path is not None:
        kpms_dj_cfg_path = config_path
    else:
        kpms_dj_cfg_path = _kpms_dj_config_path(kpms_project_dir)

    if not Path(kpms_dj_cfg_path).exists():
        raise FileNotFoundError(f"Missing DJ config at {kpms_dj_cfg_path}")

    with open(kpms_dj_cfg_path, "r") as f:
        cfg_dict = yaml.safe_load(f) or {}

    if check_if_valid:
        _check_config_validity(cfg_dict)

    if build_indexes:
        anterior = cfg_dict.get("anterior_bodyparts", [])
        posterior = cfg_dict.get("posterior_bodyparts", [])
        use_bps = cfg_dict.get("use_bodyparts", [])

        valid_anterior = [bp for bp in anterior if bp in use_bps]
        valid_posterior = [bp for bp in posterior if bp in use_bps]

        cfg_dict["anterior_idxs"] = jnp.array(
            [use_bps.index(bp) for bp in valid_anterior]
        )
        cfg_dict["posterior_idxs"] = jnp.array(
            [use_bps.index(bp) for bp in valid_posterior]
        )

    if "skeleton" not in cfg_dict:
        cfg_dict["skeleton"] = []

    return cfg_dict


# ---- Modified version of the viz function from the main branch of keypoint_moseq  ----
def update_kpms_dj_config(
    kpms_project_dir: str = None,
    config_dict: Dict[str, Any] = None,
    config_path: str = None,
    **kwargs,
) -> Dict[str, Any]:
    """
    Update kpms_dj_config with provided kwargs.
    This function updates the file on disk.
    This function returns the updated config dictionary.

    Args:
        kpms_project_dir: KPMS project output directory containing kpms_dj_config.yml (optional)
        config_dict: Existing config dictionary to update (optional)
        config_path: Direct path to config file to update (optional)
        **kwargs: Key-value pairs to update in the config

    Returns:
        Updated configuration dictionary

    Raises:
        ValueError: If neither kpms_project_dir, config_dict, nor config_path are provided

    If kpms_project_dir is provided, loads the config from file, updates it, saves it back, and returns it.
    If config_dict is provided, updates it directly and returns it (no file I/O).
    If config_path is provided, loads from that path, updates it, saves it back, and returns it.
    """

    if kpms_project_dir is None and config_dict is None and config_path is None:
        raise ValueError(
            "Either 'kpms_project_dir', 'config_dict', or 'config_path' must be provided"
        )

    if kpms_project_dir is not None:
        kpms_dj_cfg_path = _kpms_dj_config_path(kpms_project_dir)
        if not Path(kpms_dj_cfg_path).exists():
            raise FileNotFoundError(f"Missing DJ config at {kpms_dj_cfg_path}")

        with open(kpms_dj_cfg_path, "r") as f:
            cfg_dict = yaml.safe_load(f) or {}

        if "bodyparts" in kwargs:
            cfg_dict["bodyparts"] = list(kwargs.get("bodyparts"))

        if "use_bodyparts" in kwargs:
            use_bodyparts = list(kwargs.get("use_bodyparts"))
            cfg_dict["use_bodyparts"] = use_bodyparts
            # NOTE: skeleton is NOT modified - it remains from the base config

        cfg_dict.update(kwargs)

        with open(kpms_dj_cfg_path, "w") as f:
            yaml.safe_dump(
                cfg_dict,
                f,
                sort_keys=False,
                default_flow_style=False,
                allow_unicode=True,
            )
    elif config_path is not None:
        # Handle direct config_path
        if not Path(config_path).exists():
            raise FileNotFoundError(f"Missing config file at {config_path}")

        with open(config_path, "r") as f:
            cfg_dict = yaml.safe_load(f) or {}

        if "bodyparts" in kwargs:
            cfg_dict["bodyparts"] = list(kwargs.get("bodyparts"))

        if "use_bodyparts" in kwargs:
            use_bodyparts = list(kwargs.get("use_bodyparts"))
            cfg_dict["use_bodyparts"] = use_bodyparts
            # NOTE: skeleton is NOT modified - it remains from the base config

        cfg_dict.update(kwargs)

        with open(config_path, "w") as f:
            yaml.safe_dump(
                cfg_dict,
                f,
                sort_keys=False,
                default_flow_style=False,
                allow_unicode=True,
            )
    else:
        cfg_dict = config_dict.copy()

        if "bodyparts" in kwargs:
            cfg_dict["bodyparts"] = list(kwargs.get("bodyparts"))

        if "use_bodyparts" in kwargs:
            use_bodyparts = list(kwargs.get("use_bodyparts"))
            cfg_dict["use_bodyparts"] = use_bodyparts
            # NOTE: skeleton is NOT modified - it remains from the base config

        cfg_dict.update(kwargs)

    return cfg_dict


def extract_video_metadata(
    keypoint_videofile_metadata: List[Dict[str, Any]],
    get_kpms_root_data_dir,
    find_full_path,
) -> Tuple[Dict[int, Dict[str, Any]], int]:
    """
    Extract metadata (frame rate, file size, duration) for all videos and calculate average frame rate.

    Args:
        keypoint_videofile_metadata: List of dictionaries containing video metadata with
            'video_id' and 'video_path' keys
        get_kpms_root_data_dir: Function to get root data directory
        find_full_path: Function to resolve full paths

    Returns:
        Tuple of (video_metadata_dict, average_frame_rate) where:
            - video_metadata_dict: Dict mapping video_id to metadata dict with keys:
                - video_duration: Duration in minutes
                - frame_rate: Frame rate in fps
                - file_size: File size in MB
                - outlier_plot: None (placeholder for future use)
            - average_frame_rate: Average frame rate across all videos (int)

    Raises:
        ValueError: If no video files found or if video cannot be opened
    """
    if not keypoint_videofile_metadata:
        raise ValueError("No video files found in keypoint_videofile_metadata")

    video_metadata_dict = {}
    frame_rates = []

    for row in keypoint_videofile_metadata:
        video_id = int(row["video_id"])
        video_path = find_full_path(get_kpms_root_data_dir(), row["video_path"])

        # Get file size in MB (rounded to 2 decimal places)
        file_size_mb = round(video_path.stat().st_size / (1024 * 1024), 2)

        # Get video properties
        cap = cv2.VideoCapture(video_path.as_posix())
        if not cap.isOpened():
            raise ValueError(f"Could not open video {video_id} at {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        cap.release()

        # FPS validation and fallback
        if np.isnan(fps) or fps <= 0:
            logger.warning(
                f"Invalid FPS ({fps}) for video_id {video_id} at {video_path}"
            )
            logger.info("Attempting to extract FPS from video metadata...")

            # Try alternative method using ffprobe
            try:
                result = subprocess.run(
                    [
                        "ffprobe",
                        "-v",
                        "quiet",
                        "-select_streams",
                        "v:0",
                        "-show_entries",
                        "stream=r_frame_rate",
                        "-of",
                        "csv=p=0",
                        str(video_path),
                    ],
                    capture_output=True,
                    text=True,
                    timeout=10,
                )

                if result.returncode == 0 and result.stdout.strip():
                    fps_str = result.stdout.strip()
                    if "/" in fps_str:
                        num, den = fps_str.split("/")
                        fps = float(num) / float(den)
                    else:
                        fps = float(fps_str)
                    logger.info(f"Successfully extracted FPS using ffprobe: {fps}")
                else:
                    logger.warning("ffprobe failed, using default FPS")
                    fps = 30.0  # Default fallback
            except Exception as e:
                logger.warning(f"ffprobe failed: {e}, using default FPS")
                fps = 30.0  # Default fallback

            # Final validation after fallback attempts
            if np.isnan(fps) or fps <= 0:
                fps = 30.0  # Final fallback
                logger.warning(f"Using default FPS (30.0) for video_id {video_id}")

        # Calculate duration
        if frame_count > 0 and not np.isnan(frame_count):
            duration_minutes = int((frame_count / fps) / 60.0)
        else:
            duration_minutes = 0  # Unknown duration
            logger.warning(f"Could not determine frame count for video_id {video_id}")

        frame_rates.append(fps)
        video_metadata_dict[video_id] = {
            "video_duration": duration_minutes,
            "frame_rate": fps,
            "file_size": file_size_mb,
            "outlier_plot": None,
        }

    # Calculate average frame rate
    if len(frame_rates) > 0:
        # Remove any NaN or invalid values
        valid_frame_rates = [
            fps for fps in frame_rates if not np.isnan(fps) and fps > 0
        ]

        if len(valid_frame_rates) > 0:
            average_frame_rate = int(round(np.mean(valid_frame_rates)))
            logger.info(
                f"Calculated average frame rate: {average_frame_rate} fps from {len(valid_frame_rates)} videos"
            )
        else:
            average_frame_rate = 30  # Default fallback
            logger.warning("No valid frame rates found, using default value of 30 fps")
    else:
        average_frame_rate = 30  # Default fallback
        logger.warning("No frame rates found, using default value of 30 fps")

    return video_metadata_dict, average_frame_rate


def validate_video_directory(
    keypoint_videofile_metadata: List[Dict[str, Any]],
    get_kpms_root_data_dir,
    find_full_path,
) -> Path:
    """
    Validate that all videos are in the same directory and return the directory path.

    Args:
        keypoint_videofile_metadata: List of dictionaries containing video metadata with
            'video_path' keys
        get_kpms_root_data_dir: Function to get root data directory
        find_full_path: Function to resolve full paths

    Returns:
        Path to the videos directory (absolute path)

    Raises:
        ValueError: If videos are in multiple directories or no videos found
    """
    if not keypoint_videofile_metadata:
        raise ValueError("No video files found in keypoint_videofile_metadata")

    # Get all unique parent directories for all video files
    parent_dirs = {
        Path(video["video_path"]).parent for video in keypoint_videofile_metadata
    }
    # Check if there is only one unique parent
    if len(parent_dirs) > 1:
        raise ValueError(
            f"Videos are located in multiple directories: {parent_dirs}. All videos must be in the same directory."
        )

    videos_dir = find_full_path(
        get_kpms_root_data_dir(),
        Path(keypoint_videofile_metadata[0]["video_path"]).parent,
    )

    return videos_dir


def build_video_paths_dict(
    key, video_sequence_data, results, kpms_root, find_full_path
):
    """Build dictionary mapping video keys to video file paths.

    Args:
        key (dict): Primary key for querying RecordingSet.
        video_sequence_data (list): List of dictionaries with 'file' and 'file_path' keys.
        results (dict): Inference results dictionary with video keys.
        kpms_root (list): Root data directories.
        find_full_path (callable): Function to resolve relative paths to absolute paths.

    Returns:
        dict: Mapping of video keys to video file paths.
    """
    # Lazy import to avoid circular dependencies
    from .. import moseq_infer
    from ..plotting.viz_utils import extract_base_video_key

    if not video_sequence_data:
        vs_count = len((moseq_infer.MotionSequence.VideoSequence & key))
        logger.warning(
            f"No VideoSequence entries found in join for key: {key}. "
            f"Direct VideoSequence count: {vs_count}. "
            f"Ensure MotionSequence is populated."
        )
        return {}

    video_paths_dict = {}
    video_keys_from_results = list(results.keys())

    logger.info(
        f"Found {len(video_sequence_data)} VideoSequence entries. "
        f"Results.h5 has {len(video_keys_from_results)} video keys: {video_keys_from_results}"
    )

    recording_set_key = (moseq_infer.InferenceTask & key).fetch1("KEY")
    all_file_paths = (moseq_infer.RecordingSet.File & recording_set_key).fetch(
        "file_path"
    )

    logger.info(
        f"Found {len(all_file_paths)} total files in RecordingSet.File "
        f"for recording: {recording_set_key}"
    )

    stem_to_video_path = {}
    for file_path in all_file_paths:
        file_path_obj = Path(file_path)
        if file_path_obj.suffix.lower() in VIDEO_EXTENSIONS:
            stem = file_path_obj.stem
            try:
                full_video_path = find_full_path(kpms_root, file_path)
                if Path(full_video_path).exists():
                    stem_to_video_path[stem] = str(full_video_path)
                else:
                    logger.debug(f"Video file not found: {full_video_path}")
            except Exception as e:
                logger.debug(f"Could not resolve path for {file_path}: {e}")

    logger.info(
        f"Found {len(stem_to_video_path)} video files in RecordingSet.File: "
        f"{list(stem_to_video_path.keys())[:5]}..."
    )

    # Match video keys from VideoSequence entries to RecordingSet.File
    for entry in video_sequence_data:
        csv_file = entry["file"]
        csv_file_path = Path(csv_file)
        video_key = csv_file_path.stem

        if video_key not in video_keys_from_results:
            logger.warning(
                f"Video key '{video_key}' from CSV not found in results.h5 keys. "
                f"Skipping."
            )
            continue

        base_video_key = extract_base_video_key(video_key)
        video_path = None
        for stem, path in stem_to_video_path.items():
            # Try multiple matching strategies:
            # 1. Exact match (case-insensitive)
            # 2. Base video key exact match (case-insensitive) - most common case
            # 3. Prefix match: check if stem is a prefix of video_key or base_video_key
            #    (handles cases like "21_12_10_def6a_3.top.ir" matching "21_12_10_def6a_3.top.irDLC_...")
            # 4. Reverse prefix: check if base_video_key is a prefix of stem
            stem_lower = stem.lower()
            video_key_lower = video_key.lower()
            base_video_key_lower = base_video_key.lower()

            if (
                stem_lower == video_key_lower
                or stem_lower == base_video_key_lower
                or (
                    len(stem) > 5
                    and len(video_key) > 5
                    and video_key_lower.startswith(stem_lower)
                )
                or (
                    len(stem) > 5
                    and len(base_video_key) > 5
                    and base_video_key_lower.startswith(stem_lower)
                )
                or (
                    len(stem) > 5
                    and len(base_video_key) > 5
                    and stem_lower.startswith(base_video_key_lower)
                )
            ):
                video_path = path
                break

        if video_path:
            video_paths_dict[video_key] = video_path
            logger.info(
                f"Mapped video key '{video_key}' (base: '{base_video_key}') -> {video_path}"
            )
        else:
            logger.warning(
                f"✗ No video file found matching video key '{video_key}' (base: '{base_video_key}'). "
                f"Available stems: {list(stem_to_video_path.keys())[:5]}..."
            )

    if not video_paths_dict:
        logger.error(
            f"Failed to build video_paths_dict. "
            f"Video sequence entries: {len(video_sequence_data)}, "
            f"Video keys from results.h5: {video_keys_from_results}, "
            f"CSV file stems: {[Path(e['file']).stem for e in video_sequence_data]}"
        )

    return video_paths_dict


def prepare_fitting_data_and_config(
    config_path: Union[str, os.PathLike],
    pca_path: Union[str, os.PathLike],
    coordinates: Dict[str, np.ndarray],
    confidences: Dict[str, np.ndarray],
    use_bodyparts: List[str],
    average_frame_rate: float,
    latent_dim: int,
    kappa: float,
    get_kpms_processed_data_dir,
    find_full_path,
) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any], Any]:
    """
    Prepare data and config for model fitting.

    This function:
    1. Loads PCA from the provided path
    2. Formats keypoint data
    3. Estimates sigmasq_loc
    4. Updates and saves config with fitting parameters
    5. Reloads config with indexes for model initialization

    Args:
        config_path: Path to KPMS DJ config file (relative or absolute)
        pca_path: Path to PCA file (pca.p) - parent directory will be used
        coordinates: Dictionary of coordinates per recording
        confidences: Dictionary of confidences per recording
        use_bodyparts: List of bodyparts to use
        average_frame_rate: Average frame rate for sigmasq_loc estimation
        latent_dim: Latent dimension for model fitting
        kappa: Kappa value for model fitting
        get_kpms_processed_data_dir: Function to get processed data directory or processed data directory path
        find_full_path: Function to resolve full paths

    Returns:
        Tuple of (data, metadata, kpms_dj_config_dict, pca) where:
            - data: Formatted data dictionary
            - metadata: Metadata dictionary
            - kpms_dj_config_dict: Config dictionary with indexes built
            - pca: Loaded PCA object
    """
    from keypoint_moseq import estimate_sigmasq_loc, format_data, load_pca

    # Resolve config path
    kpms_dj_config_abs_path = find_full_path(get_kpms_processed_data_dir(), config_path)
    # Load PCA
    pca = load_pca(str(Path(pca_path).parent))

    # Format keypoint data
    data, metadata = format_data(
        coordinates=coordinates,
        confidences=confidences,
        use_bodyparts=use_bodyparts,
    )

    # Load config and estimate sigmasq_loc
    kpms_dj_config_dict = load_kpms_dj_config(
        config_path=kpms_dj_config_abs_path, build_indexes=False
    )

    sigmasq_loc_val = float(
        estimate_sigmasq_loc(data["Y"], data["mask"], filter_size=average_frame_rate)
    )

    # Update and save config with fitting parameters
    update_kpms_dj_config(
        config_dict=kpms_dj_config_dict,
        config_path=str(kpms_dj_config_abs_path),
        latent_dim=int(latent_dim),
        kappa=float(kappa),
        sigmasq_loc=sigmasq_loc_val,
    )

    # Reload config with indexes for model initialization
    kpms_dj_config_dict = load_kpms_dj_config(
        config_path=kpms_dj_config_abs_path, build_indexes=True
    )

    return data, metadata, kpms_dj_config_dict, pca


def find_checkpoint_file(model_name_full_path: Union[str, os.PathLike]) -> Path:
    """Find the most recent checkpoint file in the model directory.

    Args:
        model_name_full_path: Path to the model directory

    Returns:
        Path to the most recent checkpoint file

    Raises:
        FileNotFoundError: If no checkpoint files are found
    """
    model_name_full_path = Path(model_name_full_path)
    checkpoint_files = []
    for pattern in ("checkpoint*", "*.h5"):
        checkpoint_files.extend(model_name_full_path.glob(pattern))
    if checkpoint_files:
        return max(checkpoint_files, key=lambda f: f.stat().st_mtime)
    raise FileNotFoundError(f"No checkpoint files found in {model_name_full_path}")


def initialize_model_for_fitting(
    data: Dict[str, Any],
    metadata: Dict[str, Any],
    pca: Any,
    kpms_dj_config_dict: Dict[str, Any],
    pre_model: Union[str, Path, None],
    full_kappa: float,
    full_latent_dim: int,
) -> Any:
    """Initialize model for fitting, using prefit if available.

    Args:
        data: Formatted keypoint data
        metadata: Metadata dictionary
        pca: PCA object
        kpms_dj_config_dict: KPMS config dictionary
        pre_model: Path to prefit model file (if available)
        full_kappa: Kappa value for model fitting
        full_latent_dim: Latent dimension for model fitting

    Returns:
        Initialized model ready for fitting

    Raises:
        ValueError: If model initialization fails
    """
    import jax_moseq
    from keypoint_moseq import init_model, update_hypparams

    if pre_model is not None:
        return pre_model

    data = jax_moseq.utils.debugging.convert_data_precision(data)
    model_to_fit = init_model(
        data=data, metadata=metadata, pca=pca, **kpms_dj_config_dict
    )
    model_to_fit = update_hypparams(
        model_to_fit,
        kappa=float(full_kappa),
        latent_dim=int(full_latent_dim),
    )
    return model_to_fit


def find_prefit_model(
    prefit_task_table,
    prefit_file_table,
    key: Dict[str, Any],
    full_kappa: float,
    full_latent_dim: int,
) -> Union[str, Path, None]:
    """Find the best PreFit model to use as warm start.

    Args:
        prefit_task_table: PreFitTask DataJoint table
        prefit_file_table: PreFit.File DataJoint table
        key: Primary key for querying
        full_kappa: Kappa value to match
        full_latent_dim: Latent dimension to match

    Returns:
        Path to prefit model file if found, None otherwise
    """
    best_prefit_key = (
        prefit_task_table
        & key
        & {
            "pre_kappa": full_kappa,
            "pre_latent_dim": full_latent_dim,
        }
    ).fetch("KEY", order_by="pre_num_iterations desc", limit=1, as_dict=True)

    if best_prefit_key:
        pre_model_key = best_prefit_key[0]
        prefit_file_query = (
            prefit_file_table & pre_model_key & 'file_name="model_data.pkl"'
        )
        if prefit_file_query:
            pre_model = prefit_file_query.fetch1("file_path")
            logger.info(f"Using PreFit model {pre_model_key} as warm start for FullFit")
            return pre_model
        else:
            logger.info(
                f"PreFit model key {pre_model_key} found but model_data.pkl file not found. "
                "Initializing model from scratch."
            )
    else:
        logger.info(
            f"No PreFit tasks found matching kappa={full_kappa}, "
            f"latent_dim={full_latent_dim} for key {key}. "
            "Initializing model from scratch."
        )
    return None


def compute_syllable_metrics(
    checkpoint_file: Union[str, os.PathLike], fps: float
) -> Dict[str, Any]:
    """Compute syllable quality metrics from checkpoint file.

    This function extracts syllable sequences from a checkpoint file and computes
    aggregate statistics about syllable durations, which are useful for evaluating
    model quality and determining appropriate kappa values.

    Args:
        checkpoint_file: Path to checkpoint.h5 file
        fps: Frames per second for converting durations to seconds

    Returns:
        Dictionary with essential syllable quality metrics:
            - num_syllables: Number of unique syllables discovered
            - median_syllable_duration_ms: Median duration in milliseconds (KEY METRIC - target: 400ms)

    Raises:
        FileNotFoundError: If checkpoint file doesn't exist
        ValueError: If checkpoint doesn't contain syllable sequences
    """
    from keypoint_moseq import load_checkpoint

    checkpoint_path = Path(checkpoint_file)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_file}")

    # Load checkpoint
    model, data, _, _ = load_checkpoint(path=str(checkpoint_path))

    # Extract syllable sequences (z) from all videos
    if "states" not in model or "z" not in model["states"]:
        raise ValueError(
            "Checkpoint does not contain syllable sequences (model['states']['z'])"
        )

    z_sequences = model["states"]["z"]

    # Check if z_sequences is empty
    # Handle both dict and array cases
    if isinstance(z_sequences, dict):
        if len(z_sequences) == 0:
            raise ValueError("No syllable sequences found in checkpoint")
    elif isinstance(z_sequences, np.ndarray):
        if z_sequences.size == 0:
            raise ValueError("No syllable sequences found in checkpoint")
        # Convert single array to dict format for consistency
        z_sequences = {"video_0": z_sequences}
    else:
        # Try to get length, if that fails, it's probably empty
        try:
            if len(z_sequences) == 0:
                raise ValueError("No syllable sequences found in checkpoint")
        except (TypeError, ValueError):
            raise ValueError(
                f"Unexpected format for z_sequences: {type(z_sequences)}. "
                "Expected dict or numpy array."
            )

    # Compute durations for all syllables across all videos
    all_durations = []

    # Track unique syllable IDs (only needed for counting unique syllables)
    unique_syllable_ids = set()

    for video_key, z in z_sequences.items():
        # Convert to numpy array if needed
        z_array = np.array(z).flatten()

        if len(z_array) == 0:
            continue

        # Find syllable boundaries (where syllable changes)
        if len(z_array) > 1:
            # Find transitions between syllables
            # np.diff finds where consecutive elements differ
            # +1 because diff returns indices of the first element of each pair
            boundaries = np.where(np.diff(z_array) != 0)[0] + 1

            if len(boundaries) > 0:
                # Compute durations between boundaries
                # segment_starts: [0, boundary1, boundary2, ...]
                # segment_ends: [boundary1, boundary2, ..., len(z_array)]
                segment_starts = np.concatenate([[0], boundaries])
                segment_ends = np.concatenate([boundaries, [len(z_array)]])
                durations = segment_ends - segment_starts

                # Track unique syllable IDs
                unique_syllable_ids.update(z_array[segment_starts].astype(int))

                all_durations.extend(durations.tolist())
            else:
                # Single syllable for entire video (no transitions)
                syllable_id = int(z_array[0])
                unique_syllable_ids.add(syllable_id)
                duration = int(len(z_array))
                all_durations.append(duration)
        else:
            # Single frame
            syllable_id = int(z_array[0])
            unique_syllable_ids.add(syllable_id)
            all_durations.append(1)

    if len(all_durations) == 0:
        raise ValueError("No syllable durations could be computed from checkpoint")

    if len(unique_syllable_ids) == 0:
        raise ValueError("No syllables found in checkpoint")

    all_durations = np.array(all_durations, dtype=float)

    # Compute essential statistics
    num_unique_syllables = len(unique_syllable_ids)
    median_duration_frames = float(np.median(all_durations))
    median_duration_ms = (median_duration_frames / fps) * 1000.0

    aggregate_metrics = {
        "num_syllables": num_unique_syllables,
        "median_syllable_duration_ms": median_duration_ms,
    }

    return aggregate_metrics
