import os
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union

import cv2
import datajoint as dj
import numpy as np
import yaml

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
            kpms_base_config_dict = yaml.safe_load(f) or {}

    # Generate or update KPMS DJ config
    if Path(kpms_dj_config_path).exists():
        with open(kpms_dj_config_path, "r") as f:
            kpms_dj_config_dict = yaml.safe_load(f) or {}
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
