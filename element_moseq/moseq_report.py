"""
DataJoint Schema for Keypoint-MoSeq reporting and visualization
"""

import importlib
import inspect
import pickle
from datetime import datetime, timezone
from pathlib import Path

import datajoint as dj
import h5py
import numpy as np
from element_interface.utils import find_full_path
from matplotlib import pyplot as plt

from . import moseq_infer, moseq_train
from .plotting.viz_utils import (
    MIN_DURATION,
    MIN_FREQUENCY,
    extract_syllable_id_from_filename,
    find_grid_movie_files,
    find_trajectory_files,
)
from .readers import kpms_reader
from .readers.kpms_reader import build_video_paths_dict

schema = dj.schema()
_linking_module = None
logger = dj.logger


def activate(
    report_schema_name: str,
    *,
    create_schema: bool = True,
    create_tables: bool = True,
    linking_module: str = None,
):
    """Activate this schema.

    Args:
        report_schema_name (str): Schema name on the database server to activate the `moseq_report` schema.
        create_schema (bool): When True (default), create schema in the database if it
                            does not yet exist.
        create_tables (bool): When True (default), create schema tables in the database
                             if they do not yet exist.
        linking_module (str): A module (or name) containing the required dependencies.

    Functions:
        get_kpms_root_data_dir(): Returns absolute path for root data director(y/ies) with all behavioral recordings, as (list of) string(s)
        get_kpms_processed_data_dir(): Optional. Returns absolute path for processed data.
    """

    if isinstance(linking_module, str):
        linking_module = importlib.import_module(linking_module)
    assert inspect.ismodule(
        linking_module
    ), "The argument 'linking_module' must be a module or module name"

    # activate
    schema.activate(
        report_schema_name,
        create_schema=create_schema,
        create_tables=create_tables,
        add_objects=linking_module.__dict__,
    )


@schema
class BehavioralSummary(dj.Computed):
    """Generate and store behavioral analysis visualizations from Keypoint-MoSeq inference.

    Attributes:
        Inference (foreign key)              : `Inference` key.
        syllable_frequencies_plot (attach)   : File path of the syllable frequencies plot.
        similarity_dendrogram_png (attach)   : File path of the similarity dendrogram plot (PNG).
        similarity_dendrogram_pdf (attach)   : File path of the similarity dendrogram plot (PDF).
    """

    definition = """
    -> moseq_infer.Inference
    ---
    syllable_frequencies_plot   : attach # File path of the syllable frequencies plot
    similarity_dendrogram_png   : attach # File path of the similarity dendrogram plot (PNG)
    similarity_dendrogram_pdf   : attach # File path of the similarity dendrogram plot (PDF)
    """

    def make(self, key):
        """Generate behavioral summary visualizations.

        High-Level Logic:
        1. Fetch model directory and inference output directory.
        2. Load inference results from HDF5 file.
        3. Generate syllable frequencies plot and save as PNG.
        4. Fetch model training data (coordinates, bodyparts, frame rate).
        5. Generate similarity dendrogram plots (PNG and PDF).
        6. Insert visualization file paths into database.
        """

        from keypoint_moseq import (
            format_data,
            plot_similarity_dendrogram,
            plot_syllable_frequencies,
        )

        model_dir = (moseq_infer.Model & key).fetch1("model_dir")
        kpms_processed = moseq_train.get_kpms_processed_data_dir()
        inference_output_dir = (moseq_infer.InferenceTask & key).fetch1(
            "inference_output_dir"
        )
        inference_output_dir = Path(model_dir) / inference_output_dir
        inference_output_dir = find_full_path(kpms_processed, inference_output_dir)

        # Get inference data from upstream tables
        results_file = (moseq_infer.Inference & key).fetch1(
            "syllable_segmentation_file"
        )
        results = h5py.File(results_file, "r")

        # Generate syllable frequencies plot
        fig, _ = plot_syllable_frequencies(results=results, path=inference_output_dir)
        fig.savefig(inference_output_dir / "syllable_frequencies.png")
        plt.close(fig)

        # Generate similarity dendrogram plots
        model_key = (moseq_infer.Model * moseq_train.SelectedFullFit & key).fetch1(
            "KEY"
        )
        coordinates = (moseq_train.PreProcessing & model_key).fetch1("coordinates")
        use_bodyparts = (moseq_train.BodyParts & model_key).fetch1("use_bodyparts")
        fps = (moseq_train.PreProcessing & model_key).fetch1("average_frame_rate")

        plot_similarity_dendrogram(
            coordinates=coordinates,
            results=results,
            save_path=(inference_output_dir / "similarity_dendrogram").as_posix(),
            use_bodyparts=use_bodyparts,
            fps=fps,
        )

        # Insert the record
        self.insert1(
            {
                **key,
                "syllable_frequencies_plot": inference_output_dir
                / "syllable_frequencies.png",
                "similarity_dendrogram_png": inference_output_dir
                / "similarity_dendrogram.png",
                "similarity_dendrogram_pdf": inference_output_dir
                / "similarity_dendrogram.pdf",
            }
        )


@schema
class TrajectoryPlot(dj.Computed):
    """Generate per-syllable trajectory plots and grid movies for behavioral syllable analysis.

    Attributes:
        MotionSequence (foreign key)        : `MotionSequence` key.
        all_trajectories_gif (attach)       : File path of the all trajectories GIF plot.
        all_trajectories_pdf (attach)       : File path of the all trajectories PDF plot.
        traj_duration (float)               : Time duration (seconds) of trajectory plot generation.
    """

    definition = """
    -> moseq_infer.MotionSequence
    ---
    all_trajectories_gif        : attach # File path of the all trajectories GIF plot
    all_trajectories_pdf        : attach # File path of the all trajectories PDF plot
    traj_duration=NULL          : float  # Time duration (seconds)
    """

    class Syllable(dj.Part):
        """Store per-syllable trajectory plots and grid movies.

        Attributes:
            TrajectoryPlot (foreign key)    : `TrajectoryPlot` key.
            syllable_id (int)               : Syllable ID.
            plot_gif (attach)               : GIF plot file for this syllable.
            plot_pdf (attach)               : PDF plot file for this syllable.
            grid_movie (attach)             : Grid movie file for this syllable.
        """

        definition = """
        -> master
        syllable_id: int # Syllable ID
        ---
        plot_gif: attach # GIF plot file
        plot_pdf: attach # PDF plot file
        grid_movie: attach # Grid movie file
        """

    def make_fetch(self, key):
        """Fetch data from upstream tables.

        High-Level Logic:
        1. Fetch model information (model_dir, model_key, use_bodyparts, config).
        2. Fetch inference data (output_dir, coordinates_file, results_file, fps).
        3. Join MotionSequence.VideoSequence with RecordingSet.File to get video file paths.
        4. Fetch keypointset_dir as fallback for grid movie generation.
        5. Construct output directory path.
        6. Return all fetched data for use in make_compute.
        """
        kpms_processed = moseq_train.get_kpms_processed_data_dir()
        kpms_root = moseq_train.get_kpms_root_data_dir()

        # From trained model
        model_dir = (moseq_infer.Model & key).fetch1("model_dir")
        model_dir = find_full_path(kpms_processed, model_dir)
        model_dir = Path(model_dir)
        model_key = (moseq_infer.Model * moseq_train.SelectedFullFit & key).fetch1(
            "KEY"
        )
        use_bodyparts = (moseq_train.BodyParts & model_key).fetch1("use_bodyparts")
        kpms_dj_config_path = (moseq_train.FullFit.ConfigFile & model_key).fetch1(
            "config_file"
        )

        # From new recordings
        inference_output_dir = (moseq_infer.InferenceTask & key).fetch1(
            "inference_output_dir"
        )
        coordinates_file = (moseq_infer.Inference & key).fetch1("coordinates_file")
        results_file = (moseq_infer.Inference & key).fetch1(
            "syllable_segmentation_file"
        )
        fps = (moseq_infer.Inference & key).fetch1("average_frame_rate")

        # Join with RecordingSet.File to get file_paths
        video_sequence_join = (
            moseq_infer.MotionSequence.VideoSequence * moseq_infer.RecordingSet.File
        ) & key

        # Fetch all needed attributes
        video_sequence_data = video_sequence_join.fetch(
            "file", "file_path", as_dict=True
        )

        # kpset_dir still needed as fallback for generate_grid_movies
        kpset_dir = (moseq_infer.InferenceTask & key).fetch1("keypointset_dir")
        kpset_dir = find_full_path(kpms_root, kpset_dir)
        kpset_dir = Path(kpset_dir).as_posix()

        # Construct output directory
        output_dir = Path(model_dir) / inference_output_dir
        output_dir = Path(find_full_path(kpms_processed, output_dir))

        return (
            model_dir,
            model_key,
            use_bodyparts,
            kpms_dj_config_path,
            inference_output_dir,
            coordinates_file,
            results_file,
            fps,
            video_sequence_data,
            kpset_dir,
            output_dir,
        )

    def make_compute(
        self,
        key,
        model_dir,
        model_key,
        use_bodyparts,
        kpms_dj_config_path,
        inference_output_dir,
        coordinates_file,
        results_file,
        fps,
        video_sequence_data,
        kpset_dir,
        output_dir,
    ):
        """Generate trajectory plots and grid movies, and find generated files.

        High-Level Logic:
        1. Load KPMS configuration, coordinates, and inference results.
        2. Build video_paths_dict by matching video keys from results to RecordingSet.File entries.
        3. Create output directories for trajectory plots and grid movies.
        4. Generate trajectory plots (GIF and PDF) for all syllables.
        5. Generate grid movies (MP4) using video_paths_dict or fallback to kpset_dir.
        6. Search recursively for generated files (GIFs, PDFs, MP4s) in output directories.
        7. Extract syllable IDs from filenames and build file path mappings.
        8. Identify syllables with all required files (GIF, PDF, MP4).
        9. Return file paths and syllable sets for insertion.
        """
        from keypoint_moseq import (
            generate_grid_movies,
            generate_trajectory_plots,
            load_hdf5,
        )

        start_time = datetime.now(timezone.utc)

        kpms_root = moseq_train.get_kpms_root_data_dir()

        # Load kpms config
        kpms_dj_config_dict = kpms_reader.load_kpms_dj_config(
            config_path=kpms_dj_config_path
        )

        # Load coordinates and results
        with open(coordinates_file, "rb") as f:
            coordinates = pickle.load(f)
        results = load_hdf5(results_file)

        # Build video_paths_dict
        video_paths_dict = build_video_paths_dict(
            key, video_sequence_data, results, kpms_root, find_full_path
        )

        # Create output directories
        trajectory_dir = output_dir / "trajectory_plots"
        grid_movies_dir = output_dir / "grid_movies"
        trajectory_dir.mkdir(parents=True, exist_ok=True)
        grid_movies_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Generating trajectory plots for {key}")
        generate_trajectory_plots(
            coordinates=coordinates,
            results=results,
            output_dir=trajectory_dir.as_posix(),
            use_bodyparts=use_bodyparts,
            fps=fps,
            skeleton=kpms_dj_config_dict.get("skeleton", []),
            min_frequency=MIN_FREQUENCY,
            min_duration=MIN_DURATION,
        )

        logger.info(f"Generating grid movies for {key}")
        # Generate grid movies
        # Use video_paths if available and complete, otherwise fall back to video_dir
        grid_movies_kwargs = {
            "results": results,
            "coordinates": coordinates,
            "output_dir": grid_movies_dir.as_posix(),
            "use_bodyparts": use_bodyparts,
            "fps": fps,
            "overlay_keypoints": False,
        }

        # Check if video_paths_dict contains all keys from results
        results_keys = set(results.keys())
        video_paths_keys = set(video_paths_dict.keys()) if video_paths_dict else set()
        missing_keys = results_keys - video_paths_keys

        if video_paths_dict and not missing_keys:
            # All keys have video paths, use video_paths
            grid_movies_kwargs["video_paths"] = video_paths_dict
            logger.info(
                f"Using video_paths with {len(video_paths_dict)} entries for grid movies"
            )
        elif video_paths_dict and missing_keys:
            # Some keys have video paths, use video_paths for matched keys only
            # Filter results to only include keys with video paths
            filtered_results = {
                k: v for k, v in results.items() if k in video_paths_dict
            }
            filtered_coordinates = {
                k: v for k, v in coordinates.items() if k in video_paths_dict
            }

            grid_movies_kwargs["video_paths"] = video_paths_dict
            grid_movies_kwargs["results"] = filtered_results
            grid_movies_kwargs["coordinates"] = filtered_coordinates

            logger.warning(
                f"Missing video paths for {len(missing_keys)} keys: {sorted(missing_keys)[:10]}{'...' if len(missing_keys) > 10 else ''}. "
                f"Videos exist but are not registered in RecordingSet.File. "
                f"Please insert the missing video files into RecordingSet.File for key: {key}. "
                f"Generating grid movies for {len(video_paths_dict)} matched keys only. "
                f"Skipping {len(missing_keys)} keys without video files."
            )
        else:
            # No video_paths_dict available - videos are not registered in RecordingSet.File
            logger.warning(
                f"No video_paths_dict available. "
                f"Videos exist but are not registered in RecordingSet.File. "
                f"Please insert the missing video files into RecordingSet.File for key: {key}. "
                f"Results keys needing videos: {sorted(results_keys)[:10]}{'...' if len(results_keys) > 10 else ''}. "
                f"Total keys: {len(results_keys)}. "
                f"Generating grid movies in keypoints_only mode (no video overlay)."
            )
            # Set keypoints_only mode to avoid video requirement
            grid_movies_kwargs["keypoints_only"] = True
            grid_movies_kwargs["video_dir"] = kpset_dir

        # Build video_frame_indexes to handle frame count mismatches between
        # DLC coordinate files and actual video files (see keypoint-moseq#149).
        # Without this, generate_grid_movies assumes a 1:1 frame mapping
        # which causes ValueError when coordinates have more frames than video.
        used_video_paths = grid_movies_kwargs.get("video_paths")
        used_results = grid_movies_kwargs.get("results", results)
        if used_video_paths:
            import cv2

            video_frame_indexes = {}
            for vkey, vpath in used_video_paths.items():
                cap = cv2.VideoCapture(vpath)
                n_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                cap.release()
                n_result_frames = len(used_results[vkey]["syllable"])
                if n_result_frames > n_video_frames:
                    logger.warning(
                        f"Frame count mismatch for '{vkey}': "
                        f"results have {n_result_frames} frames but video has "
                        f"{n_video_frames} frames. Clamping frame indices to video bounds."
                    )
                    video_frame_indexes[vkey] = np.minimum(
                        np.arange(n_result_frames), n_video_frames - 1
                    )
                else:
                    video_frame_indexes[vkey] = np.arange(n_result_frames)
            grid_movies_kwargs["video_frame_indexes"] = video_frame_indexes

        generate_grid_movies(**grid_movies_kwargs)

        # Calculate duration
        duration_seconds = (datetime.now(timezone.utc) - start_time).total_seconds()

        # Find which syllables have all required files generated
        # Even with matching filtering parameters, some syllables may not have files
        # due to internal requirements in generate_trajectory_plots (e.g., n_neighbors)

        # Ensure directories are Path objects
        trajectory_dir = Path(trajectory_dir)
        grid_movies_dir = Path(grid_movies_dir)

        # Find trajectory GIF files
        logger.info(f"Checking for trajectory GIFs in: {trajectory_dir}")
        trajectory_gifs, trajectory_gif_paths = find_trajectory_files(
            trajectory_dir, ".gif", exclude_name="all_trajectories.gif"
        )
        logger.info(
            f"Found {len(trajectory_gifs)} trajectory GIF files (syllables: {sorted(trajectory_gifs)[:10] if trajectory_gifs else 'none'}...)"
        )

        # Find trajectory PDF files
        logger.info(f"Checking for trajectory PDFs in: {trajectory_dir}")
        trajectory_pdfs, trajectory_pdf_paths = find_trajectory_files(
            trajectory_dir, ".pdf", exclude_name="all_trajectories.pdf"
        )
        logger.info(
            f"Found {len(trajectory_pdfs)} trajectory PDF files (syllables: {sorted(trajectory_pdfs)[:10] if trajectory_pdfs else 'none'}...)"
        )

        # Find grid movie MP4 files
        logger.info(f"Checking for grid movies in: {grid_movies_dir}")
        grid_movie_mp4s, grid_movie_mp4_paths = find_grid_movie_files(grid_movies_dir)
        logger.info(
            f"Found {len(grid_movie_mp4s)} grid movie MP4 files (syllables: {sorted(grid_movie_mp4s)[:10]}...)"
        )

        # Only insert syllables that have all three file types
        syllables_with_all_files = trajectory_gifs & trajectory_pdfs & grid_movie_mp4s

        # Log warning if some sampled syllables don't have complete files
        sampled_syllables = set(
            (moseq_infer.MotionSequence.SampledInstance & key).fetch("syllable")
        )
        missing_syllables = sampled_syllables - syllables_with_all_files
        if missing_syllables:
            logger.warning(
                f"Skipping {len(missing_syllables)} syllables without complete files: "
                f"{sorted(missing_syllables)}"
            )

        return (
            duration_seconds,
            trajectory_dir,
            grid_movies_dir,
            trajectory_gif_paths,
            trajectory_pdf_paths,
            grid_movie_mp4_paths,
            syllables_with_all_files,
        )

    def make_insert(
        self,
        key,
        duration_seconds,
        trajectory_dir,
        grid_movies_dir,
        trajectory_gif_paths,
        trajectory_pdf_paths,
        grid_movie_mp4_paths,
        syllables_with_all_files,
    ):
        """Insert trajectory plot results into the database.

        High-Level Logic:
        1. Insert main TrajectoryPlot entry with all_trajectories files and duration.
        2. Insert per-syllable entries (Syllable part table) for syllables with complete files.
        3. Only insert syllables that have all three file types (GIF, PDF, MP4).
        """
        self.insert1(
            {
                **key,
                "all_trajectories_gif": trajectory_dir / "all_trajectories.gif",
                "all_trajectories_pdf": trajectory_dir / "all_trajectories.pdf",
                "traj_duration": duration_seconds,
            }
        )

        for syllable in sorted(syllables_with_all_files):
            self.Syllable.insert1(
                {
                    **key,
                    "syllable_id": syllable,
                    "plot_gif": trajectory_gif_paths[syllable],
                    "plot_pdf": trajectory_pdf_paths[syllable],
                    "grid_movie": grid_movie_mp4_paths[syllable],
                }
            )
