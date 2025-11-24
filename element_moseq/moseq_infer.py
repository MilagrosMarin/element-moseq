"""
Code adapted from the Datta Lab: https://dattalab.github.io/moseq2-website/index.html
DataJoint Schema for Keypoint-MoSeq inference pipeline
"""

import importlib
import inspect
import pickle
from datetime import datetime, timezone
from pathlib import Path

import datajoint as dj
import numpy as np
from element_interface.utils import find_full_path

from . import moseq_train
from .plotting.viz_utils import (
    MIN_DURATION,
    MIN_FREQUENCY,
    match_video_keys_to_file_paths,
)
from .readers import kpms_reader

schema = dj.schema()
_linking_module = None
logger = dj.logger


def activate(
    infer_schema_name: str,
    *,
    create_schema: bool = True,
    create_tables: bool = True,
    linking_module: str = None,
):
    """Activate this schema.

    Args:
        infer_schema_name (str): Schema name on the database server to activate the `moseq_infer` schema.
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
    ), "The argument 'linking_module' must be a module's name or a module"
    assert hasattr(
        linking_module, "get_kpms_root_data_dir"
    ), "The linking module must specify a lookup function for a root data directory"

    global _linking_module
    _linking_module = linking_module

    schema.activate(
        infer_schema_name,
        create_schema=create_schema,
        create_tables=create_tables,
        add_objects=_linking_module.__dict__,
    )


# ----------------------------- Table declarations ----------------------


@schema
class Model(dj.Manual):
    """Register a trained model.

    Attributes:
        model_id (int)                      : Unique ID for each model.
        model_name (varchar)                : User-friendly model name.
        model_dir (varchar)                 : Model directory relative to processed data directory.
        model_file (filepath)               : Checkpoint file (h5 format).
        model_desc (varchar)                : Optional. User-defined description of the model.
        SelectedFullFit (foreign key)       : Optional. FullFit key linking to training pipeline.

    """

    definition = """
    model_id                : int             # Unique ID for each model
    ---
    model_name              : varchar(1000)   # User-friendly model name
    model_dir               : varchar(1000)   # Model directory relative to root data directory
    model_file              : filepath@moseq-infer-processed        # Checkpoint file (h5 format)
    model_desc=''           : varchar(1000)   # Optional. User-defined description of the model
    -> [nullable] moseq_train.SelectedFullFit # Optional. FullFit key.
    """


@schema
class RecordingSet(dj.Manual):
    """Explicit list of files (video recordings and pose estimation keypoint data) to use for inference.

    Attributes:
        Session (foreign key)               : `Session` key.
        recording_id (int)                  : Unique ID for each recording.
        Device (foreign key)                : Device primary key.
    """

    definition = """
    -> Session                             # `Session` key
    recording_id: int                      # Unique ID for each recording
    ---
    -> Device                              # Device primary key
    """

    class File(dj.Part):
        """File IDs and paths associated with a given `recording_id`.

        Attributes:
            RecordingSet (foreign key)   : `RecordingSet` key.
            file_id (int)                : Unique ID for each file.
            file_path (varchar)          : Filepath of each video, keypoint file, or config file,
                                          relative to root data directory.
        """

        definition = """
        -> master
        file_id: int                              # Unique ID for each file
        ---
        file_path: varchar(1000) # Filepath of each video, keypoint file, and config file, relative to root data directory.
        """


@schema
class InferenceTask(dj.Manual):
    """Staging table to define the Inference task and its output directory.

    Attributes:
        RecordingSet (foreign key)         : `RecordingSet` key
        Model (foreign key)                  : `Model` key
        PoseEstimationMethod (foreign key)   : Pose estimation method used for the specified `recording_id`.
        inference_output_dir (varchar)       : Optional. Sub-directory where the results will be stored.
        inference_desc (varchar)             : Optional. User-defined description of the inference task.
        num_iterations (int)                 : Optional. Number of iterations to use for the model inference. If null, the default number internally is 500.
        task_mode (enum)                     : 'load': load computed analysis results, 'trigger': trigger computation
    """

    definition = """
    -> RecordingSet                                         # `RecordingSet` key
    -> Model                                                # `Model` key
    ---
    -> moseq_train.PoseEstimationMethod                     # Pose estimation method used for the specified `recording_id`
    keypointset_dir               : varchar(1000)           # Keypointset directory where the files for RecordingSet.File are stored
    inference_output_dir=''       : varchar(1000)           # Optional. Sub-directory where the results will be stored
    inference_desc=''             : varchar(1000)           # Optional. User-defined description of the inference task
    num_iterations=NULL           : int                     # Optional. Number of iterations to use for the model inference. If null, the default number internally is 500.
    task_mode='load'              : enum('load', 'trigger') # Task mode for the inference task
    """

    @classmethod
    def infer_output_dir(cls, key: dict, relative: bool = False, mkdir: bool = False):
        """Return the expected inference_output_dir.

        Based on convention: model_dir / inference_output_dir
        If inference_output_dir is empty, generates a default based on model and recording.

        Args:
            key: DataJoint key specifying a pairing of RecordingSet and Model.
            relative (bool): Report directory relative to processed data directory.
            mkdir (bool): Default False. Make directory if it doesn't exist.
        """
        # Get model directory
        model_dir_rel = (Model * moseq_train.SelectedFullFit & key).fetch1("model_dir")
        kpms_processed = moseq_train.get_kpms_processed_data_dir()

        # Build default output directory name based on RecordingSet and Session keys
        recording_set_key = (RecordingSet & key).fetch1("KEY")
        recording_id = recording_set_key["recording_id"]
        session_parts = []
        for field_name, field_value in sorted(recording_set_key.items()):
            if field_name != "recording_id":
                # Format field values for directory name (handle datetime, strings, etc.)
                if isinstance(field_value, datetime):
                    # Format datetime as YYYYMMDD_HHMMSS for readability
                    session_parts.append(
                        f"{field_name}_{field_value.strftime('%Y%m%d_%H%M%S')}"
                    )
                else:
                    session_parts.append(f"{field_name}_{field_value}")

        # Create a unique directory name using all primary key components
        # Format: inference_<session_fields>_recording_id_<recording_id>
        # This ensures uniqueness within the same model_dir
        if session_parts:
            default_output_dir = (
                f"inference_{'_'.join(session_parts)}_recording_id_{recording_id}"
            )
        else:
            default_output_dir = f"inference_recording_id_{recording_id}"

        if mkdir:
            output_dir = Path(kpms_processed) / model_dir_rel / default_output_dir
            output_dir.mkdir(parents=True, exist_ok=True)

        return default_output_dir


@schema
class Inference(dj.Computed):
    """Infer the model from the checkpoint file and generate the results of segmenting continuous behavior into discrete syllables.

    Attributes:
        InferenceTask (foreign key)          : `InferenceTask` key.
        average_frame_rate (float)           : Average frame rate of the videos for inference.
        coordinates_file (filepath)          : File path of cleaned coordinates dictionary after outlier removal.
        confidences_file (filepath)         : File path of cleaned confidences dictionary after outlier removal.
        syllable_segmentation_file (filepath): File path of the syllable analysis results (HDF5 format) containing syllable labels, latent states, centroids, and headings.
        inference_duration (float)           : Time duration (seconds) of the inference computation.
    """

    definition = """
    -> InferenceTask                            # `InferenceTask` key
    ---
    average_frame_rate              : float       # Average frame rate of the videos for inference.
    coordinates_file                : filepath@moseq-infer-processed  # Cleaned coordinates dictionary after outlier removal.
    confidences_file                : filepath@moseq-infer-processed  # Cleaned confidences dictionary after outlier removal.
    syllable_segmentation_file      : filepath@moseq-infer-processed    # File path of the syllable analysis results (HDF5 format) containing syllable labels, latent states, centroids, and headings
    inference_duration=NULL         : float     # Time duration (seconds) of the inference computation
    """

    def make_fetch(self, key):
        """Fetch required data for inference from database tables."""
        (
            keypointset_dir,
            inference_output_dir,
            num_iterations,
            task_mode,
            pose_estimation_method,
        ) = (InferenceTask & key).fetch1(
            "keypointset_dir",
            "inference_output_dir",
            "num_iterations",
            "task_mode",
            "pose_estimation_method",
        )

        # Compute default inference_output_dir if needed, but don't update database
        # during fetch to avoid referential integrity issues
        if not inference_output_dir:
            inference_output_dir = InferenceTask.infer_output_dir(
                key, relative=True, mkdir=True
            )

        model_dir_rel, model_file = (Model * moseq_train.SelectedFullFit & key).fetch1(
            "model_dir", "model_file"
        )  # model dir relative to processed data directory

        model_key = (Model * moseq_train.SelectedFullFit & key).fetch1("KEY")
        fullfit_checkpoint_path = (
            moseq_train.FullFit.File & model_key & 'file_name="checkpoint.h5"'
        ).fetch1("file_path")
        fullfit_kpms_dj_config_file = (
            moseq_train.FullFit.ConfigFile & model_key
        ).fetch1("config_file")
        fullfit_pca_file_path = (
            moseq_train.PCAFit.File & model_key & 'file_name="pca.p"'
        ).fetch1("file_path")
        use_bodyparts = (moseq_train.BodyParts & model_key).fetch1("use_bodyparts")

        return (
            keypointset_dir,
            inference_output_dir,
            num_iterations,
            task_mode,
            pose_estimation_method,
            model_dir_rel,
            model_file,
            fullfit_checkpoint_path,
            fullfit_kpms_dj_config_file,
            fullfit_pca_file_path,
            use_bodyparts,
        )

    def make_compute(
        self,
        key,
        keypointset_dir,
        inference_output_dir,
        num_iterations,
        task_mode,
        pose_estimation_method,
        model_dir_rel,
        model_file,
        fullfit_checkpoint_path,
        fullfit_kpms_dj_config_file,
        fullfit_pca_file_path,
        use_bodyparts,
    ):
        """Compute model inference results.

        High-Level Logic:
        1. Set up output directories and resolve paths.
        2. If task_mode is 'trigger':
           a. Load model checkpoint and configuration.
           b. Get video and keypoint files from RecordingSet.File.
           c. Calculate average frame rate from video files.
           d. Load keypoint data and perform outlier removal.
           e. Format data for model inference.
           f. Apply trained model to new data.
           g. Save results as HDF5 and CSV files.
           h. Save coordinates and confidences to pickle files.
        3. If task_mode is 'load':
           a. Load existing coordinates and confidences from pickle files.
           b. Load average_frame_rate from existing Inference entry.
        4. Return all computed results for insertion.
        """
        import cv2
        from keypoint_moseq import (
            apply_model,
            format_data,
            load_checkpoint,
            load_keypoints,
            load_pca,
            load_results,
            outlier_removal,
            save_results_as_csv,
        )

        # Constants used by default as in kpms
        DEFAULT_NUM_ITERS = 500

        start_time = datetime.now(timezone.utc)

        original_inference_output_dir = (InferenceTask & key).fetch1(
            "inference_output_dir"
        )
        if not original_inference_output_dir:
            InferenceTask.update1({**key, "inference_output_dir": inference_output_dir})

        # Get directories for new recordings
        kpms_root = moseq_train.get_kpms_root_data_dir()
        kpms_processed = moseq_train.get_kpms_processed_data_dir()

        inference_output_dir = (
            Path(kpms_processed) / model_dir_rel / inference_output_dir
        )
        # Ensure directory exists
        inference_output_dir.mkdir(parents=True, exist_ok=True)

        keypointset_dir = find_full_path(kpms_root, keypointset_dir)

        if task_mode == "trigger":
            import jax_moseq

            # load saved model data
            fullfit_kpms_dj_config_dict = kpms_reader.load_kpms_dj_config(
                config_path=fullfit_kpms_dj_config_file
            )
            recording_set_key = (RecordingSet & (InferenceTask & key)).fetch1("KEY")
            file_paths = (RecordingSet.File & recording_set_key).fetch("file_path")

            # Filter video files and keypoint files from RecordingSet.File
            video_extensions = [".mp4", ".avi", ".mov", ".wmv", ".mpeg", ".mpg"]
            video_files = []
            h5_files = []
            csv_files = []

            for file_path in file_paths:
                file_path_full = find_full_path(kpms_root, file_path)
                file_ext = Path(file_path_full).suffix.lower()

                if file_ext in video_extensions:
                    video_files.append(file_path_full)
                elif file_ext == ".h5":
                    h5_files.append(file_path_full)
                elif file_ext == ".csv":
                    csv_files.append(file_path_full)

            # Calculate average frame rate from video files in RecordingSet.File
            frame_rates = []
            for video_file in video_files:
                cap = cv2.VideoCapture(str(video_file))
                frame_rate = cap.get(cv2.CAP_PROP_FPS)
                frame_rates.append(frame_rate)

            if not frame_rates:
                raise FileNotFoundError(
                    f"No video files found in RecordingSet.File for {recording_set_key}"
                )
            average_frame_rate = np.mean(frame_rates)

            # Load fullfit model
            fullfit_model, _, _, _ = load_checkpoint(path=fullfit_checkpoint_path)

            # Determine keypoint file extension from RecordingSet.File
            if h5_files:
                extension = ".h5"
            elif csv_files:
                extension = ".csv"
                logger.warning(
                    "No .h5 files found in RecordingSet.File, using .csv files instead"
                )
            else:
                raise FileNotFoundError(
                    f"No keypoint files (.h5 or .csv) found in RecordingSet.File for {recording_set_key}"
                )

            coordinates, confidences, formatted_bodyparts = load_keypoints(
                filepath_pattern=keypointset_dir,
                format=pose_estimation_method,
                extension=extension,
            )

            # Add a sanity check to ensure that use_bodyparts are a subset of formatted_bodyparts
            if not set(use_bodyparts).issubset(set(formatted_bodyparts)):
                raise ValueError(
                    f"Model use_bodyparts ({use_bodyparts}) is not a subset of RecordingSet formatted bodyparts ({formatted_bodyparts})"
                )

            coordinates, confidences = outlier_removal(
                coordinates=coordinates,
                confidences=confidences,
                project_dir=inference_output_dir.as_posix(),
                bodyparts=formatted_bodyparts,
                use_bodyparts=use_bodyparts,
                overwrite=False,
            )

            data, metadata = format_data(
                coordinates=coordinates,
                confidences=confidences,
                bodyparts=formatted_bodyparts,
                use_bodyparts=use_bodyparts,
            )

            data = jax_moseq.utils.debugging.convert_data_precision(data)

            # Apply saved model to new data
            results = apply_model(
                model=fullfit_model,
                data=data,
                metadata=metadata,
                project_dir=inference_output_dir.as_posix(),
                model_name=inference_output_dir.name,
                results_path=(inference_output_dir / "results.h5").as_posix(),
                return_model=False,
                num_iters=num_iterations or DEFAULT_NUM_ITERS,
                overwrite=True,
                save_results=True,
                parallel_message_passing=False,
                **fullfit_kpms_dj_config_dict,
            )

            # Create results directory and save CSV files
            save_results_as_csv(
                results=results,
                save_dir=(inference_output_dir / "results_as_csv").as_posix(),
            )

            # Save coordinates and confidences to pickle files
            coordinates_filepath = inference_output_dir / "coordinates.pkl"
            confidences_filepath = inference_output_dir / "confidences.pkl"
            with open(coordinates_filepath, "wb") as f:
                pickle.dump(coordinates, f)
            with open(confidences_filepath, "wb") as f:
                pickle.dump(confidences, f)

            end_time = datetime.now(timezone.utc)
            duration_seconds = (end_time - start_time).total_seconds()

        else:
            duration_seconds = None
            coordinates_filepath = inference_output_dir / "coordinates.pkl"
            confidences_filepath = inference_output_dir / "confidences.pkl"
            if not coordinates_filepath.exists() or not confidences_filepath.exists():
                raise FileNotFoundError(
                    f"Required files not found for load mode. "
                    f"Expected: {coordinates_filepath}, {confidences_filepath}"
                )
            with open(coordinates_filepath, "rb") as f:
                coordinates = pickle.load(f)
            with open(confidences_filepath, "rb") as f:
                confidences = pickle.load(f)

            # Load average_frame_rate from existing Inference entry
            try:
                average_frame_rate = (Inference & key).fetch1("average_frame_rate")
            except Exception as e:
                raise ValueError(
                    f"Cannot load average_frame_rate for key {key}. "
                    f"Inference entry may not exist yet. Error: {e}"
                )

        results_filepath = (inference_output_dir / "results.h5").as_posix()

        return (
            duration_seconds,
            results_filepath,
            coordinates_filepath,
            confidences_filepath,
            average_frame_rate,
        )

    def make_insert(
        self,
        key,
        duration_seconds,
        results_filepath,
        coordinates_filepath,
        confidences_filepath,
        average_frame_rate,
    ):
        """Insert inference results."""
        self.insert1(
            {
                **key,
                "syllable_segmentation_file": results_filepath,
                "coordinates_file": coordinates_filepath,
                "confidences_file": confidences_filepath,
                "average_frame_rate": average_frame_rate,
                "inference_duration": duration_seconds,
            }
        )


@schema
class MotionSequence(dj.Computed):
    """Expand inference results into per-video sequences and sampled instances.

    Attributes:
        Inference (foreign key)            : `Inference` key.
        motion_sequence_duration (float)     : Time duration (seconds) of the motion sequence computation.
    """

    definition = """
    -> Inference
    ---
    motion_sequence_duration=NULL : float
    """

    class VideoSequence(dj.Part):
        """Store the per-video sequences."""

        definition = """
        -> master
        -> RecordingSet.File                              # Foreign key to RecordingSet.File
        ---
        syllables        : longblob                       # Syllable labels (z). The syllable label assigned to each frame (i.e. the state indexes assigned by the model)
        latent_states    : longblob                       # Inferred low-dim pose state (x). Low-dimensional representation of the animal's pose in each frame. These are similar to PCA scores, are modified to reflect the pose dynamics and noise estimates inferred by the model
        centroids        : longblob                       # Inferred centroid (v). The centroid of the animal in each frame, as estimated by the model
        headings         : longblob                       # Inferred heading (h). The heading of the animal in each frame, as estimated by the model
        file             : filepath@moseq-infer-processed # File path of the temporal sequence of motion data (CSV format)
        """

    class SampledInstance(dj.Part):
        """Store the sampled instances of the grid movies."""

        definition = """
        -> master
        syllable: int
        ---
        instances: longblob
        """

    def make(self, key):
        """Expand inference results into per-video sequences and sampled instances.

        Args:
            key (dict): Primary key from the `Inference` table.

        High-Level Logic:
        1. Load inference results (syllables, latent states, centroids, headings) from HDF5 file.
        2. Filter centroids and headings using smoothing filter.
        3. Match video keys from results to RecordingSet.File entries.
        4. Create VideoSequence entries for each matched video.
        5. Sample instances for grid movies based on syllable frequency and duration.
        6. Insert results into database.
        """
        import h5py
        from keypoint_moseq import (
            filter_centroids_headings,
            get_syllable_instances,
            load_keypoints,
            load_results,
            sample_instances,
        )

        execution_time = datetime.now(timezone.utc)

        # Constants that match keypoint-moseq defaults: https://github.com/dattalab/keypoint-moseq
        FILTER_SIZE = 9  # Filter size for centroid/heading smoothing (frames)
        GRID_SAMPLES = 4 * 6  # Number of samples for grid movies (rows * cols)

        (inference_output_dir, model_dir, num_iterations, task_mode,) = (
            InferenceTask * Model & key
        ).fetch1(
            "inference_output_dir",
            "model_dir",
            "num_iterations",
            "task_mode",
        )
        kpms_processed = moseq_train.get_kpms_processed_data_dir()

        # Handle default inference_output_dir if not provided
        if not inference_output_dir:
            inference_output_dir = InferenceTask.infer_output_dir(
                key, relative=True, mkdir=True
            )
            # Update the inference_output_dir in the database
            InferenceTask.update1({**key, "inference_output_dir": inference_output_dir})

        inference_output_dir = Path(model_dir) / inference_output_dir
        inference_output_dir = find_full_path(kpms_processed, inference_output_dir)

        coordinates_file = (Inference & key).fetch1("coordinates_file")
        with open(coordinates_file, "rb") as f:
            coordinates = pickle.load(f)

        results_file = (Inference & key).fetch1("syllable_segmentation_file")
        recording_set_key = (InferenceTask & key).fetch1("KEY")
        file_ids, file_paths = (RecordingSet.File & recording_set_key).fetch(
            "file_id", "file_path"
        )

        with h5py.File(results_file, "r") as results:
            syllables = {k: np.array(v["syllable"]) for k, v in results.items()}
            latent_states = {k: np.array(v["latent_state"]) for k, v in results.items()}
            centroids = {k: np.array(v["centroid"]) for k, v in results.items()}
            headings = {k: np.array(v["heading"]) for k, v in results.items()}
            video_keys = list(results.keys())

        filtered_centroids, filtered_headings = filter_centroids_headings(
            centroids, headings, filter_size=FILTER_SIZE
        )

        # Match video keys to file IDs and paths
        matched_videos = match_video_keys_to_file_paths(
            video_keys, file_ids, file_paths, video_only=False
        )
        motion_rows = []
        for vid in video_keys:
            if vid in matched_videos:
                matched_file_id, _ = matched_videos[vid]
                motion_rows.append(
                    {
                        **key,
                        "file_id": matched_file_id,
                        "syllables": syllables[vid],
                        "latent_states": latent_states[vid],
                        "centroids": filtered_centroids[vid],
                        "headings": filtered_headings[vid],
                        "file": (
                            inference_output_dir / "results_as_csv" / f"{vid}.csv"
                        ).as_posix(),
                    }
                )
            else:
                logger.warning(f"No file found for video key: '{vid}'")

        # Log an error if no video keys were matched to file IDs and paths
        if not motion_rows:
            logger.error(
                f"Failed to match any video keys. "
                f"Video keys: {video_keys}, "
                f"File stems: {[Path(fp).stem for fp in file_paths]}, "
                f"Total files in RecordingSet.File: {len(file_paths)}. "
                f"Ensure RecordingSet.File is populated with video files for this recording."
            )

        syllable_instances = get_syllable_instances(
            syllables, min_duration=MIN_DURATION, min_frequency=MIN_FREQUENCY
        )

        # Adjust num_samples to the minimum available instances across all syllables
        if syllable_instances:
            min_instances = min(len(v) for v in syllable_instances.values())
            num_samples = min(GRID_SAMPLES, min_instances)
            if num_samples < GRID_SAMPLES:
                logger.warning(
                    f"Some syllables have fewer than {GRID_SAMPLES} instances. "
                    f"Using {num_samples} samples (minimum available across all syllables)."
                )

            sampled = sample_instances(
                syllable_instances=syllable_instances,
                num_samples=num_samples,
                coordinates=coordinates,
                centroids=filtered_centroids,
                headings=filtered_headings,
            )

            sampled_rows = [
                {**key, "syllable": s, "instances": inst} for s, inst in sampled.items()
            ]
        else:
            # No syllable instances found, skip sampling
            logger.warning(
                "No syllable instances found after filtering. Skipping sampling."
            )
            sampled_rows = []

        completion_time = datetime.now(timezone.utc)
        duration_seconds = (completion_time - execution_time).total_seconds()

        self.insert1({**key, "motion_sequence_duration": duration_seconds})

        self.VideoSequence.insert(motion_rows)

        self.SampledInstance.insert(sampled_rows)
