"""
Pytest configuration and fixtures for element-moseq tests.

Following the element-array-ephys pattern for DataJoint Element testing.
"""
import os
import pathlib

import datajoint as dj
import pytest

logger = dj.logger
_tear_down = False  # Set to True to drop schemas after tests


# ---------------------- EARLY CONFIG ----------------------
# Configure DataJoint BEFORE any module imports that activate schemas

def _setup_dj_config():
    """Set up DataJoint configuration early, before schema activation."""
    # Look for config in the tests directory first
    tests_dir = pathlib.Path(__file__).parent
    config_path = tests_dir / "dj_local_conf.json"
    if config_path.exists():
        dj.config.load(str(config_path))
    elif pathlib.Path("./dj_local_conf.json").exists():
        dj.config.load("./dj_local_conf.json")

    # Get test data directories (use user's data folder)
    test_data_dir = pathlib.Path.home() / "Documents" / "data" / "element-moseq" / "test"
    test_data_dir.mkdir(parents=True, exist_ok=True)

    dj.config.update(
        {
            "safemode": False,
            "database.host": os.environ.get("DJ_HOST") or dj.config["database.host"],
            "database.password": os.environ.get("DJ_PASS")
            or dj.config["database.password"],
            "database.user": os.environ.get("DJ_USER") or dj.config["database.user"],
            # Configure external stores for filepath@ attributes
            "stores": {
                "moseq-train-processed": {
                    "protocol": "file",
                    "location": str(test_data_dir / "moseq_train"),
                    "stage": str(test_data_dir / "moseq_train"),
                },
                "moseq-infer-processed": {
                    "protocol": "file",
                    "location": str(test_data_dir / "moseq_infer"),
                    "stage": str(test_data_dir / "moseq_infer"),
                },
            },
        }
    )
    # Use test prefix to avoid conflicts with production data
    os.environ["DATABASE_PREFIX"] = os.environ.get("DATABASE_PREFIX", "test_")
    # Enable filepath feature
    os.environ["DJ_SUPPORT_FILEPATH_MANAGEMENT"] = "TRUE"


# Run config setup immediately when conftest.py is loaded
_setup_dj_config()


# ---------------------- FIXTURES ----------------------


@pytest.fixture(autouse=True, scope="session")
def dj_config():
    """DataJoint configuration fixture (config already set up at module load)."""
    # Config is already set up by _setup_dj_config() at module load time
    # This fixture just makes it available to other fixtures
    return


@pytest.fixture(scope="session")
def pipeline(dj_config):
    """Provide pipeline modules to tests that need database access."""
    from . import tutorial_pipeline as pipeline

    yield {
        "lab": pipeline.lab,
        "subject": pipeline.subject,
        "session": pipeline.session,
        "moseq_train": pipeline.moseq_train,
        "moseq_infer": pipeline.moseq_infer,
        "moseq_report": pipeline.moseq_report,
        "get_kpms_root_data_dir": pipeline.get_kpms_root_data_dir,
        "get_kpms_processed_data_dir": pipeline.get_kpms_processed_data_dir,
        "Device": pipeline.Device,
    }

    if _tear_down:
        pipeline.moseq_report.schema.drop()
        pipeline.moseq_infer.schema.drop()
        pipeline.moseq_train.schema.drop()
        pipeline.session.schema.drop()
        pipeline.subject.schema.drop()
        pipeline.lab.schema.drop()


@pytest.fixture(scope="session")
def insert_upstreams(pipeline):
    """Insert base data for testing: Subject, Session."""
    subject = pipeline["subject"]
    session = pipeline["session"]

    subject.Subject.insert1(
        dict(subject="subj001", subject_birth_date="2023-01-01", sex="U"),
        skip_duplicates=True,
    )

    session_key = dict(subject="subj001", session_datetime="2023-01-01 00:00:00")
    session.Session.insert1(session_key, skip_duplicates=True)

    return session_key


@pytest.fixture(scope="session")
def insert_moseq_upstreams(pipeline, insert_upstreams):
    """Insert MoSeq-specific upstream data: KeypointSet, BodyParts, PCATask."""
    moseq_train = pipeline["moseq_train"]
    # session_key not used for these tables (they don't reference Session)

    # Insert PoseEstimationMethod
    moseq_train.PoseEstimationMethod.insert1(
        {
            "pose_estimation_method": "deeplabcut",
            "pose_estimation_desc": "DeepLabCut pose estimation",
        },
        skip_duplicates=True,
    )

    # Insert KeypointSet (only kpset_id is primary key)
    kpset_key = {"kpset_id": 1}
    moseq_train.KeypointSet.insert1(
        {
            **kpset_key,
            "pose_estimation_method": "deeplabcut",
            "kpset_dir": "test_data/keypoints",
            "kpset_desc": "Test keypoint set",
        },
        skip_duplicates=True,
    )

    # Insert BodyParts (kpset_id + bodyparts_id are primary key)
    bodyparts_key = {"kpset_id": 1, "bodyparts_id": 1}
    moseq_train.BodyParts.insert1(
        {
            **bodyparts_key,
            "anterior_bodyparts": ["nose", "head"],
            "posterior_bodyparts": ["tail_base"],
            "use_bodyparts": ["nose", "head", "tail_base"],
            "bodyparts_desc": "Test bodyparts",
        },
        skip_duplicates=True,
    )

    return {"kpset_id": 1, "bodyparts_id": 1}


@pytest.fixture(scope="session")
def insert_pca_task(pipeline, insert_moseq_upstreams):
    """Insert PCATask entry for testing PreProcessing."""
    moseq_train = pipeline["moseq_train"]
    upstream_key = insert_moseq_upstreams

    pca_task_key = {
        **upstream_key,
        "kpms_project_output_dir": "",  # Empty to test auto-generation
        "task_mode": "trigger",
        "outlier_scale_factor": 5.0,
    }
    moseq_train.PCATask.insert1(pca_task_key, skip_duplicates=True)

    return upstream_key


@pytest.fixture(scope="session")
def insert_prefit_task_trigger(pipeline, insert_pca_task):
    """Insert PreFitTask entry with task_mode='trigger'."""
    moseq_train = pipeline["moseq_train"]
    pca_key = insert_pca_task

    # First need PCAFit to exist
    prefit_task_key = {
        **pca_key,
        "latent_dim": 4,
        "pre_kappa": 1000,
        "pre_num_iterations": 3,
        "model_name": "",
        "task_mode": "trigger",
        "pre_fit_desc": "Test PreFit trigger mode",
    }
    moseq_train.PreFitTask.insert1(prefit_task_key, skip_duplicates=True)

    return prefit_task_key


@pytest.fixture(scope="session")
def insert_prefit_task_load(pipeline, insert_pca_task):
    """Insert PreFitTask entry with task_mode='load' for testing BUG-001 fix."""
    moseq_train = pipeline["moseq_train"]
    pca_key = insert_pca_task

    prefit_task_key = {
        **pca_key,
        "latent_dim": 4,
        "pre_kappa": 2000,
        "pre_num_iterations": 10,
        "model_name": "test_model",  # Must specify model_name for load mode
        "task_mode": "load",
        "pre_fit_desc": "Test PreFit load mode (BUG-001 fix verification)",
    }
    moseq_train.PreFitTask.insert1(prefit_task_key, skip_duplicates=True)

    return prefit_task_key


@pytest.fixture(scope="session")
def insert_inference_task(pipeline, insert_moseq_upstreams):
    """Insert InferenceTask entry for testing Inference population."""
    moseq_infer = pipeline["moseq_infer"]
    moseq_train = pipeline["moseq_train"]
    upstream_key = insert_moseq_upstreams

    # Need RecordingSet and Model first
    recording_set_key = {**upstream_key, "recording_id": 1}
    moseq_infer.RecordingSet.insert1(
        {**recording_set_key, "recording_set_desc": "Test recording set"},
        skip_duplicates=True,
    )

    # Insert Model entry (requires SelectedFullFit)
    # This is a simplified setup - in real tests you'd need the full chain

    return recording_set_key
