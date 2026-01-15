"""
Production validation tests for element-moseq pipeline.

These tests validate that a production deployment is correctly configured and
operational. They do NOT insert or modify data, only read and inspect.

Usage:
    # Run with production database config
    pytest test_production.py -v

    # Skip if no production config (for local development)
    pytest test_production.py -v -m "not production"

Requirements:
    - Valid database connection
    - Pipeline schemas already activated
    - No write permissions required (read-only tests)
"""
import os
import pytest


# Mark all tests in this module as production tests
pytestmark = pytest.mark.production


# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture(scope="module")
def production_check():
    """Skip production tests if running on local test database."""
    db_prefix = os.environ.get("DATABASE_PREFIX", "")
    if "test" in db_prefix.lower():
        pytest.skip("Running on test database, skipping production tests")


# =============================================================================
# SCHEMA VALIDATION TESTS
# =============================================================================


class TestSchemaActivation:
    """Test that all required schemas are activated and accessible."""

    def test_moseq_train_schema_exists(self, pipeline):
        """Verify moseq_train schema is activated."""
        moseq_train = pipeline["moseq_train"]

        # Schema should be accessible
        assert moseq_train.schema.is_activated()
        assert moseq_train.schema.database is not None

    def test_moseq_infer_schema_exists(self, pipeline):
        """Verify moseq_infer schema is activated."""
        moseq_infer = pipeline["moseq_infer"]

        assert moseq_infer.schema.is_activated()
        assert moseq_infer.schema.database is not None

    def test_required_upstream_schemas(self, pipeline):
        """Verify required upstream schemas (lab, subject, session) are available."""
        lab = pipeline["lab"]
        subject = pipeline["subject"]
        session = pipeline["session"]

        assert lab.schema.is_activated()
        assert subject.schema.is_activated()
        assert session.schema.is_activated()


class TestTableDefinitions:
    """Validate that tables have expected structure."""

    def test_keypoint_set_has_required_attributes(self, pipeline):
        """Test KeypointSet table has all required attributes."""
        moseq_train = pipeline["moseq_train"]
        heading = moseq_train.KeypointSet.heading

        required_attrs = ["kpset_id", "pose_estimation_method", "kpset_dir"]
        for attr in required_attrs:
            assert attr in heading.names, f"Missing required attribute: {attr}"

    def test_bodyparts_has_list_columns(self, pipeline):
        """Test BodyParts table has blob columns for list data."""
        moseq_train = pipeline["moseq_train"]
        heading = moseq_train.BodyParts.heading

        blob_attrs = ["anterior_bodyparts", "posterior_bodyparts", "use_bodyparts"]
        for attr in blob_attrs:
            assert attr in heading.names, f"Missing blob attribute: {attr}"

    def test_pca_task_has_task_mode(self, pipeline):
        """Test PCATask has task_mode enum attribute."""
        moseq_train = pipeline["moseq_train"]
        heading = moseq_train.PCATask.heading

        assert "task_mode" in heading.names

    def test_inference_task_has_required_attributes(self, pipeline):
        """Test InferenceTask has all required attributes."""
        moseq_infer = pipeline["moseq_infer"]
        heading = moseq_infer.InferenceTask.heading

        required_attrs = [
            "recording_id",
            "model_id",
            "pose_estimation_method",
            "task_mode",
        ]
        for attr in required_attrs:
            assert attr in heading.names, f"Missing required attribute: {attr}"


class TestPartTables:
    """Validate part table structure and relationships."""

    def test_preprocessing_video_part_table(self, pipeline):
        """Test PreProcessing.Video part table exists."""
        moseq_train = pipeline["moseq_train"]

        assert hasattr(moseq_train.PreProcessing, "Video")

    def test_preprocessing_config_file_part_table(self, pipeline):
        """Test PreProcessing.ConfigFile part table exists."""
        moseq_train = pipeline["moseq_train"]

        assert hasattr(moseq_train.PreProcessing, "ConfigFile")

    def test_pca_fit_file_part_table(self, pipeline):
        """Test PCAFit.File part table exists."""
        moseq_train = pipeline["moseq_train"]

        assert hasattr(moseq_train.PCAFit, "File")
        heading = moseq_train.PCAFit.File.heading
        assert "file_path" in heading.names

    def test_fullfit_file_part_table(self, pipeline):
        """Test FullFit.File part table exists."""
        moseq_train = pipeline["moseq_train"]

        assert hasattr(moseq_train.FullFit, "File")
        heading = moseq_train.FullFit.File.heading
        assert "file_path" in heading.names

    def test_recording_set_file_part_table(self, pipeline):
        """Test RecordingSet.File part table exists."""
        moseq_infer = pipeline["moseq_infer"]

        assert hasattr(moseq_infer.RecordingSet, "File")
        heading = moseq_infer.RecordingSet.File.heading
        assert "file_id" in heading.names
        assert "file_path" in heading.names


# =============================================================================
# DATA INTEGRITY TESTS (READ-ONLY)
# =============================================================================


class TestDataIntegrity:
    """Validate data integrity in production database."""

    def test_keypoint_set_has_entries(self, pipeline):
        """Verify KeypointSet has at least one entry."""
        moseq_train = pipeline["moseq_train"]

        count = len(moseq_train.KeypointSet())
        # In production, we expect data to exist
        # If no data, this is informational (not a failure)
        if count == 0:
            pytest.skip("No KeypointSet entries found (expected in new deployments)")

    def test_bodyparts_associated_with_keypoint_set(self, pipeline):
        """Verify BodyParts entries are associated with valid KeypointSets."""
        moseq_train = pipeline["moseq_train"]

        bodyparts = moseq_train.BodyParts.fetch(as_dict=True)
        if len(bodyparts) == 0:
            pytest.skip("No BodyParts entries found")

        # Check each BodyParts entry has valid KeypointSet
        for bp in bodyparts:
            kpset_key = {"kpset_id": bp["kpset_id"]}
            assert len(moseq_train.KeypointSet & kpset_key) == 1

    def test_preprocessing_qa_has_qa_duration(self, pipeline):
        """Verify PreProcessingQA entries have valid qa_duration values."""
        moseq_train = pipeline["moseq_train"]

        qa_entries = moseq_train.PreProcessingQA.fetch("qa_duration")
        if len(qa_entries) == 0:
            pytest.skip("No PreProcessingQA entries found")

        # QA duration should be numeric
        for duration in qa_entries:
            assert duration is not None
            assert isinstance(duration, (int, float))

    def test_model_entries_have_valid_paths(self, pipeline):
        """Verify Model entries have valid model_file paths."""
        moseq_infer = pipeline["moseq_infer"]

        models = moseq_infer.Model.fetch(as_dict=True)
        if len(models) == 0:
            pytest.skip("No Model entries found")

        for model in models:
            # model_file should be a non-empty string
            assert model["model_file"]
            assert isinstance(model["model_file"], str)


# =============================================================================
# PIPELINE CHAIN VALIDATION
# =============================================================================


class TestPipelineChain:
    """Validate the complete pipeline chain is properly configured."""

    def test_training_pipeline_chain(self, pipeline):
        """Verify training pipeline dependency chain is correct."""
        moseq_train = pipeline["moseq_train"]

        # Expected chain: PCATask -> PreProcessing -> PCAFit -> PreFitTask -> PreFit
        #                       -> FullFitTask -> FullFit -> ModelScore -> SelectedFullFit

        # Check key tables exist
        tables = [
            "PCATask",
            "PreProcessing",
            "PCAFit",
            "PreFitTask",
            "PreFit",
            "FullFitTask",
            "FullFit",
            "ModelScore",
            "SelectedFullFit",
        ]
        for table_name in tables:
            assert hasattr(moseq_train, table_name), f"Missing table: {table_name}"

    def test_inference_pipeline_chain(self, pipeline):
        """Verify inference pipeline dependency chain is correct."""
        moseq_infer = pipeline["moseq_infer"]

        # Expected chain: Model + RecordingSet -> InferenceTask -> Inference -> MotionSequence
        tables = [
            "Model",
            "RecordingSet",
            "InferenceTask",
            "Inference",
            "MotionSequence",
        ]
        for table_name in tables:
            assert hasattr(moseq_infer, table_name), f"Missing table: {table_name}"


# =============================================================================
# COMPUTED TABLE STATUS
# =============================================================================


class TestComputedTableStatus:
    """Check status of computed tables (can reveal stuck jobs)."""

    def test_preprocessing_population_status(self, pipeline):
        """Check PreProcessing population status."""
        moseq_train = pipeline["moseq_train"]

        # Count entries that need to be populated
        key_source_count = len(moseq_train.PreProcessing.key_source)
        populated_count = len(moseq_train.PreProcessing())

        # Report status (not a failure if pending entries exist)
        print(f"\nPreProcessing: {populated_count} completed, {key_source_count} in key_source")

    def test_pca_fit_population_status(self, pipeline):
        """Check PCAFit population status."""
        moseq_train = pipeline["moseq_train"]

        key_source_count = len(moseq_train.PCAFit.key_source)
        populated_count = len(moseq_train.PCAFit())

        print(f"\nPCAFit: {populated_count} completed, {key_source_count} in key_source")

    def test_inference_population_status(self, pipeline):
        """Check Inference population status."""
        moseq_infer = pipeline["moseq_infer"]

        key_source_count = len(moseq_infer.Inference.key_source)
        populated_count = len(moseq_infer.Inference())

        print(f"\nInference: {populated_count} completed, {key_source_count} in key_source")


# =============================================================================
# EXTERNAL STORAGE VALIDATION
# =============================================================================


class TestExternalStorage:
    """Validate external storage configuration."""

    def test_moseq_train_store_configured(self):
        """Verify moseq-train-processed store is configured."""
        import datajoint as dj

        stores = dj.config.get("stores", {})
        assert "moseq-train-processed" in stores, "Missing moseq-train-processed store"

        store_config = stores["moseq-train-processed"]
        assert "protocol" in store_config
        assert "location" in store_config

    def test_moseq_infer_store_configured(self):
        """Verify moseq-infer-processed store is configured."""
        import datajoint as dj

        stores = dj.config.get("stores", {})
        assert "moseq-infer-processed" in stores, "Missing moseq-infer-processed store"

        store_config = stores["moseq-infer-processed"]
        assert "protocol" in store_config
        assert "location" in store_config


# =============================================================================
# THREE-PART MAKE PATTERN VALIDATION
# =============================================================================


class TestThreePartMakePattern:
    """Validate the 3-part make pattern is implemented correctly."""

    def test_preprocessing_has_three_part_make(self, pipeline):
        """Test PreProcessing implements 3-part make pattern."""
        moseq_train = pipeline["moseq_train"]
        PreProcessing = moseq_train.PreProcessing

        assert hasattr(PreProcessing, "make_fetch"), "Missing make_fetch method"
        assert hasattr(PreProcessing, "make_compute"), "Missing make_compute method"
        assert hasattr(PreProcessing, "make_insert"), "Missing make_insert method"

    def test_pca_fit_has_make_method(self, pipeline):
        """Test PCAFit has make method (standard Computed table pattern)."""
        moseq_train = pipeline["moseq_train"]
        PCAFit = moseq_train.PCAFit

        # PCAFit uses standard make pattern, not 3-part
        assert hasattr(PCAFit, "make"), "Missing make method"

    def test_inference_has_three_part_make(self, pipeline):
        """Test Inference implements 3-part make pattern."""
        moseq_infer = pipeline["moseq_infer"]
        Inference = moseq_infer.Inference

        assert hasattr(Inference, "make_fetch"), "Missing make_fetch method"
        assert hasattr(Inference, "make_compute"), "Missing make_compute method"
        assert hasattr(Inference, "make_insert"), "Missing make_insert method"
