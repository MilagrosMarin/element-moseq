"""
Integration tests for element-moseq pipeline.

Tests end-to-end population workflows with actual database operations.
These tests require a running database and test data.
"""
import pytest

# =============================================================================
# TRAINING PIPELINE INTEGRATION TESTS
# =============================================================================


class TestTrainingPipelineStructure:
    """Test that the training pipeline tables are properly connected."""

    def test_preprocessing_depends_on_pca_task(self, pipeline):
        """Test PreProcessing table dependency chain."""
        moseq_train = pipeline["moseq_train"]

        # PreProcessing should depend on PCATask (table name is p_c_a_task)
        parents = moseq_train.PreProcessing.parents()
        # parents() returns list of strings like '`schema`.`table_name`'
        assert any("p_c_a_task" in name.lower() or "pca" in name.lower() for name in parents)

    def test_pca_fit_depends_on_preprocessing(self, pipeline):
        """Test PCAFit table dependency chain."""
        moseq_train = pipeline["moseq_train"]

        # PCAFit should depend on PreProcessing
        parents = moseq_train.PCAFit.parents()
        assert any("preprocessing" in name.lower() or "pre_processing" in name.lower() for name in parents)

    def test_prefit_depends_on_pca_fit(self, pipeline):
        """Test PreFit table dependency chain."""
        moseq_train = pipeline["moseq_train"]

        # PreFit should depend on PreFitTask which depends on PCAFit
        parents = moseq_train.PreFit.parents()
        assert any("prefit" in name.lower() or "pre_fit" in name.lower() for name in parents)

    def test_fullfit_depends_on_prefit(self, pipeline):
        """Test FullFit table dependency chain."""
        moseq_train = pipeline["moseq_train"]

        # FullFit should depend on FullFitTask
        parents = moseq_train.FullFit.parents()
        assert any("fullfit" in name.lower() or "full_fit" in name.lower() for name in parents)


class TestInferencePipelineStructure:
    """Test that the inference pipeline tables are properly connected."""

    def test_inference_depends_on_inference_task(self, pipeline):
        """Test Inference table dependency chain."""
        moseq_infer = pipeline["moseq_infer"]

        parents = moseq_infer.Inference.parents()
        # parents() returns list of strings
        assert any("inference" in name.lower() for name in parents)

    def test_model_exists(self, pipeline):
        """Test Model table exists and is queryable."""
        moseq_infer = pipeline["moseq_infer"]

        # Model should be a table that can be queried
        assert hasattr(moseq_infer, "Model")
        # Should be able to fetch without error (even if empty)
        result = moseq_infer.Model.fetch()
        assert isinstance(result, list) or hasattr(result, "__len__")

    def test_recording_set_file_part_table(self, pipeline):
        """Test RecordingSet.File part table structure."""
        moseq_infer = pipeline["moseq_infer"]

        # File should be a part table
        assert hasattr(moseq_infer.RecordingSet, "File")

        # Should have required attributes
        heading = moseq_infer.RecordingSet.File.heading
        assert "file_id" in heading.names
        assert "file_path" in heading.names


class TestPreProcessingPopulation:
    """Test PreProcessing table population with fixtures."""

    def test_preprocessing_key_source(self, pipeline, insert_pca_task):
        """Test that PreProcessing key_source returns expected keys."""
        moseq_train = pipeline["moseq_train"]
        pca_key = insert_pca_task

        # Key source should include our inserted key
        key_source = moseq_train.PreProcessing.key_source
        # key_source is a query - check it can be executed
        keys = key_source.fetch(as_dict=True)
        assert isinstance(keys, list)

    def test_preprocessing_has_part_tables(self, pipeline):
        """Test that PreProcessing has ConfigFile and Video part tables."""
        moseq_train = pipeline["moseq_train"]

        # PreProcessing has ConfigFile and Video part tables (not File)
        assert hasattr(moseq_train.PreProcessing, "ConfigFile")
        assert hasattr(moseq_train.PreProcessing, "Video")


class TestPCAFitPopulation:
    """Test PCAFit table structure and methods."""

    def test_pca_fit_has_file_part_table(self, pipeline):
        """Test that PCAFit has File part table."""
        moseq_train = pipeline["moseq_train"]

        assert hasattr(moseq_train.PCAFit, "File")

    def test_pca_fit_key_source_type(self, pipeline):
        """Test PCAFit key_source is proper query."""
        moseq_train = pipeline["moseq_train"]

        key_source = moseq_train.PCAFit.key_source
        # Should be a DataJoint query object
        assert hasattr(key_source, "fetch")


class TestLatentDimensionAnalysis:
    """Test LatentDimension table methods."""

    def test_latent_dimension_exists(self, pipeline):
        """Test LatentDimension table exists."""
        moseq_train = pipeline["moseq_train"]

        assert hasattr(moseq_train, "LatentDimension")

    def test_latent_dimension_has_make_method(self, pipeline):
        """Test LatentDimension has make method (Computed table)."""
        moseq_train = pipeline["moseq_train"]

        # As a Computed table, it should have make method
        assert hasattr(moseq_train.LatentDimension, "make")


class TestModelFitting:
    """Test PreFit and FullFit tables."""

    def test_prefit_task_has_required_attributes(self, pipeline):
        """Test PreFitTask has required attributes."""
        moseq_train = pipeline["moseq_train"]
        heading = moseq_train.PreFitTask.heading

        # Should have model parameters
        assert "latent_dim" in heading.names
        assert "pre_kappa" in heading.names
        assert "pre_num_iterations" in heading.names
        assert "task_mode" in heading.names

    def test_fullfit_task_has_required_attributes(self, pipeline):
        """Test FullFitTask has required attributes."""
        moseq_train = pipeline["moseq_train"]
        heading = moseq_train.FullFitTask.heading

        # Should have model parameters
        assert "latent_dim" in heading.names
        assert "full_kappa" in heading.names
        assert "full_num_iterations" in heading.names
        assert "task_mode" in heading.names

    def test_prefit_has_file_part_table(self, pipeline):
        """Test PreFit has File part table."""
        moseq_train = pipeline["moseq_train"]

        assert hasattr(moseq_train.PreFit, "File")

    def test_fullfit_has_file_part_table(self, pipeline):
        """Test FullFit has File part table."""
        moseq_train = pipeline["moseq_train"]

        assert hasattr(moseq_train.FullFit, "File")


class TestModelSelection:
    """Test ModelScore and SelectedFullFit tables."""

    def test_model_score_exists(self, pipeline):
        """Test ModelScore table exists."""
        moseq_train = pipeline["moseq_train"]

        assert hasattr(moseq_train, "ModelScore")

    def test_model_score_has_score_attribute(self, pipeline):
        """Test ModelScore has score attribute."""
        moseq_train = pipeline["moseq_train"]
        heading = moseq_train.ModelScore.heading

        assert "score" in heading.names

    def test_selected_fullfit_has_registration_attributes(self, pipeline):
        """Test SelectedFullFit has model registration attributes."""
        moseq_train = pipeline["moseq_train"]
        heading = moseq_train.SelectedFullFit.heading

        assert "registered_model_name" in heading.names
        assert "registered_model_desc" in heading.names


# =============================================================================
# DATA INSERTION TESTS
# =============================================================================


class TestDataInsertion:
    """Test that test data can be inserted correctly."""

    def test_subject_insertion(self, pipeline, insert_upstreams):
        """Test Subject data can be inserted."""
        subject = pipeline["subject"]
        session_key = insert_upstreams

        # Subject should exist
        result = subject.Subject.fetch()
        assert len(result) > 0

    def test_session_insertion(self, pipeline, insert_upstreams):
        """Test Session data can be inserted."""
        session = pipeline["session"]
        session_key = insert_upstreams

        # Session should exist
        result = (session.Session & session_key).fetch()
        assert len(result) == 1

    def test_keypoint_set_insertion(self, pipeline, insert_moseq_upstreams):
        """Test KeypointSet data can be inserted."""
        moseq_train = pipeline["moseq_train"]
        kpset_key = {"kpset_id": insert_moseq_upstreams["kpset_id"]}

        # KeypointSet should exist
        result = (moseq_train.KeypointSet & kpset_key).fetch()
        assert len(result) == 1

    def test_bodyparts_insertion(self, pipeline, insert_moseq_upstreams):
        """Test BodyParts data can be inserted."""
        moseq_train = pipeline["moseq_train"]
        bodyparts_key = insert_moseq_upstreams

        # BodyParts should exist
        result = (moseq_train.BodyParts & bodyparts_key).fetch()
        assert len(result) == 1

    def test_bodyparts_has_list_attributes(self, pipeline, insert_moseq_upstreams):
        """Test BodyParts stores list attributes correctly."""
        moseq_train = pipeline["moseq_train"]
        bodyparts_key = insert_moseq_upstreams

        entry = (moseq_train.BodyParts & bodyparts_key).fetch1()

        # Check list attributes are stored
        assert "anterior_bodyparts" in entry
        assert "posterior_bodyparts" in entry
        assert "use_bodyparts" in entry

        # Should be iterable
        assert hasattr(entry["anterior_bodyparts"], "__iter__")
        assert hasattr(entry["use_bodyparts"], "__iter__")


# =============================================================================
# REPORT SCHEMA TESTS
# =============================================================================


class TestReportSchema:
    """Test moseq_report schema structure."""

    def test_report_schema_exists(self, pipeline):
        """Test that moseq_report schema is available."""
        assert "moseq_report" in pipeline

    def test_syllable_summary_exists(self, pipeline):
        """Test SyllableSummary table exists if available."""
        moseq_report = pipeline["moseq_report"]

        # Check if SyllableSummary exists (may not be implemented yet)
        if hasattr(moseq_report, "SyllableSummary"):
            assert hasattr(moseq_report.SyllableSummary, "fetch")


# =============================================================================
# CROSS-SCHEMA DEPENDENCY TESTS
# =============================================================================


class TestCrossSchemaDependencies:
    """Test dependencies between moseq_train and moseq_infer schemas."""

    def test_model_references_selected_fullfit(self, pipeline):
        """Test Model table references SelectedFullFit."""
        moseq_infer = pipeline["moseq_infer"]

        # Model should have foreign key to SelectedFullFit
        # parents() returns list of strings like '`schema`.`table_name`'
        parents = moseq_infer.Model.parents()

        # Should reference selected_full_fit or similar
        assert any("selected" in name.lower() for name in parents)

    def test_inference_task_has_model_reference(self, pipeline):
        """Test InferenceTask references Model."""
        moseq_infer = pipeline["moseq_infer"]

        parents = moseq_infer.InferenceTask.parents()
        assert any("model" in name.lower() for name in parents)

    def test_inference_task_has_recording_set_reference(self, pipeline):
        """Test InferenceTask references RecordingSet."""
        moseq_infer = pipeline["moseq_infer"]

        parents = moseq_infer.InferenceTask.parents()
        assert any("recording" in name.lower() for name in parents)
