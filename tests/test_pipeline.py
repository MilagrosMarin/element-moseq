"""
Tests for element-moseq pipeline.

Includes tests for critical bug fixes:
- BUG-001: model undefined in PreFit load mode
- BUG-002: DB update in make_compute (should be in make_fetch)
- BUG-003: Circular fetch in make_compute
"""
import pytest


class TestTableDefinitions:
    """Test table definitions and relationships."""

    def test_moseq_train_tables_exist(self, pipeline):
        """Verify moseq_train tables are defined."""
        moseq_train = pipeline["moseq_train"]

        assert hasattr(moseq_train, "PoseEstimationMethod")
        assert hasattr(moseq_train, "KeypointSet")
        assert hasattr(moseq_train, "BodyParts")
        assert hasattr(moseq_train, "PCATask")
        assert hasattr(moseq_train, "PreProcessing")
        assert hasattr(moseq_train, "PreFit")
        assert hasattr(moseq_train, "PreFitTask")
        assert hasattr(moseq_train, "FullFit")
        assert hasattr(moseq_train, "FullFitTask")

    def test_moseq_infer_tables_exist(self, pipeline):
        """Verify moseq_infer tables are defined."""
        moseq_infer = pipeline["moseq_infer"]

        assert hasattr(moseq_infer, "Model")
        assert hasattr(moseq_infer, "RecordingSet")
        assert hasattr(moseq_infer, "InferenceTask")
        assert hasattr(moseq_infer, "Inference")
        assert hasattr(moseq_infer, "MotionSequence")

    def test_preprocessing_has_3part_make(self, pipeline):
        """Verify PreProcessing uses 3-part make pattern."""
        moseq_train = pipeline["moseq_train"]
        preprocessing = moseq_train.PreProcessing

        assert hasattr(preprocessing, "make_fetch")
        assert hasattr(preprocessing, "make_compute")
        assert hasattr(preprocessing, "make_insert")

    def test_inference_has_3part_make(self, pipeline):
        """Verify Inference uses 3-part make pattern."""
        moseq_infer = pipeline["moseq_infer"]
        inference = moseq_infer.Inference

        assert hasattr(inference, "make_fetch")
        assert hasattr(inference, "make_compute")
        assert hasattr(inference, "make_insert")


class TestBugFixes:
    """Tests verifying bug fixes are working correctly."""

    def test_bug001_prefit_load_mode_has_load_checkpoint_import(self, pipeline):
        """
        BUG-001: Verify PreFit.make() imports load_checkpoint for load mode.

        The bug was that in load mode, `model` variable was never assigned,
        causing NameError when trying to pickle it.
        """
        import inspect

        moseq_train = pipeline["moseq_train"]
        prefit = moseq_train.PreFit

        # Get the source code of the make method
        source = inspect.getsource(prefit.make)

        # Verify load_checkpoint is imported
        assert "load_checkpoint" in source, (
            "BUG-001 fix: load_checkpoint should be imported in PreFit.make()"
        )

        # Verify the load mode loads model from checkpoint
        assert 'task_mode == "load"' in source or "task_mode == 'load'" in source, (
            "BUG-001 fix: PreFit.make() should handle task_mode='load'"
        )

    def test_bug002_preprocessing_update1_in_make_fetch(self, pipeline):
        """
        BUG-002: Verify PCATask.update1() is in make_fetch, not make_compute.

        The bug was that update1() was called in make_compute() which runs
        outside the transaction, breaking the 3-part make pattern.
        """
        import inspect

        moseq_train = pipeline["moseq_train"]
        preprocessing = moseq_train.PreProcessing

        # Get source code
        make_fetch_source = inspect.getsource(preprocessing.make_fetch)
        make_compute_source = inspect.getsource(preprocessing.make_compute)

        # Verify update1 is in make_fetch (where it belongs)
        assert "update1" in make_fetch_source or "PCATask.update1" in make_fetch_source, (
            "BUG-002 fix: PCATask.update1() should be in make_fetch()"
        )

        # Verify update1 is NOT in make_compute (where it was causing issues)
        # Note: The comment mentioning the move is OK, actual update1 call should not be there
        assert "PCATask.update1(" not in make_compute_source, (
            "BUG-002 fix: PCATask.update1() should NOT be called in make_compute()"
        )

    def test_bug002_inference_update1_in_make_fetch(self, pipeline):
        """
        BUG-002: Verify InferenceTask.update1() is in make_fetch for Inference.

        Same issue as PreProcessing - update1 should be in transaction.
        """
        import inspect

        moseq_infer = pipeline["moseq_infer"]
        inference = moseq_infer.Inference

        # Get source code
        make_fetch_source = inspect.getsource(inference.make_fetch)
        make_compute_source = inspect.getsource(inference.make_compute)

        # Verify update1 is in make_fetch
        assert "InferenceTask.update1" in make_fetch_source, (
            "BUG-002 fix: InferenceTask.update1() should be in make_fetch()"
        )

        # Verify update1 is NOT in make_compute
        assert "InferenceTask.update1(" not in make_compute_source, (
            "BUG-002 fix: InferenceTask.update1() should NOT be called in make_compute()"
        )

    def test_bug003_no_circular_fetch_in_preprocessing_qa(self, pipeline):
        """
        BUG-003: Verify PreProcessingQA.make_compute doesn't re-fetch config_file.

        The bug was that kpms_dj_config_path was fetched in make_fetch() and
        passed to make_compute(), but then re-fetched again in make_compute().
        """
        import inspect

        moseq_train = pipeline["moseq_train"]
        preprocessing_qa = moseq_train.PreProcessingQA

        # Get source code of make_compute
        make_compute_source = inspect.getsource(preprocessing_qa.make_compute)

        # Verify there's no re-fetch of config_file in make_compute
        # The old buggy code was: kpms_dj_config_path = (PreProcessing.ConfigFile & key).fetch1("config_file")
        assert 'fetch1("config_file")' not in make_compute_source, (
            "BUG-003 fix: make_compute should not re-fetch config_file"
        )


class TestDataInsertion:
    """Tests for data insertion operations."""

    def test_insert_pose_estimation_method(self, pipeline):
        """Test inserting PoseEstimationMethod."""
        moseq_train = pipeline["moseq_train"]

        moseq_train.PoseEstimationMethod.insert1(
            {
                "pose_estimation_method": "test_method",
                "pose_estimation_desc": "Test method for pytest",
            },
            skip_duplicates=True,
        )

        assert len(moseq_train.PoseEstimationMethod & {"pose_estimation_method": "test_method"}) == 1

    def test_insert_upstreams(self, pipeline, insert_upstreams):
        """Test upstream data insertion."""
        subject = pipeline["subject"]
        session_key = insert_upstreams

        assert len(subject.Subject & {"subject": "subj001"}) == 1

    def test_insert_moseq_upstreams(self, pipeline, insert_moseq_upstreams):
        """Test MoSeq upstream data insertion."""
        moseq_train = pipeline["moseq_train"]
        upstream_key = insert_moseq_upstreams

        assert len(moseq_train.KeypointSet & {"kpset_id": upstream_key["kpset_id"]}) == 1
        assert len(moseq_train.BodyParts & upstream_key) == 1


class TestPCATaskAutoGeneration:
    """Tests for PCATask output directory auto-generation (related to BUG-002)."""

    def test_pca_task_infer_output_dir(self, pipeline, insert_moseq_upstreams):
        """Test PCATask.infer_output_dir generates expected directory."""
        moseq_train = pipeline["moseq_train"]
        upstream_key = insert_moseq_upstreams

        output_dir = moseq_train.PCATask.infer_output_dir(upstream_key, relative=True)

        # Should generate a directory name based on kpset_id and bodyparts_id
        assert "kpset_id" in output_dir
        assert "bodyparts_id" in output_dir


class TestInferenceTaskAutoGeneration:
    """Tests for InferenceTask output directory auto-generation (related to BUG-002)."""

    def test_inference_task_has_infer_output_dir(self, pipeline):
        """Test InferenceTask has infer_output_dir method."""
        moseq_infer = pipeline["moseq_infer"]

        assert hasattr(moseq_infer.InferenceTask, "infer_output_dir")
        assert callable(moseq_infer.InferenceTask.infer_output_dir)
