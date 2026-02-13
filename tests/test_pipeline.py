"""
Tests for element-moseq pipeline.

Comprehensive tests covering all tables in moseq_train, moseq_infer, and moseq_report.
"""
import datajoint as dj
import pytest


# =============================================================================
# MOSEQ_TRAIN TABLE TESTS
# =============================================================================


class TestMoseqTrainTables:
    """Test all moseq_train table definitions."""

    def test_pose_estimation_method_exists(self, pipeline):
        """Verify PoseEstimationMethod table exists and is Lookup."""
        moseq_train = pipeline["moseq_train"]
        assert hasattr(moseq_train, "PoseEstimationMethod")
        assert issubclass(moseq_train.PoseEstimationMethod, dj.Lookup)

    def test_pose_estimation_method_attributes(self, pipeline):
        """Verify PoseEstimationMethod has expected attributes."""
        moseq_train = pipeline["moseq_train"]
        heading = moseq_train.PoseEstimationMethod.heading
        assert "pose_estimation_method" in heading.primary_key
        assert "pose_estimation_desc" in heading.secondary_attributes

    def test_keypoint_set_exists(self, pipeline):
        """Verify KeypointSet table exists and is Manual."""
        moseq_train = pipeline["moseq_train"]
        assert hasattr(moseq_train, "KeypointSet")
        assert issubclass(moseq_train.KeypointSet, dj.Manual)

    def test_keypoint_set_has_video_file_part(self, pipeline):
        """Verify KeypointSet.VideoFile part table exists."""
        moseq_train = pipeline["moseq_train"]
        assert hasattr(moseq_train.KeypointSet, "VideoFile")
        assert issubclass(moseq_train.KeypointSet.VideoFile, dj.Part)

    def test_keypoint_set_attributes(self, pipeline):
        """Verify KeypointSet has expected attributes."""
        moseq_train = pipeline["moseq_train"]
        heading = moseq_train.KeypointSet.heading
        assert "kpset_id" in heading.primary_key
        assert "kpset_dir" in heading.secondary_attributes

    def test_body_parts_exists(self, pipeline):
        """Verify BodyParts table exists and is Manual."""
        moseq_train = pipeline["moseq_train"]
        assert hasattr(moseq_train, "BodyParts")
        assert issubclass(moseq_train.BodyParts, dj.Manual)

    def test_body_parts_attributes(self, pipeline):
        """Verify BodyParts has expected attributes."""
        moseq_train = pipeline["moseq_train"]
        heading = moseq_train.BodyParts.heading
        assert "kpset_id" in heading.primary_key
        assert "bodyparts_id" in heading.primary_key
        assert "anterior_bodyparts" in heading.secondary_attributes
        assert "posterior_bodyparts" in heading.secondary_attributes
        assert "use_bodyparts" in heading.secondary_attributes

    def test_pca_task_exists(self, pipeline):
        """Verify PCATask table exists and is Manual."""
        moseq_train = pipeline["moseq_train"]
        assert hasattr(moseq_train, "PCATask")
        assert issubclass(moseq_train.PCATask, dj.Manual)

    def test_pca_task_has_infer_output_dir(self, pipeline):
        """Verify PCATask has infer_output_dir classmethod."""
        moseq_train = pipeline["moseq_train"]
        assert hasattr(moseq_train.PCATask, "infer_output_dir")
        assert callable(moseq_train.PCATask.infer_output_dir)

    def test_pca_task_attributes(self, pipeline):
        """Verify PCATask has expected attributes."""
        moseq_train = pipeline["moseq_train"]
        heading = moseq_train.PCATask.heading
        assert "kpset_id" in heading.primary_key
        assert "bodyparts_id" in heading.primary_key
        assert "task_mode" in heading.secondary_attributes

    def test_preprocessing_exists(self, pipeline):
        """Verify PreProcessing table exists and is Computed."""
        moseq_train = pipeline["moseq_train"]
        assert hasattr(moseq_train, "PreProcessing")
        assert issubclass(moseq_train.PreProcessing, dj.Computed)

    def test_preprocessing_has_config_file_part(self, pipeline):
        """Verify PreProcessing.ConfigFile part table exists."""
        moseq_train = pipeline["moseq_train"]
        assert hasattr(moseq_train.PreProcessing, "ConfigFile")
        assert issubclass(moseq_train.PreProcessing.ConfigFile, dj.Part)

    def test_preprocessing_has_3part_make(self, pipeline):
        """Verify PreProcessing uses 3-part make pattern."""
        moseq_train = pipeline["moseq_train"]
        preprocessing = moseq_train.PreProcessing
        assert hasattr(preprocessing, "make_fetch")
        assert hasattr(preprocessing, "make_compute")
        assert hasattr(preprocessing, "make_insert")
        assert callable(preprocessing.make_fetch)
        assert callable(preprocessing.make_compute)
        assert callable(preprocessing.make_insert)

    def test_preprocessing_qa_exists(self, pipeline):
        """Verify PreProcessingQA table exists and is Computed."""
        moseq_train = pipeline["moseq_train"]
        assert hasattr(moseq_train, "PreProcessingQA")
        assert issubclass(moseq_train.PreProcessingQA, dj.Computed)

    def test_preprocessing_qa_has_3part_make(self, pipeline):
        """Verify PreProcessingQA uses 3-part make pattern."""
        moseq_train = pipeline["moseq_train"]
        qa = moseq_train.PreProcessingQA
        assert hasattr(qa, "make_fetch")
        assert hasattr(qa, "make_compute")
        assert hasattr(qa, "make_insert")

    def test_pca_fit_exists(self, pipeline):
        """Verify PCAFit table exists and is Computed."""
        moseq_train = pipeline["moseq_train"]
        assert hasattr(moseq_train, "PCAFit")
        assert issubclass(moseq_train.PCAFit, dj.Computed)

    def test_pca_fit_is_computed(self, pipeline):
        """Verify PCAFit uses standard Computed make pattern (not 3-part)."""
        moseq_train = pipeline["moseq_train"]
        pca_fit = moseq_train.PCAFit
        # PCAFit uses standard make() pattern, not 3-part make
        assert hasattr(pca_fit, "make")
        assert callable(pca_fit.make)

    def test_latent_dimension_exists(self, pipeline):
        """Verify LatentDimension table exists and is Computed."""
        moseq_train = pipeline["moseq_train"]
        assert hasattr(moseq_train, "LatentDimension")
        assert issubclass(moseq_train.LatentDimension, dj.Computed)

    def test_latent_dimension_is_computed(self, pipeline):
        """Verify LatentDimension uses standard Computed make pattern (not 3-part)."""
        moseq_train = pipeline["moseq_train"]
        ld = moseq_train.LatentDimension
        # LatentDimension uses standard make() pattern, not 3-part make
        assert hasattr(ld, "make")
        assert callable(ld.make)

    def test_prefit_task_exists(self, pipeline):
        """Verify PreFitTask table exists and is Manual."""
        moseq_train = pipeline["moseq_train"]
        assert hasattr(moseq_train, "PreFitTask")
        assert issubclass(moseq_train.PreFitTask, dj.Manual)

    def test_prefit_task_attributes(self, pipeline):
        """Verify PreFitTask has expected attributes."""
        moseq_train = pipeline["moseq_train"]
        heading = moseq_train.PreFitTask.heading
        assert "latent_dim" in heading.primary_key
        assert "pre_kappa" in heading.primary_key
        assert "task_mode" in heading.secondary_attributes

    def test_prefit_exists(self, pipeline):
        """Verify PreFit table exists and is Computed."""
        moseq_train = pipeline["moseq_train"]
        assert hasattr(moseq_train, "PreFit")
        assert issubclass(moseq_train.PreFit, dj.Computed)

    def test_prefit_has_config_file_part(self, pipeline):
        """Verify PreFit.ConfigFile part table exists."""
        moseq_train = pipeline["moseq_train"]
        assert hasattr(moseq_train.PreFit, "ConfigFile")
        assert issubclass(moseq_train.PreFit.ConfigFile, dj.Part)

    def test_prefit_qa_exists(self, pipeline):
        """Verify PreFitQA table exists and is Computed."""
        moseq_train = pipeline["moseq_train"]
        assert hasattr(moseq_train, "PreFitQA")
        assert issubclass(moseq_train.PreFitQA, dj.Computed)

    def test_prefit_qa_is_computed(self, pipeline):
        """Verify PreFitQA uses standard Computed make pattern (not 3-part)."""
        moseq_train = pipeline["moseq_train"]
        qa = moseq_train.PreFitQA
        # PreFitQA uses standard make() pattern, not 3-part make
        assert hasattr(qa, "make")
        assert callable(qa.make)

    def test_fullfit_task_exists(self, pipeline):
        """Verify FullFitTask table exists and is Manual."""
        moseq_train = pipeline["moseq_train"]
        assert hasattr(moseq_train, "FullFitTask")
        assert issubclass(moseq_train.FullFitTask, dj.Manual)

    def test_fullfit_task_attributes(self, pipeline):
        """Verify FullFitTask has expected attributes."""
        moseq_train = pipeline["moseq_train"]
        heading = moseq_train.FullFitTask.heading
        assert "latent_dim" in heading.primary_key
        assert "full_kappa" in heading.primary_key
        assert "task_mode" in heading.secondary_attributes

    def test_fullfit_exists(self, pipeline):
        """Verify FullFit table exists and is Computed."""
        moseq_train = pipeline["moseq_train"]
        assert hasattr(moseq_train, "FullFit")
        assert issubclass(moseq_train.FullFit, dj.Computed)

    def test_fullfit_has_config_file_part(self, pipeline):
        """Verify FullFit.ConfigFile part table exists."""
        moseq_train = pipeline["moseq_train"]
        assert hasattr(moseq_train.FullFit, "ConfigFile")
        assert issubclass(moseq_train.FullFit.ConfigFile, dj.Part)

    def test_fullfit_has_3part_make(self, pipeline):
        """Verify FullFit uses 3-part make pattern."""
        moseq_train = pipeline["moseq_train"]
        fullfit = moseq_train.FullFit
        assert hasattr(fullfit, "make_fetch")
        assert hasattr(fullfit, "make_compute")
        assert hasattr(fullfit, "make_insert")

    def test_fullfit_qa_exists(self, pipeline):
        """Verify FullFitQA table exists and is Computed."""
        moseq_train = pipeline["moseq_train"]
        assert hasattr(moseq_train, "FullFitQA")
        assert issubclass(moseq_train.FullFitQA, dj.Computed)

    def test_fullfit_qa_is_computed(self, pipeline):
        """Verify FullFitQA uses standard Computed make pattern (not 3-part)."""
        moseq_train = pipeline["moseq_train"]
        qa = moseq_train.FullFitQA
        # FullFitQA uses standard make() pattern, not 3-part make
        assert hasattr(qa, "make")
        assert callable(qa.make)

    def test_model_score_exists(self, pipeline):
        """Verify ModelScore table exists and is Computed."""
        moseq_train = pipeline["moseq_train"]
        assert hasattr(moseq_train, "ModelScore")
        assert issubclass(moseq_train.ModelScore, dj.Computed)

    def test_selected_fullfit_exists(self, pipeline):
        """Verify SelectedFullFit table exists and is Manual."""
        moseq_train = pipeline["moseq_train"]
        assert hasattr(moseq_train, "SelectedFullFit")
        assert issubclass(moseq_train.SelectedFullFit, dj.Manual)


# =============================================================================
# MOSEQ_INFER TABLE TESTS
# =============================================================================


class TestMoseqInferTables:
    """Test all moseq_infer table definitions."""

    def test_model_exists(self, pipeline):
        """Verify Model table exists and is Manual."""
        moseq_infer = pipeline["moseq_infer"]
        assert hasattr(moseq_infer, "Model")
        assert issubclass(moseq_infer.Model, dj.Manual)

    def test_model_attributes(self, pipeline):
        """Verify Model has expected attributes."""
        moseq_infer = pipeline["moseq_infer"]
        heading = moseq_infer.Model.heading
        assert "model_id" in heading.primary_key

    def test_recording_set_exists(self, pipeline):
        """Verify RecordingSet table exists and is Manual."""
        moseq_infer = pipeline["moseq_infer"]
        assert hasattr(moseq_infer, "RecordingSet")
        assert issubclass(moseq_infer.RecordingSet, dj.Manual)

    def test_recording_set_structure(self, pipeline):
        """Verify RecordingSet is properly structured."""
        moseq_infer = pipeline["moseq_infer"]
        # RecordingSet is a simple Manual table without part tables
        assert issubclass(moseq_infer.RecordingSet, dj.Manual)

    def test_recording_set_attributes(self, pipeline):
        """Verify RecordingSet has expected attributes."""
        moseq_infer = pipeline["moseq_infer"]
        heading = moseq_infer.RecordingSet.heading
        assert "recording_id" in heading.primary_key

    def test_inference_task_exists(self, pipeline):
        """Verify InferenceTask table exists and is Manual."""
        moseq_infer = pipeline["moseq_infer"]
        assert hasattr(moseq_infer, "InferenceTask")
        assert issubclass(moseq_infer.InferenceTask, dj.Manual)

    def test_inference_task_has_infer_output_dir(self, pipeline):
        """Verify InferenceTask has infer_output_dir classmethod."""
        moseq_infer = pipeline["moseq_infer"]
        assert hasattr(moseq_infer.InferenceTask, "infer_output_dir")
        assert callable(moseq_infer.InferenceTask.infer_output_dir)

    def test_inference_exists(self, pipeline):
        """Verify Inference table exists and is Computed."""
        moseq_infer = pipeline["moseq_infer"]
        assert hasattr(moseq_infer, "Inference")
        assert issubclass(moseq_infer.Inference, dj.Computed)

    def test_inference_has_3part_make(self, pipeline):
        """Verify Inference uses 3-part make pattern."""
        moseq_infer = pipeline["moseq_infer"]
        inference = moseq_infer.Inference
        assert hasattr(inference, "make_fetch")
        assert hasattr(inference, "make_compute")
        assert hasattr(inference, "make_insert")
        assert callable(inference.make_fetch)
        assert callable(inference.make_compute)
        assert callable(inference.make_insert)

    def test_motion_sequence_exists(self, pipeline):
        """Verify MotionSequence table exists and is Computed."""
        moseq_infer = pipeline["moseq_infer"]
        assert hasattr(moseq_infer, "MotionSequence")
        assert issubclass(moseq_infer.MotionSequence, dj.Computed)


# =============================================================================
# MOSEQ_REPORT TABLE TESTS
# =============================================================================


class TestMoseqReportTables:
    """Test all moseq_report table definitions."""

    def test_behavioral_summary_exists(self, pipeline):
        """Verify BehavioralSummary table exists and is Computed."""
        moseq_report = pipeline["moseq_report"]
        assert hasattr(moseq_report, "BehavioralSummary")
        assert issubclass(moseq_report.BehavioralSummary, dj.Computed)

    def test_trajectory_plot_exists(self, pipeline):
        """Verify TrajectoryPlot table exists and is Computed."""
        moseq_report = pipeline["moseq_report"]
        assert hasattr(moseq_report, "TrajectoryPlot")
        assert issubclass(moseq_report.TrajectoryPlot, dj.Computed)


# =============================================================================
# 3-PART MAKE PATTERN TESTS
# =============================================================================


class TestThreePartMakePattern:
    """Test that Computed tables with 3-part make follow the pattern correctly.

    Note: Only specific tables use 3-part make pattern:
    - moseq_train: PreProcessing, PreProcessingQA, FullFit
    - moseq_infer: Inference
    - moseq_report: TrajectoryPlot

    Other Computed tables (PCAFit, LatentDimension, PreFit, PreFitQA,
    FullFitQA, ModelScore, MotionSequence) use standard make() pattern.
    """

    @pytest.mark.parametrize(
        "table_name",
        [
            "PreProcessing",
            "PreProcessingQA",
            "FullFit",
        ],
    )
    def test_moseq_train_3part_make_methods_exist(self, pipeline, table_name):
        """Verify moseq_train computed tables have all 3-part make methods."""
        moseq_train = pipeline["moseq_train"]
        table = getattr(moseq_train, table_name)

        assert hasattr(table, "make_fetch"), f"{table_name} missing make_fetch"
        assert hasattr(table, "make_compute"), f"{table_name} missing make_compute"
        assert hasattr(table, "make_insert"), f"{table_name} missing make_insert"

    def test_inference_3part_make_methods_exist(self, pipeline):
        """Verify Inference has all 3-part make methods."""
        moseq_infer = pipeline["moseq_infer"]
        inference = moseq_infer.Inference

        assert hasattr(inference, "make_fetch")
        assert hasattr(inference, "make_compute")
        assert hasattr(inference, "make_insert")

    def test_trajectory_plot_3part_make_methods_exist(self, pipeline):
        """Verify TrajectoryPlot has all 3-part make methods."""
        moseq_report = pipeline["moseq_report"]
        trajectory_plot = moseq_report.TrajectoryPlot

        assert hasattr(trajectory_plot, "make_fetch")
        assert hasattr(trajectory_plot, "make_compute")
        assert hasattr(trajectory_plot, "make_insert")


# =============================================================================
# DATA INSERTION TESTS
# =============================================================================


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

        assert (
            len(
                moseq_train.PoseEstimationMethod
                & {"pose_estimation_method": "test_method"}
            )
            == 1
        )

    def test_insert_upstreams(self, pipeline, insert_upstreams):
        """Test upstream data insertion."""
        subject = pipeline["subject"]

        assert len(subject.Subject & {"subject": "subj001"}) == 1

    def test_insert_keypoint_set(self, pipeline, insert_moseq_upstreams):
        """Test KeypointSet insertion."""
        moseq_train = pipeline["moseq_train"]
        upstream_key = insert_moseq_upstreams

        assert (
            len(moseq_train.KeypointSet & {"kpset_id": upstream_key["kpset_id"]}) == 1
        )

    def test_insert_body_parts(self, pipeline, insert_moseq_upstreams):
        """Test BodyParts insertion."""
        moseq_train = pipeline["moseq_train"]
        upstream_key = insert_moseq_upstreams

        assert len(moseq_train.BodyParts & upstream_key) == 1


# =============================================================================
# AUTO-GENERATION METHOD TESTS
# =============================================================================


class TestAutoGenerationMethods:
    """Tests for auto-generation methods (infer_output_dir)."""

    def test_pca_task_infer_output_dir(self, pipeline, insert_moseq_upstreams):
        """Test PCATask.infer_output_dir generates expected directory."""
        moseq_train = pipeline["moseq_train"]
        upstream_key = insert_moseq_upstreams

        output_dir = moseq_train.PCATask.infer_output_dir(upstream_key, relative=True)

        assert "kpset_id" in output_dir
        assert "bodyparts_id" in output_dir

    def test_inference_task_infer_output_dir_exists(self, pipeline):
        """Test InferenceTask has infer_output_dir method."""
        moseq_infer = pipeline["moseq_infer"]

        assert hasattr(moseq_infer.InferenceTask, "infer_output_dir")
        assert callable(moseq_infer.InferenceTask.infer_output_dir)


# =============================================================================
# TABLE RELATIONSHIP TESTS
# =============================================================================


class TestTableRelationships:
    """Test table relationships and dependencies."""

    def test_bodyparts_depends_on_keypointset(self, pipeline):
        """Verify BodyParts references KeypointSet."""
        moseq_train = pipeline["moseq_train"]
        heading = moseq_train.BodyParts.heading

        assert "kpset_id" in heading.primary_key

    def test_pcatask_depends_on_bodyparts(self, pipeline):
        """Verify PCATask references BodyParts."""
        moseq_train = pipeline["moseq_train"]
        heading = moseq_train.PCATask.heading

        assert "kpset_id" in heading.primary_key
        assert "bodyparts_id" in heading.primary_key

    def test_preprocessing_depends_on_pcatask(self, pipeline):
        """Verify PreProcessing references PCATask."""
        moseq_train = pipeline["moseq_train"]
        heading = moseq_train.PreProcessing.heading

        assert "kpset_id" in heading.primary_key
        assert "bodyparts_id" in heading.primary_key

    def test_inference_depends_on_inferencetask(self, pipeline):
        """Verify Inference references InferenceTask."""
        moseq_infer = pipeline["moseq_infer"]
        heading = moseq_infer.Inference.heading

        assert "model_id" in heading.primary_key
        assert "recording_id" in heading.primary_key

    def test_motionsequence_depends_on_inference(self, pipeline):
        """Verify MotionSequence references Inference."""
        moseq_infer = pipeline["moseq_infer"]
        heading = moseq_infer.MotionSequence.heading

        assert "model_id" in heading.primary_key
        assert "recording_id" in heading.primary_key
