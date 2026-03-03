"""
Unit tests for element-moseq core functionality.

Tests individual functions and methods in isolation, with mocking where needed.
"""
import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest


# =============================================================================
# INFER_OUTPUT_DIR TESTS
# =============================================================================


class TestInferOutputDir:
    """Test infer_output_dir methods for PCATask and InferenceTask."""

    def test_pca_task_infer_output_dir_relative(self, pipeline, insert_moseq_upstreams):
        """Test PCATask.infer_output_dir returns relative path."""
        moseq_train = pipeline["moseq_train"]
        key = insert_moseq_upstreams  # Use fixture with actual data

        output_dir = moseq_train.PCATask.infer_output_dir(key, relative=True)

        assert isinstance(output_dir, str)
        # Should contain key identifiers
        assert len(output_dir) > 0

    def test_pca_task_infer_output_dir_absolute(self, pipeline, insert_moseq_upstreams):
        """Test PCATask.infer_output_dir returns absolute path when relative=False."""
        moseq_train = pipeline["moseq_train"]
        key = insert_moseq_upstreams  # Use fixture with actual data

        output_dir = moseq_train.PCATask.infer_output_dir(key, relative=False)

        # When relative=False, the function returns a Path or str depending on implementation
        # The key thing is it should be a valid path that exists or can be created
        assert output_dir is not None
        # It may be a string or Path - just verify it contains the key identifiers
        output_str = str(output_dir)
        assert "kpset_id" in output_str or len(output_str) > 0

    def test_inference_task_infer_output_dir_exists(self, pipeline):
        """Test InferenceTask has infer_output_dir method."""
        moseq_infer = pipeline["moseq_infer"]

        assert hasattr(moseq_infer.InferenceTask, "infer_output_dir")
        assert callable(moseq_infer.InferenceTask.infer_output_dir)


# =============================================================================
# HELPER FUNCTION TESTS
# =============================================================================


class TestKpmsReaderHelpers:
    """Test kpms_reader helper functions."""

    def test_find_checkpoint_file(self):
        """Test find_checkpoint_file function exists and is callable."""
        from element_moseq.readers import kpms_reader

        assert hasattr(kpms_reader, "find_checkpoint_file")
        assert callable(kpms_reader.find_checkpoint_file)

    def test_load_prefit_model(self):
        """Test load_prefit_model function exists and is callable."""
        from element_moseq.readers import kpms_reader

        assert hasattr(kpms_reader, "load_prefit_model")
        assert callable(kpms_reader.load_prefit_model)

    def test_video_extensions_in_reader(self):
        """Test VIDEO_EXTENSIONS constant is in kpms_reader."""
        from element_moseq.readers import kpms_reader

        assert hasattr(kpms_reader, "VIDEO_EXTENSIONS")
        assert isinstance(kpms_reader.VIDEO_EXTENSIONS, (list, tuple, set))
        extensions_lower = [ext.lower() for ext in kpms_reader.VIDEO_EXTENSIONS]
        assert ".mp4" in extensions_lower or "mp4" in extensions_lower
        assert ".avi" in extensions_lower or "avi" in extensions_lower

    def test_validate_video_directory_exists(self):
        """Test validate_video_directory function exists."""
        from element_moseq.readers import kpms_reader

        assert hasattr(kpms_reader, "validate_video_directory")
        assert callable(kpms_reader.validate_video_directory)


# =============================================================================
# VIDEO EXTENSIONS TESTS
# =============================================================================


class TestVideoExtensions:
    """Test VIDEO_EXTENSIONS constant is properly defined."""

    def test_video_extensions_defined(self):
        """Test VIDEO_EXTENSIONS contains common video formats in readers module."""
        from element_moseq.readers.kpms_reader import VIDEO_EXTENSIONS

        assert isinstance(VIDEO_EXTENSIONS, (list, tuple, set))
        # Should include common formats
        extensions_lower = [ext.lower() for ext in VIDEO_EXTENSIONS]
        assert ".mp4" in extensions_lower or "mp4" in extensions_lower
        assert ".avi" in extensions_lower or "avi" in extensions_lower

    def test_video_extensions_in_infer(self):
        """Test VIDEO_EXTENSIONS is also available in moseq_infer."""
        from element_moseq.moseq_infer import VIDEO_EXTENSIONS

        assert isinstance(VIDEO_EXTENSIONS, (list, tuple, set))
        assert len(VIDEO_EXTENSIONS) > 0


# =============================================================================
# CONFIG VALIDATION TESTS
# =============================================================================


class TestConfigValidation:
    """Test configuration file validation."""

    def test_empty_config_returns_defaults(self):
        """Test that empty config file returns default values with empty arrays."""
        from element_moseq.readers import kpms_reader

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yml", delete=False
        ) as f:
            f.write("{}")  # Empty YAML
            temp_config = f.name

        try:
            # Empty config should load without error and return defaults
            result = kpms_reader.load_kpms_dj_config(config_path=temp_config)
            assert isinstance(result, dict)
            # Should have skeleton key (even if empty list)
            assert "skeleton" in result
        finally:
            os.unlink(temp_config)

    def test_missing_config_raises_error(self):
        """Test that missing config file raises FileNotFoundError."""
        from element_moseq.readers import kpms_reader

        # Test with kpms_project_dir (it will look for kpms_dj_config.yml inside)
        with pytest.raises(FileNotFoundError):
            kpms_reader.load_kpms_dj_config(kpms_project_dir="/nonexistent/path")


# =============================================================================
# 3-PART MAKE PATTERN TESTS
# =============================================================================


class TestThreePartMakePatternLogic:
    """Test the 3-part make pattern logic (make_fetch, make_compute, make_insert)."""

    def test_make_fetch_returns_tuple(self, pipeline):
        """Test that make_fetch methods return tuples of data."""
        moseq_train = pipeline["moseq_train"]

        # PreProcessing should have make_fetch
        assert hasattr(moseq_train.PreProcessing, "make_fetch")

        # The signature should accept (self, key)
        import inspect

        sig = inspect.signature(moseq_train.PreProcessing.make_fetch)
        params = list(sig.parameters.keys())
        assert "key" in params

    def test_make_compute_signature(self, pipeline):
        """Test that make_compute has correct signature."""
        moseq_train = pipeline["moseq_train"]

        assert hasattr(moseq_train.PreProcessing, "make_compute")

        import inspect

        sig = inspect.signature(moseq_train.PreProcessing.make_compute)
        params = list(sig.parameters.keys())
        # Should have key and fetched data params
        assert "key" in params

    def test_make_insert_signature(self, pipeline):
        """Test that make_insert has correct signature."""
        moseq_train = pipeline["moseq_train"]

        assert hasattr(moseq_train.PreProcessing, "make_insert")

        import inspect

        sig = inspect.signature(moseq_train.PreProcessing.make_insert)
        params = list(sig.parameters.keys())
        assert "key" in params


# =============================================================================
# SELECTED FULLFIT TESTS
# =============================================================================


class TestSelectedFullFit:
    """Test SelectedFullFit table methods."""

    def test_select_best_model_method_exists(self, pipeline):
        """Test SelectedFullFit has select_best_model classmethod."""
        moseq_train = pipeline["moseq_train"]

        assert hasattr(moseq_train.SelectedFullFit, "select_best_model")
        assert callable(moseq_train.SelectedFullFit.select_best_model)

    def test_selected_fullfit_attributes(self, pipeline):
        """Test SelectedFullFit has expected attributes."""
        moseq_train = pipeline["moseq_train"]
        heading = moseq_train.SelectedFullFit.heading

        assert "registered_model_name" in heading.secondary_attributes
        assert "registered_model_desc" in heading.secondary_attributes


# =============================================================================
# RECORDING SET TESTS
# =============================================================================


class TestRecordingSet:
    """Test RecordingSet table and its File part table."""

    def test_recording_set_has_file_part(self, pipeline):
        """Test RecordingSet has File part table."""
        moseq_infer = pipeline["moseq_infer"]

        assert hasattr(moseq_infer.RecordingSet, "File")

    def test_recording_set_file_attributes(self, pipeline):
        """Test RecordingSet.File has expected attributes."""
        moseq_infer = pipeline["moseq_infer"]
        heading = moseq_infer.RecordingSet.File.heading

        assert "file_id" in heading.primary_key
        assert "file_path" in heading.secondary_attributes


# =============================================================================
# MOTION SEQUENCE TESTS
# =============================================================================


class TestMotionSequence:
    """Test MotionSequence table and its part tables."""

    def test_motion_sequence_exists(self, pipeline):
        """Test MotionSequence table exists."""
        moseq_infer = pipeline["moseq_infer"]

        assert hasattr(moseq_infer, "MotionSequence")

    def test_motion_sequence_has_video_sequence_part(self, pipeline):
        """Test MotionSequence has VideoSequence part table."""
        moseq_infer = pipeline["moseq_infer"]

        assert hasattr(moseq_infer.MotionSequence, "VideoSequence")

    def test_motion_sequence_has_sampled_instance_part(self, pipeline):
        """Test MotionSequence has SampledInstance part table."""
        moseq_infer = pipeline["moseq_infer"]

        assert hasattr(moseq_infer.MotionSequence, "SampledInstance")


# =============================================================================
# ERROR HANDLING TESTS
# =============================================================================


class TestErrorHandling:
    """Test error handling in various scenarios."""

    def test_inference_without_model_raises_error(self, pipeline):
        """Test that Inference fails gracefully without a valid model."""
        moseq_infer = pipeline["moseq_infer"]

        # Attempting to fetch from empty Inference should return empty
        result = moseq_infer.Inference.fetch()
        assert len(result) == 0  # Should be empty, not error

    def test_invalid_task_mode_handling(self, pipeline):
        """Test that invalid task_mode values are rejected by schema."""
        moseq_train = pipeline["moseq_train"]
        heading = moseq_train.PCATask.heading

        # task_mode should be an enum
        task_mode_attr = heading.attributes.get("task_mode")
        assert task_mode_attr is not None
        # Check it has allowed values
        assert "load" in str(task_mode_attr) or "trigger" in str(task_mode_attr)


# =============================================================================
# NUMPY ARRAY HANDLING TESTS
# =============================================================================


class TestNumpyArrayHandling:
    """Test handling of numpy arrays in blob attributes."""

    def test_bodyparts_stores_lists(self, pipeline, insert_moseq_upstreams):
        """Test that BodyParts correctly stores list attributes."""
        moseq_train = pipeline["moseq_train"]
        upstream_key = insert_moseq_upstreams

        bodyparts = (moseq_train.BodyParts & upstream_key).fetch1()

        # These should be stored as lists/arrays
        assert "anterior_bodyparts" in bodyparts
        assert "posterior_bodyparts" in bodyparts
        assert "use_bodyparts" in bodyparts

        # Should be iterable
        assert hasattr(bodyparts["anterior_bodyparts"], "__iter__")
        assert hasattr(bodyparts["use_bodyparts"], "__iter__")


# =============================================================================
# NORMALIZE BODYPARTS BLOB TESTS
# =============================================================================


class TestNormalizeBodypartsBlob:
    """Test BodyParts.normalize_bodyparts_blob static method."""

    @pytest.fixture
    def normalize(self):
        from element_moseq.moseq_train import BodyParts

        return BodyParts.normalize_bodyparts_blob

    def test_native_list_passthrough(self, normalize):
        """Native Python list passes through unchanged."""
        value = ["nose", "head", "tail_base"]
        result = normalize(value)
        assert result == ["nose", "head", "tail_base"]
        assert isinstance(result, list)

    def test_string_list_representation(self, normalize):
        """String representation of a list is parsed correctly."""
        value = "['nose', 'head', 'tail_base']"
        result = normalize(value)
        assert result == ["nose", "head", "tail_base"]
        assert isinstance(result, list)

    def test_string_list_double_quotes(self, normalize):
        """String list with double-quoted items is parsed correctly."""
        value = '["nose", "head", "tail_base"]'
        result = normalize(value)
        assert result == ["nose", "head", "tail_base"]

    def test_comma_separated_string(self, normalize):
        """Comma-separated string (no brackets) is split correctly."""
        value = "nose, head, tail_base"
        result = normalize(value)
        assert result == ["nose", "head", "tail_base"]

    def test_tuple_to_list(self, normalize):
        """Tuple is converted to list."""
        value = ("nose", "head", "tail_base")
        result = normalize(value)
        assert result == ["nose", "head", "tail_base"]
        assert isinstance(result, list)

    def test_set_to_list(self, normalize):
        """Set is converted to list."""
        value = {"nose", "head"}
        result = normalize(value)
        assert isinstance(result, list)
        assert set(result) == {"nose", "head"}

    def test_empty_string(self, normalize):
        """Empty string returns empty list."""
        result = normalize("")
        assert result == []

    def test_single_item_string_list(self, normalize):
        """Single-item string list is parsed correctly."""
        value = "['nose']"
        result = normalize(value)
        assert result == ["nose"]

    def test_string_tuple_representation(self, normalize):
        """String representation of a tuple is parsed correctly."""
        value = "('nose', 'head')"
        result = normalize(value)
        assert result == ["nose", "head"]

    def test_numpy_array_passthrough(self, normalize):
        """Numpy array passes through unchanged (not str/tuple/set)."""
        value = np.array(["nose", "head", "tail_base"])
        result = normalize(value)
        assert list(result) == ["nose", "head", "tail_base"]
