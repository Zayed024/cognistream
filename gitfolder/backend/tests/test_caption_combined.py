"""Tests for CaptionProcessor.build_caption_from_combined() and related methods."""

import pytest

from backend.db.models import Keyframe, VisualCaption
from backend.visual.caption_processor import CaptionProcessor, PromptLibrary


@pytest.fixture
def processor():
    return CaptionProcessor()


@pytest.fixture
def keyframe():
    return Keyframe(
        video_id="v1",
        segment_index=0,
        frame_number=30,
        timestamp=1.0,
        file_path="/tmp/f.jpg",
    )


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# PromptLibrary.combined_prompt()
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestCombinedPrompt:
    def test_combined_prompt_contains_all_labels(self):
        prompt = PromptLibrary.combined_prompt()
        for label in ("SCENE", "OBJECTS", "ACTIVITY", "ANOMALY"):
            assert label in prompt, f"Expected '{label}' in combined prompt"

    def test_combined_prompt_is_nonempty_string(self):
        prompt = PromptLibrary.combined_prompt()
        assert isinstance(prompt, str)
        assert len(prompt) > 20


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# build_caption_from_combined() — well-formatted input
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestBuildCaptionFromCombinedHappyPath:
    def test_well_formatted_response(self, processor, keyframe):
        raw = (
            "SCENE: A busy intersection with traffic lights and pedestrians.\n"
            "OBJECTS: car, traffic light, pedestrian, crosswalk\n"
            "ACTIVITY: Vehicles are stopped at a red light.\n"
            "ANOMALY: none"
        )
        cap = processor.build_caption_from_combined(keyframe, raw)

        assert isinstance(cap, VisualCaption)
        assert "busy intersection" in cap.scene_description
        assert cap.objects == ["car", "traffic light", "pedestrian", "crosswalk"]
        assert "stopped" in cap.activity
        assert cap.anomaly is None

    def test_objects_comma_separated_list(self, processor, keyframe):
        raw = (
            "SCENE: A parking lot.\n"
            "OBJECTS: sedan, SUV, motorcycle, lamppost, fence\n"
            "ACTIVITY: Cars are parked.\n"
            "ANOMALY: none"
        )
        cap = processor.build_caption_from_combined(keyframe, raw)
        assert len(cap.objects) == 5
        assert "sedan" in cap.objects
        assert "fence" in cap.objects


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# build_caption_from_combined() — partial / degraded input
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestBuildCaptionFromCombinedDegraded:
    def test_only_scene_present(self, processor, keyframe):
        """When only SCENE is in the response, other fields get safe defaults."""
        raw = "SCENE: A dimly lit warehouse with stacked boxes."
        cap = processor.build_caption_from_combined(keyframe, raw)

        assert "warehouse" in cap.scene_description
        # Objects, activity, anomaly should degrade gracefully
        assert cap.objects == []
        assert cap.activity == "static scene"
        assert cap.anomaly is None

    def test_no_labels_at_all(self, processor, keyframe):
        """If the VLM ignores the format, the raw text becomes the scene."""
        raw = "This is a brightly lit office with desks and computers."
        cap = processor.build_caption_from_combined(keyframe, raw)

        # Fallback: entire text becomes scene description
        assert "office" in cap.scene_description
        # Others get defaults
        assert cap.objects == []
        assert cap.activity == "static scene"
        assert cap.anomaly is None

    def test_empty_response_string(self, processor, keyframe):
        """An empty string should not crash and should produce safe defaults."""
        cap = processor.build_caption_from_combined(keyframe, "")

        assert "No scene description" in cap.scene_description
        assert cap.objects == []
        assert cap.activity == "static scene"
        assert cap.anomaly is None

    def test_whitespace_only_response(self, processor, keyframe):
        cap = processor.build_caption_from_combined(keyframe, "   \n\n  ")

        assert "No scene description" in cap.scene_description
        assert cap.objects == []
        assert cap.anomaly is None


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# build_caption_from_combined() — anomaly handling
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestBuildCaptionFromCombinedAnomaly:
    def test_anomaly_none_returns_none(self, processor, keyframe):
        raw = (
            "SCENE: A quiet street.\n"
            "OBJECTS: road, building\n"
            "ACTIVITY: Nothing happening.\n"
            "ANOMALY: none"
        )
        cap = processor.build_caption_from_combined(keyframe, raw)
        assert cap.anomaly is None

    def test_anomaly_nothing_unusual_returns_none(self, processor, keyframe):
        raw = (
            "SCENE: A park.\n"
            "OBJECTS: tree, bench\n"
            "ACTIVITY: static scene\n"
            "ANOMALY: Nothing unusual"
        )
        cap = processor.build_caption_from_combined(keyframe, raw)
        assert cap.anomaly is None

    def test_real_anomaly_text_returned(self, processor, keyframe):
        raw = (
            "SCENE: A highway overpass.\n"
            "OBJECTS: car, truck, barrier\n"
            "ACTIVITY: Traffic flowing normally.\n"
            "ANOMALY: A person is standing on the median divider near oncoming traffic."
        )
        cap = processor.build_caption_from_combined(keyframe, raw)
        assert cap.anomaly is not None
        assert "person" in cap.anomaly
        assert "median divider" in cap.anomaly


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# _parse_combined_fields() — edge cases
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestParseCombinedFields:
    def test_mixed_case_labels(self, processor):
        raw = (
            "Scene: A garden with flowers.\n"
            "Objects: rose, tulip, fence\n"
            "Activity: Wind blowing through the garden.\n"
            "Anomaly: none"
        )
        fields = processor._parse_combined_fields(raw)
        assert "scene" in fields
        assert "garden" in fields["scene"]
        assert "objects" in fields
        assert "rose" in fields["objects"]

    def test_upper_case_labels(self, processor):
        raw = (
            "SCENE: Night view of a city skyline.\n"
            "OBJECTS: skyscraper, bridge, river\n"
            "ACTIVITY: Lights twinkling.\n"
            "ANOMALY: none"
        )
        fields = processor._parse_combined_fields(raw)
        assert "scene" in fields
        assert "skyline" in fields["scene"]

    def test_extra_whitespace_around_labels(self, processor):
        raw = (
            "SCENE:   Lots of extra spaces here.  \n"
            "OBJECTS:   a , b , c   \n"
            "ACTIVITY:  Walking slowly.  \n"
            "ANOMALY:  none  "
        )
        fields = processor._parse_combined_fields(raw)
        assert "scene" in fields
        assert fields["scene"].strip() == "Lots of extra spaces here."
        assert "objects" in fields

    def test_no_recognised_labels_returns_whole_text_as_scene(self, processor):
        raw = "Just a plain sentence about a dog in a field."
        fields = processor._parse_combined_fields(raw)
        assert "scene" in fields
        assert "dog" in fields["scene"]
        # No other fields should be present
        assert "objects" not in fields
        assert "activity" not in fields
        assert "anomaly" not in fields

    def test_multiline_scene_description(self, processor):
        raw = (
            "SCENE: A large warehouse.\n"
            "It has tall shelves and dim lighting.\n"
            "OBJECTS: shelf, forklift\n"
            "ACTIVITY: Forklift moving.\n"
            "ANOMALY: none"
        )
        fields = processor._parse_combined_fields(raw)
        assert "scene" in fields
        # The scene should capture the continuation line
        assert "warehouse" in fields["scene"]
        assert "shelf" in fields.get("objects", "")

    def test_empty_string_returns_scene_as_empty(self, processor):
        fields = processor._parse_combined_fields("")
        assert "scene" in fields
        assert fields["scene"] == ""
