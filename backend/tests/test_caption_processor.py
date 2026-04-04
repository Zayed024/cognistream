"""Tests for backend.visual.caption_processor — VLM output parsing."""

import pytest

from backend.db.models import Keyframe, VisualCaption
from backend.visual.caption_processor import CaptionProcessor, PromptLibrary


@pytest.fixture
def processor():
    return CaptionProcessor()


@pytest.fixture
def keyframe():
    return Keyframe(video_id="v1", segment_index=0, frame_number=30, timestamp=1.0, file_path="/tmp/f.jpg")


class TestPromptLibrary:
    def test_scene_prompt_non_empty(self):
        assert len(PromptLibrary.scene_prompt()) > 10

    def test_objects_prompt_non_empty(self):
        assert len(PromptLibrary.objects_prompt()) > 10

    def test_activity_prompt_non_empty(self):
        assert len(PromptLibrary.activity_prompt()) > 10

    def test_anomaly_prompt_non_empty(self):
        assert len(PromptLibrary.anomaly_prompt()) > 10


class TestCaptionProcessorScene:
    def test_normal_scene(self, processor, keyframe):
        cap = processor.build_caption(keyframe, "A busy street.", "car, bus", "driving", "none")
        assert cap.scene_description == "A busy street."

    def test_empty_scene_fallback(self, processor, keyframe):
        cap = processor.build_caption(keyframe, "", "car", "driving", "none")
        assert "No scene description" in cap.scene_description

    def test_quoted_scene_stripped(self, processor, keyframe):
        cap = processor.build_caption(keyframe, '"A busy street."', "car", "driving", "none")
        assert cap.scene_description == "A busy street."


class TestCaptionProcessorObjects:
    def test_comma_separated(self, processor, keyframe):
        cap = processor.build_caption(keyframe, "scene", "car, tree, person", "driving", "none")
        assert cap.objects == ["car", "tree", "person"]

    def test_json_array(self, processor, keyframe):
        cap = processor.build_caption(keyframe, "scene", '["car", "tree"]', "driving", "none")
        assert cap.objects == ["car", "tree"]

    def test_bullet_list(self, processor, keyframe):
        raw = "- car\n- tree\n- person"
        cap = processor.build_caption(keyframe, "scene", raw, "driving", "none")
        assert len(cap.objects) == 3
        assert "car" in cap.objects

    def test_single_item(self, processor, keyframe):
        cap = processor.build_caption(keyframe, "scene", "car", "driving", "none")
        assert cap.objects == ["car"]

    def test_empty_returns_empty(self, processor, keyframe):
        cap = processor.build_caption(keyframe, "scene", "", "driving", "none")
        assert cap.objects == []


class TestCaptionProcessorActivity:
    def test_normal(self, processor, keyframe):
        cap = processor.build_caption(keyframe, "scene", "car", "Cars driving fast", "none")
        assert cap.activity == "Cars driving fast"

    def test_empty_returns_static(self, processor, keyframe):
        cap = processor.build_caption(keyframe, "scene", "car", "", "none")
        assert cap.activity == "static scene"


class TestCaptionProcessorAnomaly:
    def test_none_response(self, processor, keyframe):
        for text in ["none", "None", "NONE", "none.", "No", "nothing", "Nothing unusual", "n/a", "NA"]:
            cap = processor.build_caption(keyframe, "scene", "car", "driving", text)
            assert cap.anomaly is None, f"Expected None for input '{text}'"

    def test_no_prefix_response(self, processor, keyframe):
        cap = processor.build_caption(keyframe, "scene", "car", "driving", "No, everything looks normal.")
        assert cap.anomaly is None

    def test_actual_anomaly(self, processor, keyframe):
        cap = processor.build_caption(keyframe, "scene", "car", "driving", "A person running against traffic.")
        assert cap.anomaly == "A person running against traffic."

    def test_empty_returns_none(self, processor, keyframe):
        cap = processor.build_caption(keyframe, "scene", "car", "driving", "")
        assert cap.anomaly is None


class TestCaptionProcessorSerialization:
    def test_to_json(self, processor, keyframe):
        cap = VisualCaption(
            keyframe=keyframe,
            scene_description="A street.",
            objects=["car"],
            activity="driving",
            anomaly=None,
        )
        d = processor.to_json(cap)
        assert d["video_id"] == "v1"
        assert d["objects"] == ["car"]
        assert d["anomaly"] is None

    def test_captions_to_json(self, processor, keyframe):
        cap = VisualCaption(keyframe=keyframe, scene_description="test")
        result = processor.captions_to_json([cap, cap])
        assert len(result) == 2
