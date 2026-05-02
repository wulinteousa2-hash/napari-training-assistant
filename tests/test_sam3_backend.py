from contextlib import nullcontext

import numpy as np
import pytest

from napari_training_assistant.sam3_backend.errors import SAM3InferenceError
from napari_training_assistant.sam3_backend.inference import SAM3PreviewEngine
import napari_training_assistant.sam3_backend.reference_backend as reference_backend_module
from napari_training_assistant.sam3_backend.reference_backend import ReferenceSAM3Backend
from napari_training_assistant.sam3_backend.prompt_converter import (
    convert_box_prompt,
    convert_point_prompt,
)


def test_convert_box_prompt_keeps_multiple_rectangles():
    first = np.asarray([[1, 2], [1, 7], [5, 7], [5, 2]], dtype=float)
    second = np.asarray([[10, 20], [10, 30], [15, 30], [15, 20]], dtype=float)

    prompt = convert_box_prompt([first, second])

    np.testing.assert_allclose(
        prompt.boxes_xyxy,
        np.asarray([[2, 1, 7, 5], [20, 10, 30, 15]], dtype=np.float32),
    )


def test_labels_from_masks_preserves_all_prompt_masks():
    masks = np.zeros((2, 6, 6), dtype=bool)
    masks[0, 1:3, 1:3] = True
    masks[1, 3:5, 3:5] = True

    labels = SAM3PreviewEngine._labels_from_masks(masks)

    assert labels[1, 1] == 1
    assert labels[3, 3] == 2
    assert int(labels.max()) == 2


def test_exemplar_box_uses_normalized_center_box():
    normalized = SAM3PreviewEngine._box_yxyx_to_normalized_cxcywh(
        np.asarray([10, 20, 30, 60], dtype=np.float32),
        (100, 200),
    )

    assert normalized == (0.2, 0.2, 0.2, 0.2)


def test_convert_point_prompt_preserves_positive_negative_labels():
    prompt = convert_point_prompt(
        {
            "data": np.asarray([[4, 5], [6, 7]], dtype=float),
            "properties": {"polarity": np.asarray(["positive", "negative"], dtype=object)},
        }
    )

    np.testing.assert_allclose(prompt.points_xy, np.asarray([[5, 4], [7, 6]], dtype=np.float32))
    np.testing.assert_array_equal(prompt.labels, np.asarray([1, 0], dtype=np.int64))


def test_preview_requires_reference_adapter(monkeypatch, tmp_path):
    engine = SAM3PreviewEngine()

    def raise_import_error(**kwargs):
        raise ImportError("missing reference adapter")

    monkeypatch.setattr(engine, "_run_reference_adapter_preview", raise_import_error)

    with pytest.raises(SAM3InferenceError, match="napari-sam3-assistant"):
        engine.run_preview(
            image=np.zeros((8, 8), dtype=np.uint8),
            mode="2d_exemplar",
            model_dir=str(tmp_path),
            device="cpu",
            prompt_data=[
                np.asarray([[1, 1], [1, 4], [4, 4], [4, 1]], dtype=float),
            ],
        )


def test_3d_multiplex_uses_sam31_path(monkeypatch, tmp_path):
    engine = SAM3PreviewEngine()

    def fail_2d_adapter(**kwargs):
        raise AssertionError("3D multiplex must not use the 2D SAM3 adapter")

    def fake_multiplex(**kwargs):
        assert kwargs["model_dir"] == str(tmp_path)
        assert len(kwargs["prompt_data"]) == 1
        yield type(
            "Result",
            (),
            {
                "labels": np.zeros((3, 8, 8), dtype=np.uint16),
                "metadata": {"backend_status": "napari_sam3_assistant_sam31_multiplex"},
            },
        )()

    monkeypatch.setattr(engine, "_run_reference_adapter_preview", fail_2d_adapter)
    monkeypatch.setattr(engine, "iter_multiplex_preview", fake_multiplex)

    results = list(engine.iter_multiplex_preview(
        image=np.zeros((3, 8, 8), dtype=np.uint8),
        model_dir=str(tmp_path),
        device="cpu",
        prompt_data=[
            np.asarray([[1, 2, 2], [1, 2, 5], [1, 5, 5], [1, 5, 2]], dtype=float),
        ],
        image_layer_name="image",
        dims_current_step=(1, 0, 0),
        threshold=0.5,
        compile_model=False,
    ))

    result = results[-1]
    assert result.labels.shape == (3, 8, 8)
    assert result.metadata["backend_status"] == "napari_sam3_assistant_sam31_multiplex"


def test_reference_multiplex_preload_matches_sam3_assistant_compile_default(monkeypatch):
    backend = ReferenceSAM3Backend()
    calls = []

    class FakeAdapter:
        video_predictor = None

        def load_video(self):
            self.video_predictor = type(
                "Predictor",
                (),
                {"async_loading_frames": True},
            )()
            calls.append("load_video")

    def fake_ensure_adapter(**kwargs):
        calls.append(kwargs)
        return FakeAdapter()

    monkeypatch.setattr(backend, "_ensure_adapter", fake_ensure_adapter)

    backend.preload_multiplex_model(
        model_dir="/models/sam31",
        device="cuda",
        threshold=0.5,
    )

    assert calls[0]["model_type"] == "sam3.1"
    assert calls[0]["compile_model"] is False
    assert calls[1] == "load_video"


def test_reference_multiplex_performance_mode_keeps_async_frame_loading_enabled():
    backend = ReferenceSAM3Backend()
    adapter = type(
        "Adapter",
        (),
        {
            "video_predictor": type(
                "Predictor",
                (),
                {"async_loading_frames": True},
            )()
        },
    )()

    backend.configure_multiplex_predictor(adapter)

    assert adapter.video_predictor.async_loading_frames is True


def test_reference_multiplex_performance_mode_leaves_async_frame_loading_unchanged():
    backend = ReferenceSAM3Backend()
    adapter = type(
        "Adapter",
        (),
        {
            "video_predictor": type(
                "Predictor",
                (),
                {"async_loading_frames": False},
            )()
        },
    )()

    backend.configure_multiplex_predictor(adapter)

    assert adapter.video_predictor.async_loading_frames is False


def test_reference_multiplex_windows_compatibility_can_disable_async_loading(monkeypatch):
    backend = ReferenceSAM3Backend()
    adapter = type(
        "Adapter",
        (),
        {
            "video_predictor": type(
                "Predictor",
                (),
                {"async_loading_frames": True},
            )()
        },
    )()

    monkeypatch.setattr(reference_backend_module.sys, "platform", "win32")

    backend.configure_multiplex_predictor(
        adapter,
        windows_compatibility_mode=True,
    )

    assert adapter.video_predictor.async_loading_frames is False


def test_reference_multiplex_non_windows_compatibility_leaves_async_loading_unchanged(monkeypatch):
    backend = ReferenceSAM3Backend()
    adapter = type(
        "Adapter",
        (),
        {
            "video_predictor": type(
                "Predictor",
                (),
                {"async_loading_frames": False},
            )()
        },
    )()

    monkeypatch.setattr(reference_backend_module.sys, "platform", "linux")

    backend.configure_multiplex_predictor(
        adapter,
        windows_compatibility_mode=True,
    )

    assert adapter.video_predictor.async_loading_frames is False


def test_reference_multiplex_adapter_cache_key_ignores_threshold(monkeypatch, tmp_path):
    backend = ReferenceSAM3Backend()
    checkpoint = tmp_path / "sam3.1_multiplex.pt"
    checkpoint.write_bytes(b"checkpoint")
    created = []

    class FakeConfig:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class FakeAdapter:
        def __init__(self, config):
            created.append(config)

    import napari_sam3_assistant.adapters as adapters

    monkeypatch.setattr(adapters, "Sam3Adapter", FakeAdapter)
    monkeypatch.setattr(adapters, "Sam3AdapterConfig", FakeConfig)

    first = backend._ensure_adapter(
        model_dir=str(tmp_path),
        model_type="sam3.1",
        device="cuda",
        threshold=0.4,
        compile_model=False,
    )
    second = backend._ensure_adapter(
        model_dir=str(tmp_path),
        model_type="sam3.1",
        device="cuda",
        threshold=0.9,
        compile_model=False,
    )

    assert first is second
    assert len(created) == 1


def test_reference_multiplex_reuses_session_and_frame_directory():
    backend = ReferenceSAM3Backend()
    calls = {"write_stack": 0, "start_session": 0}

    class FakePredictor:
        async_loading_frames = True

        def __init__(self):
            self._all_inference_states = {}

        def handle_request(self, request):
            calls["start_session"] += 1
            session_id = f"session-{calls['start_session']}"
            self._all_inference_states[session_id] = {"state": {}, "session_id": session_id}
            return {"session_id": session_id}

    class FakeAdapter:
        video_predictor = FakePredictor()

        def load_video(self):
            raise AssertionError("video predictor is already loaded")

        def _install_video_backend_compatibility(self):
            pass

        def _write_stack_as_jpeg_dir(self, image, bundle, video_dir):
            calls["write_stack"] += 1

        def _inference_context(self):
            return nullcontext()

        def has_video_session(self, session):
            return session.session_id in self.video_predictor._all_inference_states

    bundle = type(
        "Bundle",
        (),
        {
            "task": "segment_3d",
            "image": type("ImageSelection", (), {"layer_name": "image"})(),
        },
    )()
    image = np.zeros((3, 8, 8), dtype=np.uint8)
    key = ("model", "image", id(image), image.shape, str(image.dtype), "cuda", 0.5, False)

    first = backend.start_or_reuse_video_session(
        adapter=FakeAdapter(),
        image=image,
        bundle=bundle,
        session_key=key,
    )
    second = backend.start_or_reuse_video_session(
        adapter=FakeAdapter(),
        image=image,
        bundle=bundle,
        session_key=key,
    )

    assert first is second
    assert calls == {"write_stack": 1, "start_session": 1}


def test_video_result_labels_do_not_overwrite_existing_frame_with_empty_result():
    labels = np.ones((2, 4, 4), dtype=np.uint32)
    result = type(
        "Result",
        (),
        {
            "frame_index": 1,
            "labels": np.zeros((4, 4), dtype=np.uint32),
            "masks": None,
            "object_ids": None,
        },
    )()

    SAM3PreviewEngine._insert_video_result_labels(labels, result)

    assert labels[1].any()


def test_exemplar_prompts_use_video_visual_prompt_request():
    requests = []

    class FakeVideoPredictor:
        def handle_request(self, request):
            requests.append(request)
            if request["type"] == "start_session":
                return {"session_id": "session"}
            if request["type"] == "close_session":
                return {}
            return {
                "outputs": {
                    "out_binary_masks": np.ones((1, 4, 4), dtype=bool),
                    "out_probs": np.asarray([0.9], dtype=np.float32),
                }
            }

    masks, scores = SAM3PreviewEngine._run_video_exemplar_prompts(
        FakeVideoPredictor(),
        np.zeros((10, 20, 3), dtype=np.uint8),
        convert_box_prompt([np.asarray([[1, 2], [1, 6], [5, 6], [5, 2]], dtype=float)]),
        0.5,
    )

    add_prompt = requests[1]
    assert add_prompt["type"] == "add_prompt"
    assert add_prompt["text"] == "visual"
    assert add_prompt["bounding_boxes"] == [[0.1, 0.1, 0.2, 0.4]]
    assert add_prompt["bounding_box_labels"] == [1]
    assert masks.shape == (1, 4, 4)
    np.testing.assert_allclose(scores, np.asarray([0.9], dtype=np.float32))
