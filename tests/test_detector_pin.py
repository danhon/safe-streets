"""The detector refuses model files that aren't the pinned ones (the build
checks them too; this catches files changed on disk afterwards)."""

import pytest

pytest.importorskip("openvino")

from safe_streets.detector import RTMDET_DIR, RTMDet


def test_a_detector_that_isnt_the_pinned_one_is_refused(tmp_path):
    folder = tmp_path / RTMDET_DIR
    folder.mkdir()
    (folder / "model.xml").write_text("<net/>")
    (folder / "model.bin").write_bytes(b"\0" * 16)
    with pytest.raises(ValueError, match="sha256"):
        RTMDet(tmp_path)
