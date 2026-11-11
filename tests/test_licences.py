import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import check_licences as C  # noqa: E402


def pkg(name, lic):
    return {"Name": name, "Version": "1", "License": lic}


def test_permissive_licences_pass():
    assert C.problems([pkg("numpy", "BSD License"), pkg("fastapi", "MIT License"),
                       pkg("openvino", "Apache Software License")]) == []


def test_gpl_and_agpl_fail():
    hits = C.problems([pkg("a", "GNU General Public License v3 (GPLv3)"), pkg("b", "AGPL-3.0"),
                       pkg("c", "GNU Affero General Public License v3")])
    assert [h.split(":")[0] for h in hits] == ["a", "b", "c"]


def test_lgpl_is_allowed():
    assert C.problems([pkg("x", "GNU Lesser General Public License v2 or later (LGPLv2+)"),
                       pkg("y", "LGPL-2.1")]) == []


def test_ultralytics_fails_by_name_whatever_its_licence():
    assert C.problems([pkg("ultralytics", "MIT")]) == ["ultralytics: banned by name (AGPL-3.0 upstream)"]
