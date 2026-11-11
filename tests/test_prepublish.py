import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import prepublish  # noqa: E402


def git(repo, *args):
    subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True)


@pytest.fixture
def repo(tmp_path):
    r = tmp_path / "repo"
    r.mkdir()
    git(r, "init", "-q")
    git(r, "config", "user.email", "t@example.com")
    git(r, "config", "user.name", "t")
    (r / "README.md").write_text("A speed camera.\n")
    git(r, "add", ".")
    git(r, "commit", "-qm", "init")
    return r


@pytest.fixture
def deny(tmp_path):
    p = tmp_path / "deny.txt"
    p.write_text("# comment\nsecrethost\nre:\\bAcme\\b\n")
    return p


def problems(repo, deny, **kw):
    return [h for h in prepublish.scan(repo, deny, **kw) if "gitleaks" not in h]


def test_a_clean_tree_passes(repo, deny):
    assert problems(repo, deny) == []


def test_a_photo_fails(repo, deny):
    (repo / "street.jpg").write_bytes(b"\xff\xd8\xff")
    assert any("street.jpg: image or video outside docs/img/" in h for h in problems(repo, deny))


def test_only_listed_synthetic_pictures_pass(repo, deny):
    (repo / "docs" / "img").mkdir(parents=True)
    (repo / "docs" / "img" / "scene.png").write_bytes(b"\x89PNG")
    (repo / "docs" / "img" / "other.png").write_bytes(b"\x89PNG")
    (repo / "docs" / "img" / "SYNTHETIC").write_text("scene.png\n")
    hits = problems(repo, deny)
    assert hits == ["docs/img/other.png: image or video not listed in docs/img/SYNTHETIC"]


def test_a_denylist_term_fails_without_printing_it(repo, deny):
    (repo / "notes.md").write_text("ok\nfrom SecretHost.example\n")
    (repo / "b.md").write_text("Acme Corp\nacmeish\n")
    hits = problems(repo, deny)
    assert "notes.md:2: matches denylist line 2" in hits
    assert "b.md:1: matches denylist line 3" in hits
    assert not any("b.md:2" in h for h in hits)  # the regex is case- and word-exact
    assert not any("secrethost" in h.lower() for h in hits)


def test_the_licence_may_name_the_author(repo, deny):
    (repo / "LICENSE").write_text("Copyright (c) 2026 Acme\n")
    assert problems(repo, deny) == []


def test_the_licence_may_name_the_author_in_history_too(repo, deny):
    (repo / "LICENSE").write_text("Copyright (c) 2026 Acme\n")
    git(repo, "add", ".")
    git(repo, "commit", "-qm", "licence")
    assert problems(repo, deny, history=True) == []


def test_the_repositorys_own_address_may_name_its_owner(repo, deny):
    (repo / "README.md").write_text("git clone https://github.com/secrethost/safe-streets\nsecrethost.example\n")
    assert problems(repo, deny) == ["README.md:2: matches denylist line 2"]


def test_a_private_address_fails(repo, deny):
    (repo / "x.py").write_text("HOST = '192.168.0.42'\nPUBLIC = '8.8.8.8'\nTS = '100.64.0.9'\n")  # prepublish: allow
    hits = problems(repo, deny)
    assert "x.py:1: private address 192.168.0.42" in hits  # prepublish: allow
    assert "x.py:3: private address 100.64.0.9" in hits  # prepublish: allow
    assert not any("8.8.8.8" in h for h in hits)


def test_history_finds_a_deleted_file(repo, deny):
    (repo / "old.mp4").write_bytes(b"\0\0\0 ftyp")
    git(repo, "add", ".")
    git(repo, "commit", "-qm", "oops")
    git(repo, "rm", "-q", "old.mp4")
    git(repo, "commit", "-qm", "gone")
    assert problems(repo, deny) == []
    assert any(h.startswith("old.mp4 (in history") for h in problems(repo, deny, history=True))


def test_no_denylist_refuses(repo, tmp_path, capsys):
    assert prepublish.main([str(repo), "--denylist", str(tmp_path / "missing.txt")]) == 2


def test_an_allowed_line_is_skipped(repo, deny):
    (repo / "y.md").write_text("see 192.168.0.42 here <!-- prepublish: allow -->\n")  # prepublish: allow
    assert problems(repo, deny) == []
