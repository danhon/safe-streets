import os

from safe_streets.gpu import busy_ns

FDINFO = """pos:\t0
flags:\t0100002
drm-driver:\ti915
drm-client-id:\t{cid}
drm-pdev:\t0000:00:02.0
drm-total-system0:\t24616 KiB
drm-engine-render:\t{render} ns
drm-engine-copy:\t0 ns
drm-engine-video:\t{video} ns
drm-engine-capacity-video:\t2
drm-engine-video-enhance:\t0 ns
"""


def proc(tmp_path, fds):
    """A fake /proc/self: fds maps fd -> (link target, fdinfo text or None)."""
    (tmp_path / "fd").mkdir()
    (tmp_path / "fdinfo").mkdir()
    for fd, (target, info) in fds.items():
        os.symlink(target, tmp_path / "fd" / str(fd))  # dangling is fine: only read as a link
        (tmp_path / "fdinfo" / str(fd)).write_text(info or "pos:\t0\n")
    return tmp_path


def test_sums_every_engine_of_the_processs_drm_clients(tmp_path):
    p = proc(tmp_path, {
        3: ("/dev/null", None),
        5: ("/dev/dri/renderD128", FDINFO.format(cid=60, render=1_500_000, video=250_000)),
        7: ("socket:[1234]", None),
    })
    assert busy_ns(p) == 1_750_000  # capacity is a count, not time


def test_a_client_open_on_two_fds_counts_once(tmp_path):
    info = FDINFO.format(cid=60, render=1_000, video=0)
    p = proc(tmp_path, {5: ("/dev/dri/renderD128", info), 6: ("/dev/dri/renderD128", info),
                        8: ("/dev/dri/card0", FDINFO.format(cid=61, render=10, video=0))})
    assert busy_ns(p) == 1_010


def test_none_without_a_gpu(tmp_path):
    assert busy_ns(proc(tmp_path, {3: ("/dev/null", None)})) is None
    assert busy_ns(tmp_path / "missing") is None


def test_none_when_the_driver_reports_no_engine_times(tmp_path):
    # An older kernel: the fd is open but fdinfo has no drm-engine lines.
    assert busy_ns(proc(tmp_path, {5: ("/dev/dri/renderD128", "pos:\t0\nflags:\t02\n")})) is None


def test_reads_this_process_without_failing():
    assert busy_ns() is None or busy_ns() >= 0
