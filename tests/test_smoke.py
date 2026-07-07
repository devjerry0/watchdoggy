import doggy


def test_package_imports_and_has_version():
    assert isinstance(doggy.__version__, str)
    assert doggy.__version__


def test_entry_point_shim_survives():
    # The Pi's installed console script resolves doggy.main:main under --no-sync;
    # this import path must never break.
    from doggy.main import main as entry
    from doggy.app import main as real
    assert entry is real
