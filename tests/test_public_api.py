from importlib.metadata import version

from trace_ml import __version__, verify


def test_top_level_verify_is_public():
    assert callable(verify)


def test_package_version_matches_installed_metadata():
    assert __version__ == version("trace-ml-verifier")
