import pytest
from verifiers.v1._validation_scope import validation_scope


class ScopedTests:
    @pytest.hookimpl(hookwrapper=True)
    def pytest_runtest_call(self, item):
        with validation_scope():
            yield


raise SystemExit(
    pytest.main(
        [
            "-q",
            "tests/test_native_invocation_source.py",
            "tests/test_native_invocation_inventory.py",
            "tests/test_manifest_credit.py",
            "tests/test_manifest_record_retained_credit.py",
        ],
        plugins=[ScopedTests()],
    )
)
