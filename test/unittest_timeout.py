"""Run unittest with a per-test timeout.

This module is a small wrapper around ``unittest``'s command line entry point.
It installs a timeout before test discovery so every ``TestCase`` method gets
the same limit without changing individual tests.
"""

import inspect
import logging
import os
import signal
import sys
import traceback
import unittest
from pathlib import Path
from urllib.parse import quote

DEFAULT_TIMEOUT_SECONDS = 30
ENV_VAR = "NDSCAN_TEST_TIMEOUT"
_TEST_DIR = Path(__file__).resolve().parent
_ROOT_DIR = _TEST_DIR.parent
_original_run = unittest.TestCase.run
_original_log = logging.Logger._log
_timeout_seconds = DEFAULT_TIMEOUT_SECONDS


def _ensure_test_import_paths():
    for path in (str(_ROOT_DIR), str(_TEST_DIR)):
        if path not in sys.path:
            sys.path.insert(0, path)


class TestTimeoutError(TimeoutError):
    pass


def _test_name(test):
    if hasattr(test, "id"):
        return test.id()
    return str(test)


def _exception_summary(err):
    exc_type, exc, _tb = err
    return "".join(traceback.format_exception_only(exc_type, exc)).strip()


def _github_escape(value):
    return value.replace("%", "%25").replace("\r", "%0D").replace("\n", "%0A")


def _github_escape_property(value):
    return _github_escape(value).replace(":", "%3A").replace(",", "%2C")


def _markdown_escape(value):
    return value.replace("|", "\\|").replace("\n", "<br>")


def _repo_relative_path(path):
    try:
        return Path(path).resolve().relative_to(_ROOT_DIR).as_posix()
    except ValueError:
        return None


def _test_location(test):
    method_name = getattr(test, "_testMethodName", None)
    if method_name is None:
        return None, None

    method = getattr(type(test), method_name, None)
    if method is None:
        return None, None

    try:
        source_path = inspect.getsourcefile(method)
        _source_lines, line_number = inspect.getsourcelines(method)
    except (OSError, TypeError):
        return None, None

    if source_path is None:
        return None, None

    relative_path = _repo_relative_path(source_path)
    if relative_path is None:
        return None, None

    return relative_path, line_number


def _github_source_url(path, line):
    repository = os.environ.get("GITHUB_REPOSITORY")
    sha = os.environ.get("GITHUB_SHA")
    if path is None or line is None or repository is None or sha is None:
        return None

    server_url = os.environ.get("GITHUB_SERVER_URL", "https://github.com").rstrip("/")
    quoted_path = quote(path, safe="/")
    return f"{server_url}/{repository}/blob/{sha}/{quoted_path}#L{line}"


class ConciseFailureResult(unittest.TextTestResult):
    def startTestRun(self):
        super().startTestRun()
        self.concise_failures = []

    def addError(self, test, err):
        super().addError(test, err)
        self._add_concise_failure("ERROR", test, err)

    def addFailure(self, test, err):
        super().addFailure(test, err)
        self._add_concise_failure("FAIL", test, err)

    def _add_concise_failure(self, kind, test, err):
        self.concise_failures.append((
            kind,
            _test_name(test),
            _exception_summary(err),
            *_test_location(test),
        ))

    def printConciseFailures(self):
        if not self.concise_failures:
            return

        self.stream.writeln()
        self.stream.writeln("Failed cases:")
        for kind, name, summary, _path, _line in self.concise_failures:
            self.stream.writeln(f"- {kind}: {name}: {summary}")

        self._write_github_summary()
        self._write_github_annotations()

    def printConciseSkips(self):
        if not self.skipped:
            return

        self.stream.writeln()
        self.stream.writeln("Skipped cases:")
        for test, reason in self.skipped:
            self.stream.writeln(f"- {test}: {reason}")

    def _write_github_summary(self):
        summary_path = os.environ.get("GITHUB_STEP_SUMMARY")
        if not summary_path:
            return

        with open(summary_path, "a", encoding="utf-8") as summary:
            summary.write("## Failed unit tests\n\n")
            summary.write("| Type | Test | Reason |\n")
            summary.write("| --- | --- | --- |\n")
            for kind, name, reason, path, line in self.concise_failures:
                source_url = _github_source_url(path, line)
                test_name = _markdown_escape(name)
                if source_url:
                    test_cell = f"[`{test_name}`]({source_url})"
                else:
                    test_cell = f"`{test_name}`"
                summary.write(
                    f"| {kind} | {test_cell} | {_markdown_escape(reason)} |\n"
                )

    def _write_github_annotations(self):
        if os.environ.get("GITHUB_ACTIONS") != "true":
            return

        for _kind, name, reason, path, line in self.concise_failures:
            properties = [f"title={_github_escape_property(name)}"]
            if path is not None:
                properties.insert(0, f"file={_github_escape_property(path)}")
            message = _github_escape(f"{name}: {reason}")
            print(f"::error {','.join(properties)}::{message}", file=sys.stderr)


class ConciseFailureRunner(unittest.TextTestRunner):
    resultclass = ConciseFailureResult

    def run(self, test):
        result = super().run(test)
        result.printConciseFailures()
        result.printConciseSkips()
        return result


def _handle_timeout(signum, frame):
    raise TestTimeoutError(
        f"test exceeded {_timeout_seconds} second timeout (set {ENV_VAR}=0 to disable)"
    )


def _run_with_timeout(self, result=None):
    if _timeout_seconds <= 0 or not hasattr(signal, "SIGALRM"):
        return _original_run(self, result)

    previous_handler = signal.getsignal(signal.SIGALRM)
    previous_timer = signal.getitimer(signal.ITIMER_REAL)
    signal.signal(signal.SIGALRM, _handle_timeout)
    signal.setitimer(signal.ITIMER_REAL, _timeout_seconds)
    try:
        return _original_run(self, result)
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous_handler)
        if previous_timer[0] > 0:
            signal.setitimer(signal.ITIMER_REAL, *previous_timer)


def install(timeout_seconds):
    global _timeout_seconds
    _timeout_seconds = timeout_seconds
    unittest.TestCase.run = _run_with_timeout


def main():
    _ensure_test_import_paths()
    timeout_seconds = float(os.environ.get(ENV_VAR, DEFAULT_TIMEOUT_SECONDS))
    install(timeout_seconds)
    unittest.main(module=None, testRunner=ConciseFailureRunner)


if __name__ == "__main__":
    main()
