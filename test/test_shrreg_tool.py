#!/usr/bin/env python3
import hashlib
import json
import os
import subprocess
import sys
import tempfile


tool, fixture = sys.argv[1:3]


def create(path, kind):
    subprocess.run([fixture, path, kind], check=True)


def run(path, *args):
    env = os.environ.copy()
    env["CUDA_DEVICE_MEMORY_SHARED_CACHE"] = path
    env.pop("LD_PRELOAD", None)
    return subprocess.run([tool, *args], env=env, text=True,
                          stdout=subprocess.PIPE, stderr=subprocess.PIPE)


def digest(path):
    with open(path, "rb") as stream:
        return hashlib.sha256(stream.read()).digest()


with tempfile.TemporaryDirectory(prefix="hami-shrreg-tool-") as temp:
    cache = os.path.join(temp, "region.cache")

    create(cache, "valid")
    before = digest(cache)
    result = run(cache, "--check", "--json")
    assert result.returncode == 0, result.stderr + result.stdout
    healthy = json.loads(result.stdout)
    assert healthy["initialized"] is True
    assert healthy["version"] == {"major": 1, "minor": 2}
    assert healthy["device_count"] == 2
    assert healthy["devices"][0]["memory_limit"] == 1024
    assert healthy["process_slots"] == {
        "declared": 2, "occupied": 1, "empty": 1, "unstable": 0
    }
    assert healthy["owner_status"] == "unowned"
    assert healthy["findings"] == []
    assert digest(cache) == before, "diagnostics modified a healthy cache"

    # A recorded owner is reported without claiming PID liveness.
    create(cache, "owned")
    owner_report = json.loads(run(cache, "--json").stdout)
    assert owner_report["owner_pid"] == 4242
    assert owner_report["owner_status"] == "pid_recorded_liveness_unknown"

    invalid_cases = {
        "uninitialized": "uninitialized",
        "truncated": "truncated_file",
        "bad_version": "incompatible_version",
        "bad_device_count": "device_count_out_of_range",
        "bad_process_count": "process_count_out_of_range",
        "unstable_slot": "concurrent_slot_update",
        "exhausted": "process_slots_exhausted",
    }
    for kind, finding_code in invalid_cases.items():
        create(cache, kind)
        before = digest(cache)
        json_result = run(cache, "--json")
        assert json_result.returncode == 0, json_result.stderr
        report = json.loads(json_result.stdout)
        assert finding_code in [item["code"] for item in report["findings"]], report
        assert digest(cache) == before, f"--json modified {kind} cache"
        check_result = run(cache, "--check")
        assert check_result.returncode != 0, (kind, check_result.stdout, check_result.stderr)
        assert digest(cache) == before, f"--check modified {kind} cache"

    missing = os.path.join(temp, "missing.cache")
    missing_result = run(missing, "--check", "--json")
    assert missing_result.returncode != 0
    missing_report = json.loads(missing_result.stdout)
    assert missing_report["findings"][0]["code"] == "missing_file"
    assert not os.path.exists(missing), "diagnostics created a missing cache"

    help_result = run(cache, "--help")
    assert help_result.returncode == 0
    assert "--check" in help_result.stdout and "--json" in help_result.stdout
    version_result = run(cache, "--version")
    assert version_result.returncode == 0
    assert "shrreg size:" in version_result.stdout

print("shrreg-tool read-only diagnostics passed")
