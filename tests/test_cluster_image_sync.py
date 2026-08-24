from __future__ import annotations

import os
import pathlib
import shlex
import shutil
import stat
import subprocess
import textwrap


ROOT = pathlib.Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / "deployments/anemll-vision/start-cluster.sh"


def _find_bash() -> str:
    if os.name != "nt":
        bash = shutil.which("bash")
    else:
        bash = shutil.which("bash.exe")
        if bash is None and "LOCALAPPDATA" in os.environ:
            candidate = pathlib.Path(os.environ["LOCALAPPDATA"]) / "hermes/git/usr/bin/bash.exe"
            bash = str(candidate) if candidate.is_file() else None
    if bash is None:
        raise RuntimeError("A POSIX Bash executable is required for launcher tests")
    return bash


def _bash_path(path: pathlib.Path) -> str:
    resolved = path.resolve()
    if os.name != "nt":
        return resolved.as_posix()
    drive = resolved.drive.rstrip(":").lower()
    tail = resolved.as_posix().split(":", 1)[1]
    return f"/{drive}{tail}"


def _executable(path: pathlib.Path, content: str) -> None:
    path.write_text(content, newline="\n")
    path.chmod(path.stat().st_mode | stat.S_IXUSR)


def _run_launcher(
    tmp_path: pathlib.Path,
    *,
    local_image_id: str = "sha256:same",
    remote_image_id: str = "sha256:same",
    expected_image_id: str = "",
    worker_image: str = "anemll-dsv4-vision:test",
    worker_expected_image_id: str = "",
    worker_repo: str = "/srv/dsv4/deployments/anemll-vision",
    worker_env_name: str = "worker.env",
) -> tuple[subprocess.CompletedProcess[str], str]:
    root = tmp_path / "deployment"
    bindir = tmp_path / "bin"
    root.mkdir()
    bindir.mkdir()
    shutil.copy2(LAUNCHER, root / "start-cluster.sh")
    log = tmp_path / "actions.log"

    (root / "head.env").write_text(
        textwrap.dedent(
            f"""\
            WORKER_SSH=worker.example
            WORKER_REPO_DIR={shlex.quote(worker_repo)}
            DSPARK_VLLM_IMAGE=anemll-dsv4-vision:test
            DSPARK_VLLM_IMAGE_ID={expected_image_id}
            VLLM_PORT=8000
            """
        ),
        newline="\n",
    )
    (root / worker_env_name).write_text(
        f"NODE_RANK=1\nDSPARK_VLLM_IMAGE={worker_image}\n"
        f"DSPARK_VLLM_IMAGE_ID={worker_expected_image_id}\n",
        newline="\n",
    )

    _executable(
        root / "start-node.sh",
        "#!/usr/bin/env bash\nprintf 'local-start %s\\n' \"$*\" >> \"$ACTION_LOG\"\n",
    )
    _executable(
        bindir / "docker",
        "#!/usr/bin/env bash\n"
        "printf 'docker %s\\n' \"$*\" >> \"$ACTION_LOG\"\n"
        "if [[ \"$*\" == *'image inspect'* ]]; then\n"
        "  [[ -n \"${LOCAL_IMAGE_ID:-}\" ]] || exit 1\n"
        "  printf '%s\\n' \"$LOCAL_IMAGE_ID\"\n"
        "fi\n",
    )
    _executable(
        bindir / "ssh",
        "#!/usr/bin/env bash\n"
        "printf 'ssh %s\\n' \"$*\" >> \"$ACTION_LOG\"\n"
        "if [[ \"$*\" == *'source '* ]]; then\n"
        "  printf '%s\\n%s\\n' \"$WORKER_IMAGE\" \"$WORKER_EXPECTED_IMAGE_ID\"\n"
        "elif [[ \"$*\" == *'docker image inspect'* ]]; then\n"
        "  [[ -n \"${REMOTE_IMAGE_ID:-}\" ]] || exit 1\n"
        "  printf '%s\\n' \"$REMOTE_IMAGE_ID\"\n"
        "fi\n",
    )
    _executable(bindir / "sleep", "#!/usr/bin/env bash\nexit 0\n")
    _executable(bindir / "curl", "#!/usr/bin/env bash\nexit 0\n")

    env = dict(os.environ)
    env.update(
        {
            "LOCAL_IMAGE_ID": local_image_id,
            "REMOTE_IMAGE_ID": remote_image_id,
            "WORKER_IMAGE": worker_image,
            "WORKER_EXPECTED_IMAGE_ID": worker_expected_image_id,
            "ACTION_LOG": _bash_path(log),
            "PATH": f"{_bash_path(bindir)}:{env['PATH']}",
        }
    )
    result = subprocess.run(
        [
            _find_bash(),
            _bash_path(root / "start-cluster.sh"),
            "head.env",
            worker_env_name,
        ],
        cwd=root,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    return result, log.read_text() if log.exists() else ""


def test_matching_cluster_image_ids_allow_both_ranks_to_start(tmp_path: pathlib.Path) -> None:
    result, actions = _run_launcher(tmp_path)

    assert result.returncode == 0, result.stderr
    assert "Docker image consistency check passed" in result.stdout
    assert "./start-node.sh worker.env" in actions
    assert "local-start" in actions


def test_image_mismatch_aborts_before_either_rank_starts(tmp_path: pathlib.Path) -> None:
    result, actions = _run_launcher(tmp_path, remote_image_id="sha256:different")

    assert result.returncode != 0
    assert "Docker image mismatch" in result.stderr
    assert "local-start" not in actions
    assert "./start-node.sh worker.env" not in actions


def test_worker_environment_image_is_the_one_verified(tmp_path: pathlib.Path) -> None:
    result, actions = _run_launcher(
        tmp_path,
        remote_image_id="sha256:different",
        worker_image="anemll-dsv4-vision:worker-only",
    )

    assert result.returncode != 0
    assert "Docker image mismatch" in result.stderr
    assert "anemll-dsv4-vision:worker-only" in actions
    assert "local-start" not in actions
    assert "./start-node.sh worker.env" not in actions


def test_missing_worker_image_aborts_before_launch(tmp_path: pathlib.Path) -> None:
    result, actions = _run_launcher(tmp_path, remote_image_id="")

    assert result.returncode != 0
    assert "Could not inspect vision image" in result.stderr
    assert "local-start" not in actions
    assert "./start-node.sh worker.env" not in actions


def test_expected_image_id_is_enforced_before_launch(tmp_path: pathlib.Path) -> None:
    result, actions = _run_launcher(
        tmp_path,
        local_image_id="sha256:same",
        remote_image_id="sha256:same",
        expected_image_id="sha256:expected",
    )

    assert result.returncode != 0
    assert "does not match DSPARK_VLLM_IMAGE_ID" in result.stderr
    assert "local-start" not in actions
    assert "./start-node.sh worker.env" not in actions


def test_worker_expected_image_id_is_enforced_before_launch(tmp_path: pathlib.Path) -> None:
    result, actions = _run_launcher(
        tmp_path,
        worker_expected_image_id="sha256:expected",
    )

    assert result.returncode != 0
    assert "worker DSPARK_VLLM_IMAGE_ID" in result.stderr
    assert "local-start" not in actions
    assert "./start-node.sh worker.env" not in actions


def test_worker_launch_shell_escapes_repository_and_env_paths(
    tmp_path: pathlib.Path,
) -> None:
    result, actions = _run_launcher(
        tmp_path,
        worker_repo="/srv/dsv4/worker's repo",
        worker_env_name="worker env",
    )

    assert result.returncode == 0, result.stderr
    assert "cd /srv/dsv4/worker\\'s\\ repo" in actions
    assert "./start-node.sh worker\\ env" in actions
    assert "cd '/srv/dsv4/worker's repo'" not in actions
    launcher = LAUNCHER.read_text()
    assert 'printf -v worker_launch_cmd "cd %q && ./start-node.sh %q"' in launcher
    assert '"cd \'$WORKER_REPO_DIR\' && ./start-node.sh \'$worker_env\'"' not in launcher
