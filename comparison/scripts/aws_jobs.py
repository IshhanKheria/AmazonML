"""Upload frozen inputs and launch budget-gated SageMaker Processing pilots."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import time
from typing import Iterable

import boto3
from botocore.config import Config
from botocore.exceptions import ClientError
from boto3.s3.transfer import TransferConfig


REGION = "eu-north-1"
PROFILE = "amazon-ml-codex"
BUCKET = "amazon-sagemaker-303936065943-eu-north-1-4g5ie80zojahhv"
ROLE_ARN = "arn:aws:iam::303936065943:role/service-role/AmazonSageMakerAdminIAMExecutionRole"
IMAGE_URI = "662702820516.dkr.ecr.eu-north-1.amazonaws.com/sagemaker-scikit-learn:1.2-1-cpu-py3"
DATASET_ID = "70bc1d8a16c667e0155c2105d0ab2ebe41d7e7a85d8a529e3ca81c6c3a5af037"
PROJECT_PREFIX = "ber-comparison"
INSTANCE_TYPE = "ml.m6i.2xlarge"
HOURLY_USD = 0.49
MAX_RUNTIME_SECONDS = 7_200
PILOT_ROWS = 10_000

ROOT = Path(__file__).resolve().parents[2]
ORIGINAL_ROOT = ROOT.parent / "AmazonML"
DATASET_ROOT = ORIGINAL_ROOT / "6ab10eb3b23ba_student_resource" / "student_resource" / "dataset"
LEDGER = ROOT / "comparison" / "aws" / "budget_ledger.tsv"
WORKER = ROOT / "comparison" / "scripts" / "sagemaker_worker.py"
PREFLIGHT_REPORT = ROOT / "comparison" / "reports" / "aws_preflight_2026-09-25.json"

BRANCHES = {
    "m1": {
        "sha": "e04336d23d6da988218ad1f5afe4abb413dd5472",
        "root": ROOT / "branch_m1" / "artifacts" / "functional" / "_embedded_package"
        / "e04336d23d6da988218ad1f5afe4abb413dd5472" / "code" / "business_entity_resolution",
    },
    "hardening": {
        "sha": "a658723ef81f5c7c70fd15f720c2e440f50f4694",
        "root": ROOT / "branch_codex_competition_hardening" / "artifacts" / "functional" / "_embedded_package"
        / "a658723ef81f5c7c70fd15f720c2e440f50f4694" / "code" / "business_entity_resolution",
    },
}

EXPECTED_TRAIN = {
    "train_source1.tsv": (210069713, "591af0e1dfeb65cab71ea6ee8cb69df00f92d6ba6fa79e05746c938775d14973"),
    "train_source2.tsv": (489301488, "6336c1a055eec79cf8a6d99fdc8d32a2e4d9dc2662e00963cb35d66b89ed09ed"),
    "train_source3.tsv": (503705637, "67da22f5151898ff3006febd836c1a159e97ae95efa7257a5aff4fda685e58e9"),
    "train_ground_truth.tsv": (127015583, DATASET_ID),
}


def session():
    os.environ["AWS_SDK_UA_APP_ID"] = "AWSSkill-SageMaker"
    boto_session = boto3.Session(profile_name=PROFILE, region_name=REGION)
    retry = Config(
        retries={"mode": "adaptive", "total_max_attempts": 10},
        connect_timeout=10,
        read_timeout=90,
        user_agent_appid="AWSSkill-SageMaker",
    )
    return boto_session, retry


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_train_files() -> list[Path]:
    files = []
    for name, (size, digest) in EXPECTED_TRAIN.items():
        path = DATASET_ROOT / "train" / name
        if path.stat().st_size != size or file_sha256(path) != digest:
            raise RuntimeError(f"Frozen input verification failed: {path}")
        files.append(path)
    return files


def upload_file(s3, source: Path, key: str) -> str:
    try:
        current = s3.head_object(Bucket=BUCKET, Key=key)
        if int(current["ContentLength"]) == source.stat().st_size:
            return "SKIPPED_EXISTING"
    except s3.exceptions.ClientError as exc:
        code = exc.response.get("Error", {}).get("Code")
        if code not in {"404", "NoSuchKey", "NotFound"}:
            raise
    transfer = TransferConfig(
        multipart_threshold=64 * 1024 * 1024,
        multipart_chunksize=64 * 1024 * 1024,
        max_concurrency=8,
        use_threads=True,
    )
    s3.upload_file(str(source), BUCKET, key, Config=transfer)
    return "UPLOADED"


def upload_train() -> None:
    boto_session, retry = session()
    s3 = boto_session.client("s3", config=retry)
    for path in verify_train_files():
        key = f"{PROJECT_PREFIX}/shared/data/{DATASET_ID}/train/{path.name}"
        print(json.dumps({"file": str(path), "s3_key": key, "result": upload_file(s3, path, key)}), flush=True)


def iter_code_files(branch_root: Path) -> Iterable[tuple[Path, str]]:
    ignored = {"__pycache__", ".pytest_cache", ".mypy_cache"}
    for path in sorted(branch_root.rglob("*")):
        if path.is_file() and not any(part in ignored for part in path.parts):
            yield path, (Path("business_entity_resolution") / path.relative_to(branch_root)).as_posix()
    yield WORKER, WORKER.name


def upload_code(branch: str) -> None:
    branch_info = BRANCHES[branch]
    branch_root = Path(branch_info["root"])
    if not branch_root.is_dir():
        raise FileNotFoundError(branch_root)
    boto_session, retry = session()
    s3 = boto_session.client("s3", config=retry)
    prefix = f"{PROJECT_PREFIX}/code/{branch}/{branch_info['sha']}"
    count = 0
    for path, relative in iter_code_files(branch_root):
        key = f"{prefix}/{relative}"
        upload_file(s3, path, key)
        count += 1
    print(json.dumps({"branch": branch, "sha": branch_info["sha"], "files": count, "s3_prefix": prefix}))


def projected_cost(max_runtime_seconds: int = MAX_RUNTIME_SECONDS) -> float:
    return round((max_runtime_seconds / 3600) * HOURLY_USD * 1.25, 4)


def require_fresh_preflight() -> dict:
    if not PREFLIGHT_REPORT.is_file():
        raise RuntimeError(f"Required preflight report is missing: {PREFLIGHT_REPORT}")
    report = json.loads(PREFLIGHT_REPORT.read_text(encoding="utf-8"))
    completed = datetime.fromisoformat(report["completed_at"])
    age_seconds = (datetime.now(timezone.utc) - completed).total_seconds()
    requirements = {
        "status": report.get("status") == "PASSED",
        "paid_execution_allowed": report.get("paid_execution_allowed") is True,
        "region": report.get("required_region") == REGION,
        "execution_role": report.get("execution_role_arn") == ROLE_ARN,
        "pass_role": report.get("checks", {}).get("pass_role", {}).get("status") == "PASSROLE_CONFIRMED",
        "execution_role_trust": (
            report.get("checks", {}).get("execution_role_trust", {}).get("status") == "CONFIRMED"
        ),
        "execution_role_s3": report.get("checks", {}).get("execution_role_s3", {}).get("status") == "CONFIRMED",
        "caller_sagemaker_controls": (
            report.get("checks", {}).get("caller_sagemaker_controls", {}).get("status") == "CONFIRMED"
        ),
        "fresh_within_one_hour": 0 <= age_seconds <= 3_600,
    }
    failed = [name for name, passed in requirements.items() if not passed]
    if failed:
        raise RuntimeError(f"Paid execution blocked by preflight checks: {failed}")
    return {"report": str(PREFLIGHT_REPORT), "age_seconds": age_seconds, "requirements": requirements}


def append_ledger(values: list[str]) -> None:
    LEDGER.parent.mkdir(parents=True, exist_ok=True)
    with LEDGER.open("a", encoding="utf-8", newline="") as handle:
        csv.writer(handle, delimiter="\t", lineterminator="\n").writerow(values)


def launch_pilot(branch: str, wait: bool, requested_job_name: str | None = None) -> None:
    preflight = require_fresh_preflight()
    branch_info = BRANCHES[branch]
    estimate = projected_cost()
    if estimate > 5.0:
        raise RuntimeError(f"Pilot budget gate rejected projected cost ${estimate:.2f}")
    boto_session, retry = session()
    sm = boto_session.client("sagemaker", config=retry)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    job_name = requested_job_name or f"ber-{branch}-pilot-{timestamp}"
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9-]{0,62}", job_name):
        raise ValueError(f"Invalid SageMaker job name: {job_name!r}")
    code_uri = f"s3://{BUCKET}/{PROJECT_PREFIX}/code/{branch}/{branch_info['sha']}/"
    data_uri = f"s3://{BUCKET}/{PROJECT_PREFIX}/shared/data/{DATASET_ID}/"
    output_uri = f"s3://{BUCKET}/{PROJECT_PREFIX}/runs/{job_name}/"
    gate = {
        "job_name": job_name,
        "branch": branch,
        "branch_sha": branch_info["sha"],
        "instance_type": INSTANCE_TYPE,
        "instance_count": 1,
        "max_runtime_seconds": MAX_RUNTIME_SECONDS,
        "hourly_usd": HOURLY_USD,
        "buffered_projected_usd": estimate,
        "necessity": "Representative train-only branch pilot before full-scale comparison",
        "cumulative_budget_cap_usd": 100.0,
        "test_data_uploaded": False,
        "role_arn": ROLE_ARN,
        "image_uri": IMAGE_URI,
        "preflight": preflight,
    }
    print("COST_GATE " + json.dumps(gate, sort_keys=True), flush=True)
    append_ledger(
        [
            datetime.now(timezone.utc).isoformat(), "PLANNED", job_name, branch, "pilot", "native-pipeline",
            INSTANCE_TYPE, f"{MAX_RUNTIME_SECONDS / 3600:.4f}", f"{MAX_RUNTIME_SECONDS / 3600:.4f}",
            f"{HOURLY_USD:.4f}", f"{estimate:.4f}", "0.0000", f"{100.0 - estimate:.4f}",
            "train-only representative pilot", output_uri,
        ]
    )
    try:
        response = sm.create_processing_job(
            ProcessingJobName=job_name,
            RoleArn=ROLE_ARN,
        AppSpecification={
            "ImageUri": IMAGE_URI,
            "ContainerEntrypoint": ["python3", "/opt/ml/processing/input/code/sagemaker_worker.py"],
            "ContainerArguments": [
                "--branch-root", "/opt/ml/processing/input/code/business_entity_resolution",
                "--data-root", "/opt/ml/processing/input/data",
                "--branch-name", branch,
                "--branch-sha", str(branch_info["sha"]),
                "--pilot-rows", str(PILOT_ROWS),
            ],
        },
        ProcessingInputs=[
            {
                "InputName": "code",
                "S3Input": {
                    "S3Uri": code_uri,
                    "LocalPath": "/opt/ml/processing/input/code",
                    "S3DataType": "S3Prefix",
                    "S3InputMode": "File",
                    "S3DataDistributionType": "FullyReplicated",
                },
            },
            {
                "InputName": "train-data",
                "S3Input": {
                    "S3Uri": data_uri,
                    "LocalPath": "/opt/ml/processing/input/data",
                    "S3DataType": "S3Prefix",
                    "S3InputMode": "File",
                    "S3DataDistributionType": "FullyReplicated",
                },
            },
        ],
        ProcessingOutputConfig={
            "Outputs": [
                {
                    "OutputName": "pilot-output",
                    "S3Output": {
                        "S3Uri": output_uri,
                        "LocalPath": "/opt/ml/processing/output",
                        "S3UploadMode": "Continuous",
                    },
                }
            ]
        },
        ProcessingResources={
            "ClusterConfig": {
                "InstanceCount": 1,
                "InstanceType": INSTANCE_TYPE,
                "VolumeSizeInGB": 200,
            }
        },
        StoppingCondition={"MaxRuntimeInSeconds": MAX_RUNTIME_SECONDS},
        Environment={"AWS_SDK_UA_APP_ID": "AWSSkill-SageMaker", "PYTHONUNBUFFERED": "1"},
            Tags=[
                {"Key": "Project", "Value": "AmazonML2026-BER-Comparison"},
                {"Key": "Branch", "Value": branch},
                {"Key": "Phase", "Value": "Pilot"},
            ],
        )
    except ClientError as exc:
        reason = str(exc).replace("\t", " ").replace("\n", " ")
        append_ledger(
            [
                datetime.now(timezone.utc).isoformat(), "BLOCKED", job_name, branch, "pilot",
                "create-processing-job", INSTANCE_TYPE, "0.0000",
                f"{MAX_RUNTIME_SECONDS / 3600:.4f}", f"{HOURLY_USD:.4f}", f"{estimate:.4f}",
                "0.0000", "100.0000", reason, output_uri,
            ]
        )
        raise
    print(json.dumps({"job_name": job_name, "arn": response["ProcessingJobArn"], "output": output_uri}), flush=True)
    if wait:
        wait_for_job(sm, job_name)


def wait_for_job(sm, job_name: str) -> None:
    while True:
        result = sm.describe_processing_job(ProcessingJobName=job_name)
        status = result["ProcessingJobStatus"]
        print(json.dumps({"job_name": job_name, "status": status, "failure_reason": result.get("FailureReason")}), flush=True)
        if status in {"Completed", "Failed", "Stopped"}:
            started = result.get("ProcessingStartTime")
            ended = result.get("ProcessingEndTime")
            elapsed = (ended - started).total_seconds() if started and ended else 0.0
            actual = round(elapsed / 3600 * HOURLY_USD, 4)
            append_ledger(
                [
                    datetime.now(timezone.utc).isoformat(), status.upper(), job_name, "", "pilot",
                    "native-pipeline", INSTANCE_TYPE, f"{elapsed / 3600:.4f}",
                    f"{MAX_RUNTIME_SECONDS / 3600:.4f}", f"{HOURLY_USD:.4f}",
                    f"{projected_cost():.4f}", f"{actual:.4f}", f"{100.0 - actual:.4f}",
                    result.get("FailureReason", ""), "",
                ]
            )
            if status != "Completed":
                raise RuntimeError(f"{job_name} ended {status}: {result.get('FailureReason')}")
            return
        time.sleep(30)


def status(job_name: str, wait: bool) -> None:
    boto_session, retry = session()
    sm = boto_session.client("sagemaker", config=retry)
    if wait:
        wait_for_job(sm, job_name)
    else:
        result = sm.describe_processing_job(ProcessingJobName=job_name)
        print(json.dumps(result, default=str, indent=2))


def main() -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("upload-train")
    upload_code_parser = sub.add_parser("upload-code")
    upload_code_parser.add_argument("branch", choices=sorted(BRANCHES))
    launch = sub.add_parser("launch-pilot")
    launch.add_argument("branch", choices=sorted(BRANCHES))
    launch.add_argument("--job-name")
    launch.add_argument("--wait", action="store_true")
    check = sub.add_parser("status")
    check.add_argument("job_name")
    check.add_argument("--wait", action="store_true")
    args = parser.parse_args()
    if args.command == "upload-train":
        upload_train()
    elif args.command == "upload-code":
        upload_code(args.branch)
    elif args.command == "launch-pilot":
        launch_pilot(args.branch, args.wait, args.job_name)
    elif args.command == "status":
        status(args.job_name, args.wait)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
