"""Complete, non-compute AWS preflight for the BER SageMaker pilot."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
import json
import os
from pathlib import Path
import sys
from typing import Any

import boto3
from botocore.config import Config
from botocore.exceptions import ClientError


PROFILE = "amazon-ml-codex"
REGION = "eu-north-1"
ACCOUNT = "303936065943"
CALLER_ROLE_ARN = f"arn:aws:iam::{ACCOUNT}:role/AmazonML-Codex-Role"
EXECUTION_ROLE_ARN = f"arn:aws:iam::{ACCOUNT}:role/service-role/AmazonSageMakerAdminIAMExecutionRole"
EXECUTION_ROLE_NAME = "AmazonSageMakerAdminIAMExecutionRole"
BUCKET = "amazon-sagemaker-303936065943-eu-north-1-4g5ie80zojahhv"
DATASET_ID = "70bc1d8a16c667e0155c2105d0ab2ebe41d7e7a85d8a529e3ca81c6c3a5af037"
DATA_PREFIX = f"ber-comparison/shared/data/{DATASET_ID}/"
OUTPUT_PREFIX = "ber-comparison/runs/"
PILOT_JOB_ARN = f"arn:aws:sagemaker:{REGION}:{ACCOUNT}:processing-job/ber-m1-pilot-preflight"
EXPECTED_OBJECTS = {
    f"{DATA_PREFIX}train/train_source1.tsv": 210069713,
    f"{DATA_PREFIX}train/train_source2.tsv": 489301488,
    f"{DATA_PREFIX}train/train_source3.tsv": 503705637,
    f"{DATA_PREFIX}train/train_ground_truth.tsv": 127015583,
}

PASSROLE_CONFIRMED = "PASSROLE_CONFIRMED"
PASSROLE_DENIED = "PASSROLE_DENIED"
PASSROLE_UNVERIFIABLE = "PASSROLE_UNVERIFIABLE"
CHECK_CONFIRMED = "CONFIRMED"
CHECK_DENIED = "DENIED"
CHECK_UNVERIFIABLE = "UNVERIFIABLE"


def package_version(name: str) -> str | None:
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def trust_principals(document: dict[str, Any]) -> set[str]:
    principals: set[str] = set()
    statements = document.get("Statement", [])
    if isinstance(statements, dict):
        statements = [statements]
    for statement in statements:
        principal = statement.get("Principal", {})
        services = principal.get("Service", []) if isinstance(principal, dict) else []
        if isinstance(services, str):
            services = [services]
        principals.update(str(service) for service in services)
    return principals


def simulate(iam, principal_arn: str, action: str, resource_arn: str, context=None) -> dict[str, Any]:
    request = {
        "PolicySourceArn": principal_arn,
        "ActionNames": [action],
        "ResourceArns": [resource_arn],
    }
    if context:
        request["ContextEntries"] = context
    response = iam.simulate_principal_policy(**request)
    evaluation = response["EvaluationResults"][0]
    return {
        "action": action,
        "resource": resource_arn,
        "decision": evaluation["EvalDecision"],
        "missing_context": evaluation.get("MissingContextValues", []),
        "organizations_decision": evaluation.get("OrganizationsDecisionDetail"),
        "permissions_boundary_decision": evaluation.get("PermissionsBoundaryDecisionDetail"),
    }


def client_error_detail(exc: ClientError) -> dict[str, Any]:
    return {
        "code": exc.response.get("Error", {}).get("Code"),
        "message": exc.response.get("Error", {}).get("Message"),
        "operation": exc.operation_name,
        "request_id": exc.response.get("ResponseMetadata", {}).get("RequestId"),
        "http_status": exc.response.get("ResponseMetadata", {}).get("HTTPStatusCode"),
        "full": str(exc),
    }


def simulation_is_fully_allowed(result: dict[str, Any]) -> bool:
    organizations = result.get("organizations_decision") or {}
    boundary = result.get("permissions_boundary_decision") or {}
    return (
        result.get("decision") == "allowed"
        and not result.get("missing_context")
        and organizations.get("AllowedByOrganizations", True) is not False
        and boundary.get("AllowedByPermissionsBoundary", True) is not False
    )


def verify_pass_role(iam) -> dict[str, Any]:
    """Classify PassRole without confusing simulator access with PassRole access."""
    context = [
        {
            "ContextKeyName": "iam:PassedToService",
            "ContextKeyValues": ["sagemaker.amazonaws.com"],
            "ContextKeyType": "string",
        }
    ]
    try:
        result = simulate(iam, CALLER_ROLE_ARN, "iam:PassRole", EXECUTION_ROLE_ARN, context=context)
    except ClientError as exc:
        return {
            "status": PASSROLE_UNVERIFIABLE,
            "verification_method": "iam:SimulatePrincipalPolicy",
            "reason": "The administrative simulator API could not be called; this is not a PassRole denial.",
            "error": client_error_detail(exc),
        }
    if result.get("decision") != "allowed":
        status = PASSROLE_DENIED
    elif result.get("missing_context"):
        status = PASSROLE_UNVERIFIABLE
    else:
        organizations = result.get("organizations_decision") or {}
        boundary = result.get("permissions_boundary_decision") or {}
        denied_by_guardrail = (
            organizations.get("AllowedByOrganizations", True) is False
            or boundary.get("AllowedByPermissionsBoundary", True) is False
        )
        status = PASSROLE_DENIED if denied_by_guardrail else PASSROLE_CONFIRMED
    return {
        "status": status,
        "verification_method": "iam:SimulatePrincipalPolicy",
        "principal_arn": CALLER_ROLE_ARN,
        "target_role_arn": EXECUTION_ROLE_ARN,
        "passed_to_service": "sagemaker.amazonaws.com",
        "simulation": result,
    }


def verify_execution_role(iam) -> dict[str, Any]:
    """Verify the execution-role trust policy; policy inventory is informational."""
    try:
        role = iam.get_role(RoleName=EXECUTION_ROLE_NAME)["Role"]
    except ClientError as exc:
        return {"status": CHECK_UNVERIFIABLE, "error": client_error_detail(exc)}
    principals = trust_principals(role["AssumeRolePolicyDocument"])
    status = CHECK_CONFIRMED if "sagemaker.amazonaws.com" in principals else CHECK_DENIED
    return {
        "status": status,
        "arn": role["Arn"],
        "role_id": role["RoleId"],
        "path": role["Path"],
        "max_session_duration": role["MaxSessionDuration"],
        "service_trust_principals": sorted(principals),
        "required_service_principal": "sagemaker.amazonaws.com",
    }


def inspect_execution_role_policy_inventory(iam) -> dict[str, Any]:
    """Collect human-readable policy names without treating them as effective-access proof."""
    inventory: dict[str, Any] = {"purpose": "informational_only"}
    calls = {
        "attached_policies": lambda: iam.list_attached_role_policies(RoleName=EXECUTION_ROLE_NAME).get(
            "AttachedPolicies", []
        ),
        "inline_policy_names": lambda: iam.list_role_policies(RoleName=EXECUTION_ROLE_NAME).get(
            "PolicyNames", []
        ),
    }
    for name, call in calls.items():
        try:
            inventory[name] = call()
        except ClientError as exc:
            inventory[f"{name}_error"] = client_error_detail(exc)
    return inventory


def verify_execution_role_s3(iam, simulator_available: bool) -> dict[str, Any]:
    """Verify execution-role S3 permissions when the administrative simulator is available."""
    if not simulator_available:
        return {
            "status": CHECK_UNVERIFIABLE,
            "reason": "iam:SimulatePrincipalPolicy is unavailable; no runtime S3 claim is inferred.",
        }
    bucket_arn = f"arn:aws:s3:::{BUCKET}"
    checks = [
        ("s3:ListBucket", bucket_arn),
        ("s3:GetObject", f"{bucket_arn}/{next(iter(EXPECTED_OBJECTS))}"),
        ("s3:PutObject", f"{bucket_arn}/{OUTPUT_PREFIX}preflight-output"),
    ]
    results = []
    try:
        for action, resource in checks:
            results.append(simulate(iam, EXECUTION_ROLE_ARN, action, resource))
    except ClientError as exc:
        return {"status": CHECK_UNVERIFIABLE, "error": client_error_detail(exc), "results": results}
    status = CHECK_CONFIRMED if all(simulation_is_fully_allowed(item) for item in results) else CHECK_DENIED
    return {"status": status, "simulations": results}


def verify_caller_sagemaker_controls(iam, simulator_available: bool) -> dict[str, Any]:
    """Verify every SageMaker action used by launch, monitoring, tagging, and emergency stop."""
    if not simulator_available:
        return {
            "status": CHECK_UNVERIFIABLE,
            "reason": "iam:SimulatePrincipalPolicy is unavailable; SageMaker controls cannot be verified.",
        }
    actions = [
        "sagemaker:CreateProcessingJob",
        "sagemaker:AddTags",
        "sagemaker:DescribeProcessingJob",
        "sagemaker:StopProcessingJob",
    ]
    results = []
    try:
        for action in actions:
            results.append(simulate(iam, CALLER_ROLE_ARN, action, PILOT_JOB_ARN))
    except ClientError as exc:
        return {"status": CHECK_UNVERIFIABLE, "error": client_error_detail(exc), "results": results}
    status = CHECK_CONFIRMED if all(simulation_is_fully_allowed(item) for item in results) else CHECK_DENIED
    return {"status": status, "prospective_job_arn": PILOT_JOB_ARN, "simulations": results}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument(
        "--inspect-policy-inventory",
        action="store_true",
        help="Optionally list policy names; this is informational and not required for the paid-execution gate.",
    )
    args = parser.parse_args()
    args.report.parent.mkdir(parents=True, exist_ok=True)
    os.environ["AWS_SDK_UA_APP_ID"] = "AWSSkill-SageMaker"

    report: dict[str, Any] = {
        "started_at": datetime.now(timezone.utc).isoformat(),
        "profile": PROFILE,
        "required_region": REGION,
        "execution_role_arn": EXECUTION_ROLE_ARN,
        "pass_role_states": [PASSROLE_CONFIRMED, PASSROLE_DENIED, PASSROLE_UNVERIFIABLE],
        "verification_contract": {
            "pass_role_is_runtime_permission_check": True,
            "simulate_principal_policy_is_administrative_verification_only": True,
            "paid_execution_requires": [
                PASSROLE_CONFIRMED,
                "EXECUTION_ROLE_TRUST_CONFIRMED",
                "EXECUTION_ROLE_S3_CONFIRMED",
                "CALLER_SAGEMAKER_CONTROLS_CONFIRMED",
            ],
        },
        "checks": {},
        "status": "IN_PROGRESS",
    }
    try:
        config = Config(
            retries={"total_max_attempts": 10, "mode": "adaptive"},
            connect_timeout=10,
            read_timeout=90,
            user_agent_appid="AWSSkill-SageMaker",
        )
        aws = boto3.Session(profile_name=PROFILE, region_name=REGION)
        clients = {
            service: aws.client(service, config=config)
            for service in ("sts", "iam", "s3", "sagemaker")
        }
        report["checks"]["sdk"] = {
            "boto3": package_version("boto3"),
            "botocore": package_version("botocore"),
            "sagemaker": package_version("sagemaker"),
            "sagemaker_core": package_version("sagemaker-core"),
        }
        report["checks"]["region"] = {
            "session_region": aws.region_name,
            "sagemaker_endpoint_region": clients["sagemaker"].meta.region_name,
        }
        if aws.region_name != REGION or clients["sagemaker"].meta.region_name != REGION:
            raise RuntimeError(f"Region mismatch: {report['checks']['region']}")

        identity = clients["sts"].get_caller_identity()
        report["checks"]["identity"] = {
            "account": identity["Account"],
            "arn": identity["Arn"],
            "user_id": identity["UserId"],
        }
        if identity["Account"] != ACCOUNT or ":assumed-role/AmazonML-Codex-Role/" not in identity["Arn"]:
            raise RuntimeError(f"Unexpected caller identity: {identity['Arn']}")

        sm = clients["sagemaker"]
        processing = sm.list_processing_jobs(MaxResults=10, SortBy="CreationTime", SortOrder="Descending")
        training = sm.list_training_jobs(MaxResults=10, SortBy="CreationTime", SortOrder="Descending")
        report["checks"]["sagemaker_list_access"] = {
            "processing_jobs_found": len(processing.get("ProcessingJobSummaries", [])),
            "training_jobs_found": len(training.get("TrainingJobSummaries", [])),
        }

        iam = clients["iam"]
        pass_role = verify_pass_role(iam)
        report["checks"]["pass_role"] = pass_role
        report["checks"]["execution_role_trust"] = verify_execution_role(iam)
        if args.inspect_policy_inventory:
            report["checks"]["execution_role_policy_inventory"] = inspect_execution_role_policy_inventory(iam)
        else:
            report["checks"]["execution_role_policy_inventory"] = {
                "status": "SKIPPED",
                "reason": "Policy-name listing is not proof of effective access and is not required by this gate.",
            }

        s3 = clients["s3"]
        location = s3.get_bucket_location(Bucket=BUCKET).get("LocationConstraint") or "us-east-1"
        s3.head_bucket(Bucket=BUCKET)
        listed = s3.list_objects_v2(Bucket=BUCKET, Prefix=DATA_PREFIX, MaxKeys=20)
        object_checks = []
        for key, expected_size in EXPECTED_OBJECTS.items():
            metadata = s3.head_object(Bucket=BUCKET, Key=key)
            actual_size = int(metadata["ContentLength"])
            if actual_size != expected_size:
                raise RuntimeError(f"S3 size mismatch for {key}: {actual_size} != {expected_size}")
            object_checks.append({"key": key, "bytes": actual_size, "etag": metadata.get("ETag")})
        report["checks"]["s3_actual"] = {
            "bucket": BUCKET,
            "bucket_region": location,
            "listed_objects": int(listed.get("KeyCount", 0)),
            "verified_objects": object_checks,
            "verified_total_bytes": sum(item["bytes"] for item in object_checks),
        }
        if location != REGION:
            raise RuntimeError(f"Bucket region is {location}, expected {REGION}")

        simulator_available = pass_role["status"] != PASSROLE_UNVERIFIABLE
        report["checks"]["execution_role_s3"] = verify_execution_role_s3(iam, simulator_available)
        report["checks"]["caller_sagemaker_controls"] = verify_caller_sagemaker_controls(
            iam, simulator_available
        )

        blockers = []
        if pass_role["status"] != PASSROLE_CONFIRMED:
            blockers.append(pass_role["status"])
        if report["checks"]["execution_role_trust"]["status"] != CHECK_CONFIRMED:
            blockers.append(f"EXECUTION_ROLE_TRUST_{report['checks']['execution_role_trust']['status']}")
        if report["checks"]["execution_role_s3"]["status"] != CHECK_CONFIRMED:
            blockers.append(f"EXECUTION_ROLE_S3_{report['checks']['execution_role_s3']['status']}")
        if report["checks"]["caller_sagemaker_controls"]["status"] != CHECK_CONFIRMED:
            blockers.append(
                f"CALLER_SAGEMAKER_CONTROLS_{report['checks']['caller_sagemaker_controls']['status']}"
            )
        report["blocking_reasons"] = blockers
        report["paid_execution_allowed"] = not blockers
        if blockers:
            report["required_admin_action"] = {
                "preferred": (
                    "Temporarily grant iam:SimulatePrincipalPolicy for the caller and execution-role ARNs, "
                    "and iam:GetRole for the execution role; then rerun this non-compute preflight."
                ),
                "alternative": (
                    "An administrator may run the equivalent simulations and trust-policy inspection and "
                    "provide the raw AWS JSON evidence. There is no standalone iam:PassRole API or SageMaker "
                    "dry-run API; CreateProcessingJob is an end-to-end check but can provision paid compute."
                ),
                "not_required": ["iam:ListAttachedRolePolicies", "iam:ListRolePolicies"],
            }
        report["status"] = "PASSED" if not blockers else "BLOCKED"
        report["completed_at"] = datetime.now(timezone.utc).isoformat()
        args.report.write_text(json.dumps(report, indent=2, sort_keys=True, default=str), encoding="utf-8")
        print(json.dumps(report, indent=2, sort_keys=True, default=str))
        return 0 if not blockers else 4
    except ClientError as exc:
        report["status"] = "FAILED"
        report["completed_at"] = datetime.now(timezone.utc).isoformat()
        report["aws_error"] = client_error_detail(exc)
        args.report.write_text(json.dumps(report, indent=2, sort_keys=True, default=str), encoding="utf-8")
        print(json.dumps(report, indent=2, sort_keys=True, default=str), file=sys.stderr)
        return 2
    except Exception as exc:
        report["status"] = "FAILED"
        report["completed_at"] = datetime.now(timezone.utc).isoformat()
        report["error"] = {"type": type(exc).__name__, "message": str(exc)}
        args.report.write_text(json.dumps(report, indent=2, sort_keys=True, default=str), encoding="utf-8")
        print(json.dumps(report, indent=2, sort_keys=True, default=str), file=sys.stderr)
        return 3


if __name__ == "__main__":
    raise SystemExit(main())
