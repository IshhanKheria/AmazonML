from __future__ import annotations

import importlib.util
from pathlib import Path
import unittest

from botocore.exceptions import ClientError


MODULE_PATH = Path(__file__).with_name("aws_preflight.py")
SPEC = importlib.util.spec_from_file_location("aws_preflight", MODULE_PATH)
assert SPEC and SPEC.loader
PREFLIGHT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PREFLIGHT)


class FakeIam:
    def __init__(self, result=None, error=None):
        self.result = result
        self.error = error

    def simulate_principal_policy(self, **kwargs):
        if self.error:
            raise self.error
        return {"EvaluationResults": [self.result]}


def access_denied() -> ClientError:
    return ClientError(
        {
            "Error": {"Code": "AccessDenied", "Message": "simulator unavailable"},
            "ResponseMetadata": {"HTTPStatusCode": 403, "RequestId": "request-1"},
        },
        "SimulatePrincipalPolicy",
    )


class PassRoleClassificationTests(unittest.TestCase):
    def test_allowed_is_confirmed(self):
        iam = FakeIam({"EvalActionName": "iam:PassRole", "EvalResourceName": "role", "EvalDecision": "allowed"})
        result = PREFLIGHT.verify_pass_role(iam)
        self.assertEqual(PREFLIGHT.PASSROLE_CONFIRMED, result["status"])

    def test_implicit_deny_is_denied(self):
        iam = FakeIam(
            {"EvalActionName": "iam:PassRole", "EvalResourceName": "role", "EvalDecision": "implicitDeny"}
        )
        result = PREFLIGHT.verify_pass_role(iam)
        self.assertEqual(PREFLIGHT.PASSROLE_DENIED, result["status"])

    def test_simulator_access_denied_is_unverifiable(self):
        result = PREFLIGHT.verify_pass_role(FakeIam(error=access_denied()))
        self.assertEqual(PREFLIGHT.PASSROLE_UNVERIFIABLE, result["status"])
        self.assertEqual("AccessDenied", result["error"]["code"])

    def test_missing_context_is_unverifiable(self):
        iam = FakeIam(
            {
                "EvalActionName": "iam:PassRole",
                "EvalResourceName": "role",
                "EvalDecision": "allowed",
                "MissingContextValues": ["iam:PassedToService"],
            }
        )
        result = PREFLIGHT.verify_pass_role(iam)
        self.assertEqual(PREFLIGHT.PASSROLE_UNVERIFIABLE, result["status"])

    def test_permissions_boundary_deny_is_denied(self):
        iam = FakeIam(
            {
                "EvalActionName": "iam:PassRole",
                "EvalResourceName": "role",
                "EvalDecision": "allowed",
                "PermissionsBoundaryDecisionDetail": {"AllowedByPermissionsBoundary": False},
            }
        )
        result = PREFLIGHT.verify_pass_role(iam)
        self.assertEqual(PREFLIGHT.PASSROLE_DENIED, result["status"])

    def test_caller_sagemaker_controls_require_add_tags(self):
        iam = FakeIam(
            {
                "EvalActionName": "sagemaker:AddTags",
                "EvalResourceName": "processing-job",
                "EvalDecision": "implicitDeny",
            }
        )
        result = PREFLIGHT.verify_caller_sagemaker_controls(iam, simulator_available=True)
        self.assertEqual(PREFLIGHT.CHECK_DENIED, result["status"])


if __name__ == "__main__":
    unittest.main()
