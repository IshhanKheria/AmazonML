# IAM preflight design review — 2026-09-25

## Finding

The original preflight incorrectly treated access to
`iam:SimulatePrincipalPolicy` as if it were the `iam:PassRole` runtime
permission itself. `iam:PassRole` is not a standalone API operation. AWS checks
it when a caller supplies a role ARN to another service operation such as
`CreateProcessingJob`.

The policy simulator is an administrative, non-compute verification mechanism.
Failure to call it means PassRole is **unverifiable**, not denied.

## Corrected states

The preflight now emits exactly one of:

- `PASSROLE_CONFIRMED`: the simulator evaluated `iam:PassRole` as allowed for
  the exact execution-role ARN with
  `iam:PassedToService=sagemaker.amazonaws.com`, with no missing context or
  denying boundary/Organizations detail.
- `PASSROLE_DENIED`: the simulator returned an implicit/explicit deny, or a
  permissions-boundary/Organizations denial.
- `PASSROLE_UNVERIFIABLE`: the simulator API could not be called or required
  context was missing.

Both `PASSROLE_DENIED` and `PASSROLE_UNVERIFIABLE` keep
`paid_execution_allowed=false`.

Policy-name listing is now optional and informational. The default preflight no
longer calls or requires `iam:ListAttachedRolePolicies` or
`iam:ListRolePolicies`, because lists of policy names do not establish effective
access.

## Actual runtime permission — do not weaken

The caller still requires this scoped permission:

```json
{
  "Effect": "Allow",
  "Action": "iam:PassRole",
  "Resource": "arn:aws:iam::303936065943:role/service-role/AmazonSageMakerAdminIAMExecutionRole",
  "Condition": {
    "StringEquals": {
      "iam:PassedToService": "sagemaker.amazonaws.com"
    }
  }
}
```

## Minimum permissions for Codex-side non-compute verification

Grant the caller the following read-only verification permissions. These do not
grant PassRole and do not allow a SageMaker job to be created:

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Sid": "VerifyOnlyTheseRoles",
      "Effect": "Allow",
      "Action": "iam:SimulatePrincipalPolicy",
      "Resource": [
        "arn:aws:iam::303936065943:role/AmazonML-Codex-Role",
        "arn:aws:iam::303936065943:role/service-role/AmazonSageMakerAdminIAMExecutionRole"
      ]
    },
    {
      "Sid": "InspectSageMakerExecutionRoleTrust",
      "Effect": "Allow",
      "Action": "iam:GetRole",
      "Resource": "arn:aws:iam::303936065943:role/service-role/AmazonSageMakerAdminIAMExecutionRole"
    }
  ]
}
```

`iam:SimulatePrincipalPolicy` is needed for two principals: the caller role for
the PassRole decision, and the execution role for S3 input/output decisions.
`iam:GetRole` is needed to verify that the execution role trusts
`sagemaker.amazonaws.com`.

## Administrator-run alternative

If the Codex role should not receive simulator access, an administrator can run
the following non-compute checks and provide the complete JSON responses:

```powershell
aws iam simulate-principal-policy `
  --policy-source-arn arn:aws:iam::303936065943:role/AmazonML-Codex-Role `
  --action-names iam:PassRole `
  --resource-arns arn:aws:iam::303936065943:role/service-role/AmazonSageMakerAdminIAMExecutionRole `
  --context-entries ContextKeyName=iam:PassedToService,ContextKeyValues=sagemaker.amazonaws.com,ContextKeyType=string

aws iam get-role `
  --role-name AmazonSageMakerAdminIAMExecutionRole

aws iam simulate-principal-policy `
  --policy-source-arn arn:aws:iam::303936065943:role/service-role/AmazonSageMakerAdminIAMExecutionRole `
  --action-names s3:ListBucket `
  --resource-arns arn:aws:s3:::amazon-sagemaker-303936065943-eu-north-1-4g5ie80zojahhv

aws iam simulate-principal-policy `
  --policy-source-arn arn:aws:iam::303936065943:role/service-role/AmazonSageMakerAdminIAMExecutionRole `
  --action-names s3:GetObject `
  --resource-arns arn:aws:s3:::amazon-sagemaker-303936065943-eu-north-1-4g5ie80zojahhv/ber-comparison/shared/data/70bc1d8a16c667e0155c2105d0ab2ebe41d7e7a85d8a529e3ca81c6c3a5af037/train/train_source1.tsv

aws iam simulate-principal-policy `
  --policy-source-arn arn:aws:iam::303936065943:role/service-role/AmazonSageMakerAdminIAMExecutionRole `
  --action-names s3:PutObject `
  --resource-arns arn:aws:s3:::amazon-sagemaker-303936065943-eu-north-1-4g5ie80zojahhv/ber-comparison/runs/preflight-output
```

The PassRole result must be `allowed` for the exact role and service context;
the trust policy must contain `sagemaker.amazonaws.com`; and all three S3
results must be `allowed`.

## Why there is no smaller end-to-end probe

AWS has no standalone PassRole API and SageMaker `CreateProcessingJob` has no
dry-run option. A real `CreateProcessingJob` request is the end-to-end
authorization check, but if allowed it can provision paid compute. Deliberately
invalid job parameters are not a safe substitute because AWS does not contract
the order of validation and authorization checks.

Therefore, while job creation is prohibited, the minimum safe options are:

1. grant the two read-only verification permissions above; or
2. have an administrator run the exact checks and provide their raw results.

Until one option succeeds, paid execution remains blocked.
