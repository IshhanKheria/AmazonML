# AWS preflight checkpoint — 2026-09-25

## Outcome

The refreshed preflight did not pass. No SageMaker Processing or Training job
was submitted. SageMaker compute cost for this checkpoint is `$0.00`.

## Checks that passed

- AWS CLI: `2.37.3`
- Caller account: `303936065943`
- Caller session:
  `arn:aws:sts::303936065943:assumed-role/AmazonML-Codex-Role/AmazonMLCodex`
- Explicit boto3 profile: `amazon-ml-codex`
- Session and SageMaker endpoint region: `eu-north-1`
- `sagemaker:ListProcessingJobs`: allowed; zero jobs found
- `sagemaker:ListTrainingJobs`: allowed; zero jobs found
- S3 bucket region: `eu-north-1`
- S3 caller access: bucket head/list, all four input-object heads, and a scoped
  write/head/delete probe passed
- Verified train-only S3 input bytes: `1,330,092,421`
- No test object was read or uploaded

## Blocking IAM results

### PassRole verification

The required non-launching PassRole policy simulation could not run:

```text
AccessDenied when calling SimulatePrincipalPolicy:
User arn:aws:sts::303936065943:assumed-role/AmazonML-Codex-Role/AmazonMLCodex
is not authorized to perform iam:SimulatePrincipalPolicy on
arn:aws:iam::303936065943:role/AmazonML-Codex-Role because no identity-based
policy allows iam:SimulatePrincipalPolicy.
```

- HTTP status: `403`
- AWS request ID: `d949ee98-89b7-4645-8940-8fb176d00eb5`

This is not evidence that `iam:PassRole` itself is denied. It means PassRole
cannot be verified before launch as explicitly required.

### Execution-role configuration inspection

The caller cannot inspect the configured execution role:

- `iam:GetRole`: denied, request ID
  `dfdb28c0-820f-4f53-9e44-8f59d2c4e08e`
- `iam:ListAttachedRolePolicies`: denied, request ID
  `2ce7ac0a-b94f-40b5-910d-fa14a130a607`
- `iam:ListRolePolicies`: denied, request ID
  `62404cc4-73c0-480c-824a-41b91c5beb34`

Consequently the preflight cannot verify:

- that `AmazonSageMakerAdminIAMExecutionRole` trusts
  `sagemaker.amazonaws.com`;
- that its attached/inline policies allow reading the frozen S3 input prefix;
- that it can write the pilot output prefix.

## Required unblock

For the agent to perform the requested non-launching verification, allow the
caller to use:

- `iam:GetRole`
- `iam:ListAttachedRolePolicies`
- `iam:ListRolePolicies`
- `iam:SimulatePrincipalPolicy`

The simulation must cover both
`arn:aws:iam::303936065943:role/AmazonML-Codex-Role` and
`arn:aws:iam::303936065943:role/AmazonSageMakerAdminIAMExecutionRole`.

The caller must separately have `iam:PassRole` on the execution role, ideally
conditioned on `iam:PassedToService = sagemaker.amazonaws.com`.

## Resume point

Rerun `comparison/scripts/aws_preflight.py`. Only after it records `PASSED`
should the m1 pilot cost gate allocate a job name and call
`CreateProcessingJob`.

## Recheck at 2026-09-25 15:28 UTC

The complete preflight was rerun against fresh AWS state. SageMaker list
access, `eu-north-1`, and the S3 read/write/delete checks passed again. The IAM
checks remain blocked:

- `iam:SimulatePrincipalPolicy`: denied, request ID
  `bc75eb33-7a0f-4370-939d-749bc7b5e3a0`
- `iam:GetRole`: denied, request ID
  `1c316aef-fdc9-4fe4-aae8-8227000691cc`
- `iam:ListAttachedRolePolicies`: denied, request ID
  `773cad4b-f0d3-4adc-893e-1da434ffbecc`
- `iam:ListRolePolicies`: denied, request ID
  `dcb468d6-d21a-41f9-8003-8c57a0f750d6`

No job name was allocated and no `CreateProcessingJob` call was made.
