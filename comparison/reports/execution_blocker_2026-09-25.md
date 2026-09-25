# Execution blocker — 2026-09-25

## Outcome

The approved train-only SageMaker pilot was submitted only after its cost gate
passed, but AWS rejected the request before provisioning. No Processing job was
created and no SageMaker compute charge was incurred.

## Failed request

- Job name: `ber-m1-pilot-20260925-141053`
- Region: `eu-north-1`
- Calling principal: `arn:aws:sts::303936065943:assumed-role/AmazonML-Codex-Role/AmazonMLCodex`
- Requested execution role: `arn:aws:iam::303936065943:role/AmazonSageMakerAdminIAMExecutionRole`
- Instance: one `ml.m6i.2xlarge`
- Hard runtime limit: 7,200 seconds
- Current AWS public-catalog rate: `$0.49/hour`
- Raw maximum compute estimate: `$0.98`
- Buffered maximum estimate: `$1.225`
- Actual compute cost: `$0.00`

## AWS evidence

`CreateProcessingJob` returned:

```text
AccessDeniedException: User:
arn:aws:sts::303936065943:assumed-role/AmazonML-Codex-Role/AmazonMLCodex
is not authorized to perform: iam:PassRole on resource:
arn:aws:iam::303936065943:role/AmazonSageMakerAdminIAMExecutionRole
because no identity-based policy allows the iam:PassRole action
```

A follow-up read-only `ListProcessingJobs` check was also denied because the
caller lacks `sagemaker:ListProcessingJobs`. The synchronous
`CreateProcessingJob` rejection occurred before an ARN was returned, which is
the controlling evidence that no job was provisioned.

## Minimal unblock

An AWS administrator must allow the calling role to pass the named SageMaker
execution role. A least-privilege identity-policy statement is:

```json
{
  "Effect": "Allow",
  "Action": "iam:PassRole",
  "Resource": "arn:aws:iam::303936065943:role/AmazonSageMakerAdminIAMExecutionRole",
  "Condition": {
    "StringEquals": {
      "iam:PassedToService": "sagemaker.amazonaws.com"
    }
  }
}
```

For monitoring, the caller also needs `sagemaker:DescribeProcessingJob`,
`sagemaker:ListProcessingJobs`, and `sagemaker:StopProcessingJob` scoped as
appropriate. The execution role's trust policy must allow
`sagemaker.amazonaws.com`; that policy could not be inspected because the
caller also lacks `iam:GetRole`.

## Preserved state

- Four hash-verified train files are in the designated S3 bucket under the
  immutable ground-truth-SHA prefix. No test file was uploaded.
- Both pinned branch packages and the isolated worker are in separate
  branch-SHA S3 prefixes.
- The local notebooks, common fold manifest, worker, launcher, and budget
  ledger are preserved in the neutral comparison worktree.
- Resume point: rerun the m1 pilot after the IAM change; do not proceed to
  hardening or full-scale jobs before m1 pilot evidence passes.
