# m1 pilot launch blocker — 2026-09-25

## Outcome

The corrected IAM preflight passed for the exact execution role:

`arn:aws:iam::303936065943:role/service-role/AmazonSageMakerAdminIAMExecutionRole`

The m1 pilot submission was then rejected atomically before provisioning. A
read-only `ListProcessingJobs` query found zero jobs matching the requested
name. No job ARN exists and SageMaker compute cost is `$0.00`.

## Requested pilot

- Job name: `ber-m1-pilot-20260925-1605-iamfix`
- Job type: SageMaker ProcessingJob
- Instance: one `ml.m6i.2xlarge`
- Runtime hard limit: 7,200 seconds
- Price: `$0.49/hour`
- Raw maximum compute estimate: `$0.98`
- Buffered cost gate: `$1.225`
- Output prefix:
  `s3://amazon-sagemaker-303936065943-eu-north-1-4g5ie80zojahhv/ber-comparison/runs/ber-m1-pilot-20260925-1605-iamfix/`

## Exact AWS error

```text
AccessDeniedException when calling CreateProcessingJob:
User arn:aws:sts::303936065943:assumed-role/AmazonML-Codex-Role/AmazonMLCodex
is not authorized to perform sagemaker:AddTags on resource
arn:aws:sagemaker:eu-north-1:303936065943:processing-job/ber-m1-pilot-20260925-1605-iamfix
because no identity-based policy allows the sagemaker:AddTags action.
```

## Required AWS-side change

The launcher supplies `Project`, `Branch`, and `Phase` tags for audit and cost
governance. Those tags will not be removed to bypass IAM. Grant:

```json
{
  "Effect": "Allow",
  "Action": "sagemaker:AddTags",
  "Resource": "arn:aws:sagemaker:eu-north-1:303936065943:processing-job/ber-m1-pilot-*"
}
```

The preflight has been extended to verify `CreateProcessingJob`, `AddTags`,
`DescribeProcessingJob`, and `StopProcessingJob`. The launcher now refuses any
retry unless all four controls are confirmed in a fresh preflight report.
