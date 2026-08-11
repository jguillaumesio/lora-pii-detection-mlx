## Results

| Metric | Base, zero-shot | Base, few-shot (6) | LoRA fine-tuned |
|---|---|---|---|
| Accuracy | 66% | 66% | 95% |
| Precision | 0.639 | 0.610 | 0.926 |
| Recall | 0.686 | 0.840 | 0.974 |
| F1 | 0.662 | 0.707 | 0.950 |
| False positives (hard negatives) | 75 | 104 | 15 |
| Missed PII lines | 61 | 31 | 5 |
| Valid JSON | 100% | 100% | 100% |
| Seconds per line | 0.83 | 1.67 | 0.91 |

## Per-type F1

| PII type | Base, zero-shot | Base, few-shot (6) | LoRA fine-tuned |
|---|---|---|---|
| email | 0.745 | 0.782 | 1.000 |
| phone | 0.628 | 0.575 | 0.983 |
| name | 0.531 | 0.663 | 0.855 |
| iban | 0.358 | 0.194 | 0.950 |
| address | 0.500 | 0.597 | 0.914 |
| dob | 0.383 | 0.366 | 0.875 |

## Training

- Validation loss: 3.537 at iter 1, best 0.491 at iter 500, last 0.491 at iter 500
- Peak memory: 5.994 GB
- Final train loss: 0.475
- Adapter size: 42.0 MB

## Out-of-distribution test (30 hand-written lines)

| Metric | Base, zero-shot | Base, few-shot (6) | LoRA fine-tuned |
|---|---|---|---|
| Accuracy | 80% | 90% | 100% |
| Precision | 0.857 | 0.933 | 1.000 |
| Recall | 0.750 | 0.875 | 1.000 |
| F1 | 0.800 | 0.903 | 1.000 |
| False positives | 2 | 1 | 0 |
| Missed PII lines | 4 | 2 | 0 |
| Seconds per line | 0.66 | 1.01 | 0.77 |
