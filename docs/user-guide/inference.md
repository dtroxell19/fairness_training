# Inference

The fairness_training package provides two inference regimes to handle different deployment scenarios. This guide explains when to use each and how they work.

---

## The Two Regimes

| Regime | Batch Size | Guarantee | Use Case |
|--------|-----------|-----------|----------|
| **Large-Batch** | ≥ b_tau | Per-batch fairness | Batch processing, offline inference |
| **Small-Batch** | < b_tau | Aggregate fairness over time | Real-time, streaming |

**Use Large-Batch When:**

- You're doing batch processing (not real-time)
- You can control batch composition
- You need per-batch and aggregate fairness **guarantees**

**Use Small-Batch When:**

- Real-time/streaming inference
- Can't control when requests arrive
- Fairness over time (i.e. after many, many inference inputs seen) is sufficient

---

## Large-Batch Inference

When your inference batch size is ≥ `b_tau` (default 64), fairness_training enforces **hard per-batch constraints**.

**Requirements**

- Batch size ≥ `b_tau`
- Both groups must be present in each batch, and batch composition (i.e. group membership ratios) must be equal

!!! important "Critical Requirement"
    Similar to training, the fairness_training package assumes you can use stratified sampling in your inference dataset when operating in the large-batch regime. This leads to exact fairness guarantees.

---

## Small-Batch Inference (Primal-Dual Algorithm)

For real-time inference where you can't control batch sizes, fairness_training uses an **online primal-dual algorithm** that guarantees aggregate fairness over time

**Key insight**: Individual batches may violate constraints, but the weighted average violation over time converges to satisfy the constraint.

### Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `eta_0` | 0.5 | Initial dual step size. Increase → more strict enforcement |
| `b_tau` | 64 | Threshold between large-batch (hard constraints) and small-batch (primal-dual) regimes |

---

## State Management

**Always reset before a new inference sequence**:

```python
model.reset_inference_state()
```

This resets:
- `lambda_dual` → 0
- `dual_update_count` → 0
- `cumulative_samples` → 0
- `cumulative_weighted_violation` → 0
- `lambda_max` → 0

**When to Reset:**

- Before evaluating on a new test set
- At the start of each day/session for a production system
- When you want to measure fairness over a specific time window

**When NOT to Reset:**

- During a continuous inference session where you want aggregate guarantees
- In the middle of processing a batch sequence

---

## Debugging Inference Issues

**Common Issues**

**Fairness gap exceeds tolerance in aggregate stats for small-batch regime**:

- Likely need more samples for convergence
- Check if group proportions are skewed
- Try increasing `eta_0`

**Fairness gap exceeds tolerance in aggregate stats for large-batch regime**:

- Check if stratified sampling is used
- Try drastically increasing the width of prediction bounds
- Optimization solvers often operate with some slight numerical error. Try slightly decreasing epsilon to account for this

**Solver failures during inference**:

- Ensure both groups present in batch
- Check for NaN/Inf in inputs
- Relax `fairness_tolerance` if too tight

### Logging for Production

```python
import logging

logger = logging.getLogger('fairness_training')

def inference_with_logging(model, X, y=None):
    predictions = model(X, y=y, inference=True)
    
    # Log batch-level metrics
    logger.info(f"Batch size: {len(X)}")
    logger.info(f"Lambda: {model.lambda_dual:.4f}")
    logger.info(f"Cumulative samples: {model.cumulative_samples}")
    
    return predictions
```

---

## Next Steps

- **[Examples: Small-Batch Inference](../examples/small-batch.md)**: Complete example
- **[API Reference: FairModel](../api/fair-model.md)**: Full API documentation
