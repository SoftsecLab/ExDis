# ExDis: Modular training and evaluation
## Files

| Module | Responsibility |
| --- | --- |
| `exdis/deps.py` | Dependencies, local paths and numerical configuration |
| `exdis/data.py` | Paired JSON data loading, train/validation/test checks |
| `exdis/models.py` | Tokenizer, causal model and device utilities |
| `exdis/scoring.py` | Token log probability, LRP and DeltaLRP |
| `exdis/rewriting.py` | T5 sentinel inputs, generated fills and reconstruction |
| `exdis/network.py` | Scoring model, paraphraser and joint optimization logic |
| `exdis/evaluation.py` | Validation and two-pass final LRP evaluation |
| `exdis/training.py` | Training loop, validation AUROC and in-memory best adapter |
| `exdis/pipeline.py` | Full experiment orchestration and persistent outputs |
| `run.py` | Single-command entry point |
