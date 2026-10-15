# Klein-Code-Bench integration

Dataset: https://huggingface.co/datasets/klein-ml/klein-code-bench

Evaluator release: `072246b6c125b3de01eb5a7740b89326438eb46e`

The `klein` extra installs the evaluator wheel from that immutable Hugging Face commit, with a SHA-256 integrity fragment. Python 3.11+ is required. No model or dataset is downloaded at import time. The `datasets` extra is separately needed for remote task loading; local JSONL loading works offline.

```bash
pip install -e '.[klein,datasets,models]'
banana run klein-code-bench --dataset-config easy --allow-downloads \
  --revision 072246b6c125b3de01eb5a7740b89326438eb46e --backend native \
  --model openai/your-local-model --model-options local-model.json \
  --output jobs/klein-easy
```

`local-model.json` can contain `{"api_base":"http://127.0.0.1:8000/v1","api_key":"local","temperature":0}`.

For direct Transformers inference, use `--model-runtime transformers --model /path/to/model` and the existing Transformers generation options.

Use `--dataset-config default` for all 260 tasks. Each of easy/intermediate/hard/max has exactly 65 tasks. To use local data, replace remote-loading arguments with `--source /path/to/klein-code-bench/klein_code_bench/data/easy.jsonl`.

The coding agent starts in an empty Git workspace. Only its prompt and examples are public-to-agent; acceptance scenarios remain in `Task.grading`. The generated multi-file project is captured as a patch, replayed into a fresh native verifier workspace, and evaluated by the benchmark's parent-side process runner. Max-tier sessions test persisted state by force-killing and restarting child processes. The runner does not provide a security sandbox.

`klein_acceptance.json` holds per-scenario results; `summary.json` includes tier scores with denominators 65 and overall denominator 260. The top-level harness pass rate continues to describe the selected job, so use the dedicated `klein_code_bench` summary for benchmark-wide reporting.

Local integration checks include public/private separation, remote configurations and revision forwarding, malformed rows, optional evaluator errors, no-container enforcement, empty-project failure, a generated multi-file project, patch replay, resume behavior, and an actual max-tier reference project. Reference tests validate the evaluator and are not model scores.
