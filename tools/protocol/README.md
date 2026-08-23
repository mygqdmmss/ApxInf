# Protocol Gate Probe

This directory contains the dependency-free protocol gate probe for the
standalone stub and the production evaluator endpoint. It intentionally uses
only Python's standard library and does not modify the frozen evaluator.

Start the stub in one terminal:

```bash
cargo run --locked --bin apxinf_protocol_stub -- --bind 127.0.0.1:18001
```

Run the gates in another terminal:

```bash
python3 tools/protocol/run_protocol_gates.py \
  --base-url http://127.0.0.1:18001 \
  --output docs/collaboration/records/M2-P0-protocol-evidence.json
```

The probe reads `max_model_len` from `/health`, records the exact canonical
request body, HTTP status, response, elapsed time, and source commit, then
writes a sibling `.sha256` file for the evidence JSON. Structured negative
controls all explicitly set `stream=false`.
