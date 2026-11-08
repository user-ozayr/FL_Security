# Securing Federated Learning: MTD + two-stage additive SMPC

Research prototype for the master's thesis on horizontal federated learning,
Moving Target Defence (MTD), poisoning attacks, and client-side additive
secret sharing for secure aggregation.

## Quick start: end-to-end local simulation

Use Python 3.11 and install the Flower simulation extra:

    python -m pip install -e ".[simulation,dev]"
    pytest -q
    env OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python run_experiment.py --config configs/benchmark_lr001/baseline.yaml
    env OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python run_experiment.py --config configs/benchmark_lr001/hybrid_verified.yaml --local-aggregators

The local command starts one separate aggregation-party process per
smpc.num_aggregation_parties, runs the Flower simulation, then shuts the
party processes down. CIFAR-10 is downloaded by torchvision into ./data;
metrics are written to ./results.

The experiment server is pinned to the legacy Flower 1.38.x simulation API
(start_simulation plus a custom Strategy). It is not yet migrated to the
ClientApp/ServerApp project model.

## Training stability and reproduction

With seed 42, five CIFAR-10 clients, one local epoch, and batch size 32, the
historical learning rate of 0.1 caused client 1 to produce non-finite local
weights. Flower reported two results and one client failure; Ray had dispatched
the clients successfully. The same one-round setup at 0.01 completed all three
selected clients. The `client_fn(cid)` deprecation warning is separate from
this numerical failure.

The original experiment configs remain unchanged. Use the configs under
`configs/benchmark_lr001/` for the stabilized benchmark. The clean baseline is
`configs/benchmark_lr001/baseline.yaml`; the hybrid run is
`configs/benchmark_lr001/hybrid_verified.yaml`.

The local party processes share one host and kernel with the coordinator and
clients. The local helper places the per-client token map in the shared
simulation worker environment so credentials can be routed by client ID; this
is a wiring/demo convenience, not credential isolation. A compromised local
worker or coordinator can inspect that map. This demonstrates the complete
message flow only, not protection against a malicious local host, coordinator,
or simulated client. It is not a multi-host trust-boundary deployment.

## External aggregation-party deployment

Run each party as its own process, preferably on a separate host or a
separately administered container. Each party needs:
- FL_SMPC_CLIENT_KEYS_JSON: a client-ID-to-token map for client uploads to this party.
- FL_SMPC_CONTROL_TOKEN: a unique coordinator-to-party control secret.
- FL_SMPC_PEER_ENDPOINTS_JSON: URLs for every other party.
- FL_SMPC_PEER_TOKENS_JSON: symmetric pairwise HMAC secrets shared with every other party.

Example for party 0 in a four-party deployment:

    export FL_SMPC_CLIENT_KEYS_JSON='{"0":"CLIENT0_PARTY0_SECRET","1":"CLIENT1_PARTY0_SECRET"}'
    export FL_SMPC_CONTROL_TOKEN='PARTY0_COORDINATOR_SECRET'
    export FL_SMPC_PEER_ENDPOINTS_JSON='{"1":"https://party1.example.org","2":"https://party2.example.org","3":"https://party3.example.org"}'
    export FL_SMPC_PEER_TOKENS_JSON='{"1":"LINK_0_1","2":"LINK_0_2","3":"LINK_0_3"}'
    python -m fl_security.smpc.aggregator_server --party-id 0 --party-count 4 --host 0.0.0.0 --port 8700

Repeat for parties 1, 2 and 3. Each pairwise token must be provisioned to
exactly the two parties in that pair (for example, the token named LINK_0_1
must be the same on parties 0 and 1, and unavailable to parties 2 and 3).
Every party must know the URL of all other parties. The coordinator's
FL_SMPC_AGGREGATORS_JSON contains each party URL and that party's unique
control token.

Each real client runtime must receive only its own
FL_SMPC_CLIENT_TOKENS_JSON map from party ID to that client's write token.
Provision the credentials through a secret manager or deployment platform;
never commit them to Git, put them in YAML, or send them in Flower round
configuration. The local helper is not a suitable credential-isolation
mechanism for real multi-tenant clients.

The aggregation-party service implements HTTP routes but does not terminate
TLS itself. For external network deployment, put each service behind HTTPS
with TLS/mTLS (or an equivalently trusted service mesh), restrict ingress,
and use HTTPS URLs. The built-in service does not implement account
provisioning, persistent storage, or audit logging.

## MTD randomness

For reproducible thesis runs, leave FL_MTD_SECRET unset: MTD selection and
committee assignments are deterministic from the experiment seed and round.
This mode is intended for repeatable simulation only; a public experiment seed
does not provide unpredictability against an adversary who knows it.

For a live deployment, set a randomly generated secret in the Flower server
environment only:

    export FL_MTD_SECRET="$(python -c 'import secrets; print(secrets.token_urlsafe(32))')"

The controller derives a round-scoped HMAC-SHA256 counter stream from this
secret and the experiment seed. Rejection sampling avoids modulo bias when
selecting clients and assigning committee positions. The keyed stream is
unpredictable to parties that do not know FL_MTD_SECRET. The secret is not sent
in Flower client configuration or to aggregation parties. Keep it in the
server's secret manager, rotate it deliberately, and do not publish it with
experimental configs. Record whether secret-seeded MTD was enabled, but never
record the secret itself.

## Data flow

1. Flower selects clients using the round-scoped MTD controller.
2. The coordinator sends the current global model, the selected client's
   committee IDs, and the relevant party URLs. It does not send client
   authentication secrets in Flower's fit configuration.
3. Each client trains locally and applies configured label flipping or
   model-update poisoning.
4. In SMPC mode the client sample-weights and fixed-point encodes its
   floating-point model tensors, creates additive shares over the prime field
   Z_q using OS-backed cryptographic randomness, and POSTs one share directly
   to each party in its assigned committee.
5. The Flower client returns an empty parameter payload and its local example
   count. The coordinator binds each FitRes to the registered Flower proxy ID
   and rejects a missing, duplicate, unregistered, unselected, or mismatched
   client-reported logical ID. It verifies the reported count against the
   server-known simulation partition and uses the server-known count for
   weighting. No individual update arrays reach the Flower coordinator.
6. Each aggregation party authenticates client uploads and sums the shares
   addressed to it for the successful participant set. It does not return
   this first-stage partial sum to the coordinator.
7. Each party creates fresh additive shares of its first-stage partial sum
   and sends one re-share directly to every other party. Pairwise party HMAC
   keys authenticate these second-stage messages.
8. Each party sums the second-stage shares received from all source parties
   and returns one random share of the global aggregate. The coordinator
   combines only these second-stage shares, decodes the fixed-point result,
   divides by total sample count, and broadcasts the new global model.
9. After reconstruction the coordinator acknowledges every party; services
   clear client shares, re-share buffers, and cached output shares.

In non-SMPC configurations, Flower returns model arrays to the coordinator
and ordinary sample-weighted FedAvg is used. The server verifies each FitRes
logical client ID against its registered Flower proxy ID before accepting the
result. This is an identity-consistency check within the Flower runtime, not a
standalone cryptographic client-authentication mechanism; external deployments
must still authenticate and protect their Flower transport and client runtime.

## Output-file and CSV safety

Experiment result filenames are restricted to a single file component within
the configured output directory; path separators, traversal components and
symbolic-link output targets are rejected. CSV exports neutralize formula-like
text cells (including values beginning with `=`, `+`, `-` or `@`) so
spreadsheet applications treat them as text. Numeric values remain numeric.
Benchmark CSVs and experiment logs use the same sanitizer.

The legacy root-level `audit_artifacts.py`, `generate_audit_artifacts.py`,
and `generate_reports.py` scripts are not part of the current pipeline;
`run_benchmark.py` is the supported report-generation path.

## Security model and limitations

The intended confidentiality assumptions are: fewer than all parties in a
client's committee collude; fewer than all aggregation parties collude in the
second-stage re-sharing; transport channels are authenticated and confidential;
client credentials remain secret; and the coordinator follows the protocol.
The coordinator sees client identities and participation metadata, the global
model, the final aggregate, and second-stage random shares. It does not see
first-stage committee partial sums, individual client shares, or raw client
model arrays in SMPC mode. A coalition that obtains every first-stage share
for a client can reconstruct that client's weighted update.

The coordinator is assumed honest-but-curious. A malicious coordinator that
sends inconsistent models, selects overlapping cohorts to difference out
participants, or deviates from the aggregation protocol is not covered. A
small-cohort global aggregate can itself reveal information, especially when
an observer already knows other participants' updates; the configured
minimum cohort reduces trivial single-client releases but is not a formal
differential-privacy guarantee.

This is a research prototype, not a cryptographic security proof or a
production MPC implementation. In particular:

- The local demo starts separate processes on one host; a host/root/kernel
  compromise can read all processes. Use separate administration domains for
  a meaningful cross-party trust boundary.
- The protocol is semi-honest. It does not provide malicious-secure MPC,
  verifiable computation, zero-knowledge range proofs, or Byzantine
  aggregation-party resistance.
- Party control tokens are held by the coordinator. Pairwise party keys are
  shared only by the two parties in that link. Client runtimes must receive
  only their own per-party write tokens. Provisioning and rotation remain
  deployment responsibilities.
- Share transport uses authenticated HMAC requests. Network confidentiality
  requires HTTPS/mTLS or a trusted secure service mesh; the service itself
  binds plain HTTP.
- Party round state is held in memory. Acknowledgements clear state after
  successful reconstruction, but restart recovery, persistent storage,
  garbage collection after crashed rounds, and availability/fault tolerance
  require further work.
- No Shamir threshold recovery or dropout reconstruction is implemented.
  A party outage or too few successful client uploads aborts the secure round.
  The second sharing stage is not a substitute for threshold recovery.
- There are no range proofs. Honest clients must remain within the configured
  centred-field range. A malicious client can submit arbitrary updates; the
  protocol protects confidentiality, not integrity.
- MTD changes participation and committee assignment. It is not a
  Byzantine-robust aggregation algorithm and does not neutralize poisoning.
- Differential privacy, backdoor triggers and attack-success rate, membership
  inference, property inference, and gradient inversion are outside the
  current implementation and must not be claimed as mitigated.

See docs/threat-model.md for the assumptions, data-flow boundary, implemented
protections, and out-of-scope threats in a thesis-oriented structure.

## Multi-seed benchmark and statistical comparison

Run baseline, MTD-only, additive-sharing-only and hybrid conditions with
identical seed lists:

    python run_benchmark.py \
      --configs configs/baseline_sign_flip.yaml configs/mtd_sign_flip.yaml \
                 configs/smpc_sign_flip.yaml configs/hybrid.yaml \
      --baseline baseline_sign_flip --seeds 5

The runner writes one CSV per condition/seed under results/benchmark/runs,
then produces:
- all_rounds.csv: every round from every seed and condition.
- summary_by_round.csv: per-condition/per-round mean, sample standard deviation,
  and two-sided 95% Student-t confidence intervals for up to 31 seeds (a
  normal approximation is used for larger samples).
- paired_accuracy_differences.csv: per-seed paired differences in global test
  accuracy and loss against the chosen baseline.
- condition_metadata.csv: attack, defence and seed metadata.

Use at least five seeds for an initial thesis comparison and more when compute
allows. A one-seed run deliberately reports no confidence interval. These
summaries quantify clean global test accuracy and overhead; they are not
backdoor attack-success rates. Do not report ASR or inference-resistance
claims from these runs.

## Reproducible evaluation

Compare FedAvg, MTD-only, additive-SMPC-only, and hybrid configurations while
holding the dataset, partitioning, model, local epochs, optimizer, client
fraction, malicious-client population, and paired random seeds constant.
Report mean, standard deviation, confidence intervals, clean test accuracy,
malicious-client participation, paired accuracy/loss differences, aggregation
time, party count, and application share payload bytes. The byte count excludes
HTTP/TLS framing and transport retries; report it as application payload, not
total network traffic.

## Poisoning detection and robust aggregation (plaintext validation mode)

The optional robust-update module is a first integrity layer for research
experiments. It computes each completed client's update relative to the
current global model and scores four signals: MAD-normalized distance from
the coordinate-wise median, cosine alignment with the current cohort median
direction, update norm relative to the cohort median norm, and temporal
contribution consistency measured by cosine similarity with an exponential
moving average of that client's previously accepted update directions.
Historical profiles are updated only from non-flagged clients after a
successful screened-FedAvg round. Suspicious updates can be excluded before
sample-weighted FedAvg. If screening would leave fewer than the configured
minimum survivors, the implementation falls back to coordinate-wise median
aggregation and does not update client history. The experiment logs detected,
accepted and flagged client IDs, individual score details, and detection
true-positive/false-positive/false-negative counts against simulator ground
truth.

Enable it with a `defences.robust_aggregation` section in a configuration:

```yaml
defences:
  robust_aggregation:
    enabled: true
    max_malicious: 2
    residual_z_threshold: 3.5
    min_cosine_similarity: 0.0
    max_norm_ratio: 5.0
    min_survivors: 3
    history_decay: 0.8
    min_history_rounds: 2
    history_cosine_threshold: -0.25
```

### Tuned detector validation (held-out seeds)

The original screening thresholds (`residual_z_threshold: 3.5`,
`min_cosine_similarity: 0.0`) detected all simulated sign-flip updates in
the initial five-seed experiment, but the clean-data control exposed 4 false
positives among 75 benign client-round updates (seeds 42-46). This motivated
a candidate threshold adjustment; it is not evidence that the original
thresholds generalize without false positives.

The tuned configurations use:

```yaml
robust_aggregation:
  enabled: true
  max_malicious: 2
  residual_z_threshold: 20.0
  min_cosine_similarity: 0.1
  max_norm_ratio: 5.0
  min_survivors: 3
  history_decay: 0.8
  min_history_rounds: 2
  history_cosine_threshold: -0.25
```

These settings were evaluated on a separate seed set (47-51):

- **Clean-data control:** 0 false positives among 75 benign updates; all 75
  updates were accepted. Final accuracy and loss matched the FedAvg control
  in each of the five paired seeds (mean final accuracy 63.896%).
- **Sign-flip poisoning:** 30 true positives, 0 false positives and 0 false
  negatives across 15 rounds (five clients per round, two simulated attackers
  per round). Observed precision and recall were both 100%; no fallback
  aggregation was used.
- **Model performance under sign-flip poisoning:** mean final accuracy was
  62.850% with screening versus 46.362% for FedAvg, a paired improvement of
  16.488 percentage points. Mean paired final-loss difference (screened minus
  FedAvg) was -0.75540.

Reproduction configurations are:
- `fedavg_no_attack.yaml` and `robust_no_attack.yaml`: clean-data controls
  with the original screening thresholds.
- `robust_no_attack_tuned.yaml`: tuned clean-data control.
- `baseline_sign_flip.yaml` and `robust_sign_flip.yaml`: original
  sign-flip comparison.
- `robust_sign_flip_tuned.yaml`: tuned sign-flip detector.

These are controlled simulation results on IID CIFAR-10 with five clients,
full participation, three rounds, and a 40% sign-flip attacker fraction.
The thresholds were calibrated after inspecting earlier runs (seeds 42-46)
and evaluated on seeds 47-51. The results therefore provide a held-out
seed check within the same setup, not a broad generalization guarantee.
They do not establish performance for non-IID clients, other poisoning
strategies, different attacker proportions, or real deployments. The
detection labels are available from simulator ground truth; they would not
be available in an ordinary deployment.

**Important security boundary:** this screening currently requires individual
plaintext model updates at the Flower coordinator. It is therefore explicitly
incompatible with the current additive sum-only SMPC backend; the runner
rejects configurations that enable both. This is deliberate. Ordinary
additive sharing supports sums, not secure pairwise distances, medians, or
comparisons. The robust detector is a heuristic and is not a formal Byzantine
guarantee. It can flag benign non-IID updates and may miss coordinated or
adaptive poisoning. The next research step is to implement robust scoring and
selection inside a malicious-secure MPC protocol, or to introduce a separately
trusted and independently administered validation boundary with an explicit
trust assumption.
