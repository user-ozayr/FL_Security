# Securing Federated Learning: MTD + two-stage additive SMPC

Research prototype for the master's thesis on horizontal federated learning,
Moving Target Defence (MTD), poisoning attacks, and client-side additive
secret sharing for secure aggregation.

## Quick start: end-to-end local simulation

Use Python 3.11 and install the Flower simulation extra:

    python -m pip install -e ".[simulation,dev]"
    pytest -q
    python run_experiment.py --config configs/hybrid.yaml --local-aggregators

The local command starts one separate aggregation-party process per
smpc.num_aggregation_parties, runs the Flower simulation, then shuts the
party processes down. CIFAR-10 is downloaded by torchvision into ./data;
metrics are written to ./results.

The experiment server is pinned to the legacy Flower 1.38.x simulation API
(start_simulation plus a custom Strategy). It is not yet migrated to the
ClientApp/ServerApp project model.

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
committee assignments are then derived from experiment.seed and the round.
This is deterministic and should not be described as unpredictable to an
adversary who knows the seed.

For a live deployment, set a randomly generated secret in the Flower server
environment only:

    export FL_MTD_SECRET="$(python -c 'import secrets; print(secrets.token_urlsafe(32))')"

The controller derives its private round schedule from this secret and the
configured experiment seed. The secret is not sent in Flower client
configuration or to aggregation parties. Keep it in the server's secret
manager, rotate it deliberately, and do not publish it with experimental
configs. Record whether secret-seeded MTD was enabled, but never record the
secret itself.

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
   count. The coordinator verifies that the reported count matches its
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
and ordinary sample-weighted FedAvg is used.

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
current global model, compares update vectors with the coordinate-wise median,
and reports median-residual, cosine-direction, and update-norm indicators.
Suspicious updates can be excluded before sample-weighted FedAvg. If screening
would leave fewer than the configured minimum survivors, the implementation
falls back to coordinate-wise median aggregation.

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
```

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
