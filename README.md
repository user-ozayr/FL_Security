# Securing Federated Learning: MTD + client-side additive SMPC

Research prototype for the master's thesis on horizontal federated learning
with Moving Target Defence (MTD), poisoning attacks, and additive secret
sharing for secure aggregation.

## Quick start: end-to-end local simulation

Use Python 3.11 and install the Flower simulation extra:

    python -m pip install -e ".[simulation,dev]"
    pytest -q
    python run_experiment.py --config configs/hybrid.yaml --local-aggregators

The local command starts one separate aggregation-party process per
smpc.num_aggregation_parties, runs the Flower simulation, then shuts the
party processes down. CIFAR-10 is downloaded by torchvision into ./data;
metrics are written to ./results.

The local party processes share one host and kernel with the coordinator and
clients. This demonstrates the complete client-to-party-to-coordinator data
flow, but it is not a multi-host trust-boundary deployment.

## External aggregation-party deployment

Run each party as its own process (preferably on a separate host or a
separately administered container). Example for party 0:

    export FL_SMPC_CLIENT_KEYS_JSON='{"0":"CLIENT0_PARTY0_SECRET","1":"CLIENT1_PARTY0_SECRET"}'
    export FL_SMPC_CONTROL_TOKEN='PARTY0_COORDINATOR_SECRET'
    python -m fl_security.smpc.aggregator_server --party-id 0 --host 0.0.0.0 --port 8700

Repeat with a unique client-key map and control token for every party. Provision
client secrets out-of-band; do not commit them to Git or put them in YAML.

Configure the Flower coordinator's environment variable
FL_SMPC_AGGREGATORS_JSON as a JSON object keyed by party ID:

    {
      "0": {"url": "https://party0.example.org", "control_token": "PARTY0_COORDINATOR_SECRET"},
      "1": {"url": "https://party1.example.org", "control_token": "PARTY1_COORDINATOR_SECRET"},
      "2": {"url": "https://party2.example.org", "control_token": "PARTY2_COORDINATOR_SECRET"}
    }

Each client runtime must receive its own FL_SMPC_CLIENT_TOKENS_JSON mapping
from party ID to that client's token. In local simulation the helper supplies
this mapping separately to simulated client workers. For a real deployment,
provision only the individual client's credentials through a secret manager
or the deployment platform. Never send client credentials inside Flower
round configuration.

The aggregation-party service uses HTTP for its application protocol. For
network deployment, put it behind TLS/mTLS (or a service mesh with equivalent
mutual authentication), restrict ingress to the appropriate clients/coordinator,
and use HTTPS URLs. The built-in service intentionally does not implement TLS,
account provisioning, persistent storage, or audit logging.

## Data flow

1. Flower selects clients using the round-scoped MTD controller.
2. The server sends the current global model plus the selected client's
   committee IDs and aggregator URLs. It does not send client authentication
   secrets in Flower's fit configuration.
3. Each client trains locally and applies configured label flipping or
   model-update poisoning.
4. In SMPC mode the client sample-weights and fixed-point encodes its
   floating-point model tensors, creates additive shares over the prime field
   Z_q using OS-backed cryptographic randomness, and POSTs one share directly
   to each party in its MTD committee.
5. The Flower client returns an empty parameter payload and the local example
   count. The Flower coordinator therefore receives no individual update
   arrays in SMPC mode.
6. Each aggregation party verifies the client HMAC, sums shares only for
   clients in the successful participant set, clears individual shares after
   finalization, and returns its partial sum.
7. The coordinator combines the partial sums, decodes the fixed-point values,
   divides by the total sample count, and broadcasts the new global model.

In non-SMPC configurations, Flower returns model arrays to the coordinator
and ordinary sample-weighted FedAvg is used.

## Security model and limitations

The intended additive-sharing confidentiality assumption is that fewer than
all parties in a client's assigned committee collude, transport channels are
authenticated and confidential, client credentials are not compromised, and
the coordinator follows the aggregation protocol. The coordinator sees each
party's partial sum and the final aggregate, but not the client shares or raw
client model update arrays. A colluding set that obtains every share for one
client can reconstruct that client's weighted update.

This is a research prototype, not a cryptographic security proof or a
production MPC implementation. In particular:

- The local demo starts separate processes on one host; a host/root/kernel
  compromise can read all processes. Use separate administration domains for
  a meaningful cross-party trust boundary.
- The protocol is semi-honest at the aggregator/coordinator layer. It does not
  provide malicious-secure MPC, verifiable computation, zero-knowledge range
  proofs, or protection from a malicious coordinator choosing inconsistent
  training models.
- The coordinator holds aggregator control tokens. Each client holds only its
  own per-party write tokens in a real deployment. Credential provisioning and
  rotation remain deployment responsibilities.
- Share transport uses authenticated HMAC requests. Network confidentiality
  requires HTTPS/mTLS or a trusted secure service mesh; the service itself
  binds plain HTTP.
- Party round state is held in memory. Restart recovery, persistent storage,
  round garbage collection, and availability/fault tolerance require further
  work.
- No Shamir threshold recovery or dropout reconstruction is implemented.
  A client whose upload to any assigned party fails is excluded from the
  Flower successful-result set; each party finalizes only for successful
  clients. A party outage fails the round rather than recovering.
- There are no range proofs. Clients encode signed fixed-point values modulo q;
  honest clients must remain within the configured centred-field range.
- Differential privacy, backdoor triggers, membership inference, gradient
  inversion, and Byzantine-robust aggregation are outside the current
  implementation and must not be claimed as mitigated.

## Reproducible evaluation

Compare FedAvg, MTD-only, additive-SMPC-only, and hybrid configurations while
holding the dataset, partitioning, model, local epochs, optimizer, client
fraction, malicious-client population, and random seeds constant. Run multiple
seeds and report mean, standard deviation, confidence intervals, clean
accuracy, attack-specific metrics, aggregation time, party count, and actual
share payload bytes. The byte count excludes HTTP/TLS framing and transport
retries; report it as application payload, not total network traffic.
