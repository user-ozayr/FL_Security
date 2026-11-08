# Threat Model and Security Claims

## 1. System boundary

The prototype implements horizontal federated learning with a coordinating
Flower server, simulated clients, round-scoped Moving Target Defence (MTD), and
separately running aggregation-party services.

The secure aggregation data path has two additive-sharing stages:

1. Each client sample-weights and fixed-point encodes its model parameters,
   maps them into the prime field Z_q, creates random additive shares, and
   sends one share only to each aggregation party in its assigned committee.
2. Each aggregation party sums the client shares addressed to it. It then
   re-shares that first-stage partial sum with fresh randomness across every
   aggregation party. The first-stage partial sums are never returned to the
   coordinator.
3. Each party sums the second-stage shares it receives and returns one fresh
   share of the global aggregate. The coordinator combines only these
   second-stage shares, decodes the fixed-point aggregate, and divides by the
   total number of client examples.

The secure-mode Flower FitRes contains an empty parameter list and the local
example count. The coordinator checks that the reported count agrees with the
server-known client partition size and uses the server-known value for
aggregation weights.

## 2. Adversaries and assumptions

The intended confidentiality analysis assumes:

- **Honest-but-curious coordinator.** The coordinator follows client
  selection, model distribution, committee assignment, party finalization,
  and aggregation as implemented. It observes client identities, participation
  and cohort sizes, the current global model, the final aggregate, and the
  second-stage random shares. It does not deliberately send inconsistent
  models to clients, manipulate participant cohorts to isolate a client, or
  issue overlapping queries intended to difference out a participant.
- **Aggregation-party non-collusion.** Fewer than all parties collude in the
  second-stage re-sharing. For an individual client's first-stage update
  shares, a coalition must also obtain every share in that client's committee
  to reconstruct the update. Collusion of all relevant parties is outside the
  confidentiality guarantee.
- **Authenticated client uploads.** Each client has an independent per-party
  HMAC write token. Aggregator control tokens are held by the coordinator;
  pairwise party-to-party HMAC keys are provisioned to the two parties in that
  pair. Secrets are provisioned out-of-band and are never committed to Git.
- **Confidential transport.** External deployments use HTTPS with TLS/mTLS or
  an equivalently trusted service mesh. The built-in server binds plain HTTP
  and does not itself terminate TLS.
- **Bounded fixed-point arithmetic.** Weighted model values stay within the
  centred finite-field range. The system has no zero-knowledge range proofs
  to enforce this against an arbitrary malicious binary.
- **Client partition and weight integrity.** In the simulation, the
  coordinator knows each client's partition size and rejects mismatched
  client-reported example counts. A real deployment would need an independently
  verifiable weighting policy and must not trust a self-reported sample count.

The coordinator receives all second-stage shares because it must reconstruct
the global aggregate. The sum of those shares is the intended output and may
itself leak information in very small cohorts or where an observer already
knows all but one participant's update.

## 3. Implemented protections

### 3.1 Individual-update confidentiality in SMPC mode

Client-side shares are created before the Flower FitRes boundary. The
coordinator does not receive individual client parameter arrays. Each party
authenticates client uploads, sums only shares addressed to it, and sends its
first-stage partial sum only as fresh shares to the other parties. The
coordinator cannot inspect the first-stage committee partial sums or combine
their overlapping client-membership equations. It receives only the second
stage shares of the global aggregate.

Party services reject conflicting duplicate client uploads and conflicting
party re-shares. They cache outgoing re-shares so a retry uses identical
messages. After finalization, the coordinator acknowledges the result and
party services clear client shares, first-stage outgoing re-shares,
second-stage received shares, and cached output shares.

### 3.2 Moving Target Defence

The MTD controller varies selected clients and per-client aggregation
committees each round. Committee allocations are balanced so every configured
party receives contributions from multiple selected clients. A secure round
is aborted if client dropouts leave a party without the required minimum
number of client shares or if too few clients complete.

For reproducible thesis runs, MTD schedules derive from experiment.seed.
This is deterministic and should not be described as unpredictable to an
adversary who knows the seed. A live run can set FL_MTD_SECRET in the Flower
coordinator environment; the controller derives its private schedule from
that server-only secret and the experiment seed. It is not sent to clients or
aggregation parties and is removed before simulated Ray workers start.

### 3.3 Poisoning attacks

The simulator executes label flipping during local training and sign-flip or
random-noise perturbations on client updates. These attacks exercise the
configured training and aggregation path. Additive sharing protects the
confidentiality of client updates under its assumptions; it does not detect,
filter, or neutralize poisoning. MTD changes participation and party routing;
it is not a Byzantine-robust aggregation algorithm.

## 4. Explicitly out of scope

- Malicious-secure MPC, verifiable computation, zero-knowledge range proofs,
  or Byzantine aggregation-party resistance.
- A malicious coordinator that sends inconsistent models, manipulates client
  cohorts, or deviates from the protocol. The MTD root secret prevents clients
  from predicting future schedules when deployed correctly; it does not
  prevent a malicious coordinator from choosing a harmful schedule.
- Collusion of all aggregation parties in the second-stage re-sharing, or a
  coalition that obtains every first-stage share in a client's committee.
- Shamir threshold recovery, reconstruction after aggregator dropout, or
  availability guarantees. A required party outage aborts the secure round.
- Persistent storage and restart recovery for in-memory party state.
- Differential privacy, backdoor triggers and attack-success rate, membership
  inference, property inference, gradient inversion, or other inference attacks.
- Traffic-analysis and timing side channels, compromised client operating
  systems, credential theft, and production-grade key rotation.

This prototype is not a production secure aggregation service or a formal
cryptographic security proof. Claims must remain limited to the semi-honest
assumptions, attacks, datasets, client partitions, and seeds actually tested.

## 5. Local simulation limitation

The local demo starts each aggregation party in a separate process, but all
processes share one host and kernel. The local Ray simulation also uses a
shared worker environment containing per-client test tokens so the synthetic
workers can route requests by client ID; a compromised local worker can inspect
that map. The local demo therefore verifies the end-to-end message flow and
service interactions, but is not a malicious-host or malicious-client
isolation test.

For a meaningful multi-party trust boundary, deploy parties in separate
administrative domains, use TLS/mTLS, provision pairwise party keys only to
the intended pair, and provision each client's own party tokens only into that
client's isolated runtime.


## 6. Robust poisoning screening status

The current codebase contains an optional plaintext update-screening mode:
median-residual and cosine-direction anomaly indicators, update-norm checks,
and filtering before sample-weighted aggregation, with a coordinate-wise
median fallback. It is not enabled by the legacy benchmark configurations.
It is not available together with the current SMPC implementation, because
that backend exposes only aggregate sums and does not securely evaluate
comparisons or anomaly scores. This is a documented integrity/privacy gap,
not a protection that should be claimed for the hybrid configuration.

A future privacy-preserving poisoning defence must compute the relevant
robust statistics and client-selection decisions inside an MPC protocol,
rather than exporting individual updates to the coordinator.
