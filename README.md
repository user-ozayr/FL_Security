# Securing Federated Learning: MTD + SMPC

Research prototype for a master's thesis on securing horizontal federated
learning with Moving Target Defence (MTD) and secure aggregation based on
additive secret sharing.

## Run an experiment

Install the package and Flower simulation dependencies:

    python -m pip install -e ".[simulation]"
    python run_experiment.py --config configs/hybrid.yaml

CIFAR-10 is downloaded by torchvision into ./data. Results are written to
./results. Flower coordinates client training and the custom server strategy
controls client sampling, aggregation committees, weighted FedAvg and
centralized evaluation.

## Implemented pipeline

- Flower simulation client/server workflow.
- Sample-count weighted FedAvg.
- Round-varying MTD client selection and per-client aggregation committee assignment.
- Fixed-point encoding and additive secret sharing over the finite field
  Z_q, using operating-system-backed cryptographic randomness for share generation.
- Per-party accumulation of shares, followed by reconstruction of the total
  aggregate only; individual client updates are not reconstructed by the
  aggregation helper.
- Sign-flip and random-noise model-update attacks, plus label flipping during
  the local training loss computation.
- CSV and accuracy-plot output.

## Security limitations

This is a simulation, not a production MPC deployment or a cryptographic
security proof. Logical aggregation parties are simulated in the Flower server
process: there is no network/process isolation between parties, no independent
key management, no malicious-secure protocol, no Shamir threshold recovery and
no dropout recovery. The server process can access the shares in memory, so
this implementation must not be described as production-grade confidentiality.
Its additive-sharing confidentiality claim assumes that not all parties in
each client's assigned committee collude.

The current benchmark does not implement differential privacy, backdoor
triggers, membership inference, gradient inversion, or a Byzantine-robust
aggregation rule. Those threats require separate implementations and
attack-specific evaluations. Experimental claims must be restricted to the
configured attacks, partitions, data, seeds and simulation model.

## Reproducible evaluation

Keep the dataset, partitioning, model, local epochs, optimizer, client fraction,
attack population and random seeds fixed when comparing FedAvg, MTD-only,
SMPC-only and hybrid conditions. Use multiple seeds and report mean,
standard deviation, confidence intervals, clean accuracy, attack-specific
metrics, aggregation time and estimated share bytes. The share-byte metric is
a logical payload estimate, not a measured network transfer.
