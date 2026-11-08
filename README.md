# Securing Federated Learning: MTD + additive-sharing prototype

Research prototype for a master's thesis on horizontal federated learning,
Moving Target Defence (MTD), poisoning attacks and secure aggregation design.

## Run an experiment

Install the package and Flower simulation dependencies:

    python -m pip install -e ".[simulation]"
    python run_experiment.py --config configs/hybrid.yaml

CIFAR-10 is downloaded by torchvision into ./data. Results are written to
./results. Flower coordinates client training rounds; a custom server strategy
controls client selection, committee configuration, weighted FedAvg and
centralized evaluation.

## Implemented pipeline

- Flower simulation client/server workflow (legacy 1.38.x simulation API).
- Sample-count weighted FedAvg.
- Round-varying MTD client selection and per-client logical aggregation committee assignment.
- Fixed-point encoding and additive secret sharing over the prime field Z_q.
- OS-backed cryptographic randomness and unbiased field-element sampling.
- Committee-routed share addition and reconstruction of the aggregate sum.
- Sign-flip and random-noise model-update attacks, plus label flipping during
  local training before the loss is computed.
- CSV metrics and accuracy plots.

## Critical confidentiality limitation

**The current Flower path does not provide server-private secure aggregation.**
Flower client fit returns the raw model arrays in its FitRes; the Flower server
receives those clear arrays before the aggregation helper creates additive
shares. Therefore the server can already inspect each individual client update.
The additive-sharing helper demonstrates finite-field sharing arithmetic and
MTD committee routing only; it does not hide client updates from the Flower
coordinator. The smpc.enabled option currently activates this arithmetic
simulation, not a completed confidentiality guarantee.

The logical aggregation parties are also simulated in the coordinator process:
there is no process or network isolation, independent party key management,
malicious-secure protocol, Shamir threshold recovery, or dropout recovery.
This is not a production MPC deployment or a cryptographic security proof.
Do not report the current implementation as protecting client updates against
the coordinating server.

To meet the thesis confidentiality objective, the next architecture step is
to generate shares on each client before the Flower FitRes boundary, route
each share to a distinct independently isolated aggregation service, and
return only the combined aggregate to the Flower coordinator. A collusion and
dropout threat model, authenticated transport, party lifecycle, and recovery
protocol must be specified and tested as part of that extension.

## Other scope limitations

The current benchmark does not implement differential privacy, backdoor
triggers, membership inference, gradient inversion, or a Byzantine-robust
aggregation rule. Those threats require separate implementations and
attack-specific evaluations. Experimental claims must be restricted to the
configured attacks, partitions, data, seeds and simulation model.

## Reproducible evaluation

Keep the dataset, partitioning, model, local epochs, optimizer, client fraction,
attack population and random seeds fixed when comparing FedAvg, MTD-only,
sharing-arithmetic-only and hybrid conditions. Use multiple seeds and report
mean, standard deviation, confidence intervals, clean accuracy,
attack-specific metrics, aggregation time and share payload estimates. These
byte counts are logical estimates, not measured network transfers.
