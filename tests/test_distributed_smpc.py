import secrets
import threading
import urllib.error
import urllib.request

import numpy as np
import pytest
import torch

from fl_security.federated.aggregation import reconstruct_weighted_average, weighted_fedavg
from fl_security.smpc.additive_sharing import DEFAULT_FIELD_MODULUS
from fl_security.smpc.aggregator_server import AggregatorState, _encode_arrays, create_server
from fl_security.smpc.client_sharing import (
    acknowledge_party,
    finalize_party,
    upload_client_update,
)
from fl_security.smpc.fixed_point import FixedPointEncoder


class PartyHarness:
    def __init__(self, client_keys, control_token):
        self.state = AggregatorState(client_keys, control_token)
        self.server = create_server("127.0.0.1", 0, self.state)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.url = f"http://127.0.0.1:{self.server.server_port}"

    def close(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)


def test_clients_send_shares_directly_and_coordinator_reconstructs_weighted_sum():
    client_ids, party_ids = [0, 1, 2], [0, 1, 2]
    client_tokens = {
        party: {str(cid): secrets.token_urlsafe(24) for cid in client_ids}
        for party in party_ids
    }
    control_tokens = {party: secrets.token_urlsafe(24) for party in party_ids}
    parties = [
        PartyHarness(client_tokens[party], control_tokens[party])
        for party in party_ids
    ]
    endpoints = {str(party): parties[party].url for party in party_ids}
    encoder = FixedPointEncoder(1_000_000)
    global_state = {"weight": torch.tensor([0.0, 0.0])}
    local_states = [
        {"weight": torch.tensor([1.25, -2.0])},
        {"weight": torch.tensor([4.0, 1.0])},
        {"weight": torch.tensor([-0.5, 2.0])},
    ]
    example_counts = [2, 6, 4]
    # Every committee contains two parties, and every party receives shares
    # from two clients so no party output is a single-client contribution.
    committees = {0: [0, 1], 1: [1, 2], 2: [0, 2]}
    round_id = "unittest-1"

    try:
        for client_id in client_ids:
            tokens = {
                str(party): client_tokens[party][str(client_id)]
                for party in committees[client_id]
            }
            payload_bytes = upload_client_update(
                state=local_states[client_id],
                num_examples=example_counts[client_id],
                client_id=client_id,
                round_id=round_id,
                committee_ids=committees[client_id],
                aggregator_endpoints=endpoints,
                client_tokens=tokens,
                encoder=encoder,
            )
            assert payload_bytes > 0

        party_partials, used_parties = [], []
        for party in party_ids:
            participants = [cid for cid in client_ids if party in committees[cid]]
            assert len(participants) >= 2
            party_partials.append(finalize_party(
                endpoints[str(party)], round_id, participants, control_tokens[party]
            ))
            used_parties.append(party)

        actual = reconstruct_weighted_average(
            party_partials, global_state, sum(example_counts), encoder
        )
        expected = weighted_fedavg(local_states, example_counts, global_state)
        torch.testing.assert_close(
            actual["weight"], expected["weight"], atol=1e-6, rtol=0
        )

        for party in used_parties:
            acknowledge_party(endpoints[str(party)], round_id, control_tokens[party])
            assert round_id not in parties[party].state.rounds
            assert round_id not in parties[party].state.finalized
    finally:
        for party in parties:
            party.close()


def test_party_rejects_invalid_client_authentication():
    party = PartyHarness({"4": secrets.token_urlsafe(24)}, secrets.token_urlsafe(24))
    try:
        body, keys_json = _encode_arrays({"weight": np.array([1], dtype=np.int64)})
        request = urllib.request.Request(
            party.url + "/v1/rounds/bad-auth/shares",
            data=body,
            headers={
                "Content-Type": "application/octet-stream",
                "X-Client-ID": "4",
                "X-Tensor-Keys": keys_json,
                "X-Signature": "incorrect",
            },
            method="POST",
        )
        with pytest.raises(urllib.error.HTTPError) as exc:
            urllib.request.urlopen(request, timeout=2)
        assert exc.value.code == 403
    finally:
        party.close()


def test_party_refuses_finalization_when_an_expected_client_share_is_missing():
    client_token, control_token = secrets.token_urlsafe(24), secrets.token_urlsafe(24)
    state = AggregatorState({"0": client_token}, control_token)
    body, keys_json = _encode_arrays({"weight": np.array([1], dtype=np.int64)})
    from fl_security.smpc.aggregator_server import _signature

    message = b"round-one\n0\n" + keys_json.encode() + b"\n" + body
    state.submit_share(
        "round-one", "0", body, keys_json, _signature(client_token, message)
    )
    canonical = '["0"]'
    signature = _signature(
        control_token, b"finalize\nround-one\n" + canonical.encode()
    )
    with pytest.raises(ValueError, match="at least two clients"):
        state.finalize("round-one", ["0"], signature)
