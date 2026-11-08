import secrets
import socket
import threading
import urllib.error
import urllib.request

import numpy as np
import pytest
import torch

from fl_security.federated.aggregation import reconstruct_weighted_average, weighted_fedavg
from fl_security.smpc.aggregator_server import AggregatorState, _encode_arrays, _signature, create_server
from fl_security.smpc.client_sharing import (
    acknowledge_party,
    finalize_party,
    retrieve_party_share,
    upload_client_update,
)
from fl_security.smpc.fixed_point import FixedPointEncoder


class PartyHarness:
    def __init__(self, party_id, party_count, client_keys, control_token,
                 peer_endpoints, peer_tokens):
        self.state = AggregatorState(
            client_keys,
            control_token,
            party_id=party_id,
            party_count=party_count,
            peer_endpoints=peer_endpoints,
            peer_tokens=peer_tokens,
        )
        self.server = create_server("127.0.0.1", 0, self.state)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.url = f"http://127.0.0.1:{self.server.server_port}"

    def close(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)


def _build_party_harnesses(client_ids, party_ids):
    party_count = len(party_ids)
    client_tokens = {
        party: {str(cid): secrets.token_urlsafe(24) for cid in client_ids}
        for party in party_ids
    }
    control_tokens = {party: secrets.token_urlsafe(24) for party in party_ids}
    pair_tokens = {
        (left, right): secrets.token_urlsafe(32)
        for left in party_ids
        for right in party_ids
        if left < right
    }

    # First bind all party sockets so every party can be configured with the
    # full peer directory before its service thread starts.
    servers, parties = [], []
    for party_id in party_ids:
        peer_auth = {
            str(peer): pair_tokens[(min(party_id, peer), max(party_id, peer))]
            for peer in party_ids if peer != party_id
        }
        state = AggregatorState(
            client_tokens[party_id],
            control_tokens[party_id],
            party_id=party_id,
            party_count=party_count,
            peer_endpoints={str(peer): "pending" for peer in party_ids if peer != party_id},
            peer_tokens=peer_auth,
        )
        server = create_server("127.0.0.1", 0, state)
        servers.append((server, state, peer_auth))

    endpoints = {
        str(party_id): f"http://127.0.0.1:{server.server_port}"
        for party_id, (server, _, _) in enumerate(servers)
    }
    for party_id, (server, state, peer_auth) in enumerate(servers):
        state.peer_endpoints = {
            str(peer): endpoints[str(peer)]
            for peer in party_ids if peer != party_id
        }
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        harness = PartyHarness.__new__(PartyHarness)
        harness.state, harness.server, harness.thread = state, server, thread
        harness.url = endpoints[str(party_id)]
        parties.append(harness)

    return parties, endpoints, client_tokens, control_tokens


def test_clients_send_shares_directly_and_coordinator_gets_only_second_stage_shares():
    client_ids, party_ids = [0, 1, 2], [0, 1, 2]
    parties, endpoints, client_tokens, control_tokens = _build_party_harnesses(
        client_ids, party_ids
    )
    encoder = FixedPointEncoder(1_000_000)
    global_state = {"weight": torch.tensor([0.0, 0.0])}
    local_states = [
        {"weight": torch.tensor([1.25, -2.0])},
        {"weight": torch.tensor([4.0, 1.0])},
        {"weight": torch.tensor([-0.5, 2.0])},
    ]
    example_counts = [2, 6, 4]
    committees = {0: [0, 1], 1: [1, 2], 2: [0, 2]}
    round_id = "two-stage-test-1"

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

        participants_by_party = {
            party: [cid for cid in client_ids if party in committees[cid]]
            for party in party_ids
        }
        assert all(len(members) == 2 for members in participants_by_party.values())

        # Stage 1: first-stage committee partials are re-shared peer-to-peer;
        # the coordinator receives only a status acknowledgement.
        for party in party_ids:
            result = finalize_party(
                endpoints[str(party)],
                round_id,
                participants_by_party[party],
                control_tokens[party],
            )
            assert result is None

        # Stage 2: only fresh random shares of the global sum reach the server.
        party_outputs = [
            retrieve_party_share(
                endpoints[str(party)], round_id, party_ids, control_tokens[party]
            )
            for party in party_ids
        ]
        actual = reconstruct_weighted_average(
            party_outputs, global_state, sum(example_counts), encoder
        )
        expected = weighted_fedavg(local_states, example_counts, global_state)
        torch.testing.assert_close(
            actual["weight"], expected["weight"], atol=1e-6, rtol=0
        )

        for party in party_ids:
            state = parties[party].state
            assert round_id in state.output_cache
            acknowledge_party(endpoints[str(party)], round_id, control_tokens[party])
            assert round_id not in state.rounds
            assert round_id not in state.digests
            assert round_id not in state.outgoing
            assert round_id not in state.reshared
            assert round_id not in state.reshare_digests
            assert round_id not in state.finalized
            assert round_id not in state.output_cache
    finally:
        for party in parties:
            party.close()


def test_party_rejects_invalid_client_authentication():
    client_token, control_token = secrets.token_urlsafe(24), secrets.token_urlsafe(24)
    peer_token = secrets.token_urlsafe(24)
    state = AggregatorState(
        {"4": client_token},
        control_token,
        party_id=0,
        party_count=2,
        peer_endpoints={"1": "http://127.0.0.1:1"},
        peer_tokens={"1": peer_token},
    )
    server = create_server("127.0.0.1", 0, state)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    url = f"http://127.0.0.1:{server.server_port}"
    try:
        body, keys_json = _encode_arrays({"weight": np.array([1], dtype=np.int64)})
        request = urllib.request.Request(
            url + "/v1/rounds/bad-auth/shares",
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
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def test_party_rejects_finalize_with_missing_client_share():
    client_token, control_token = secrets.token_urlsafe(24), secrets.token_urlsafe(24)
    state = AggregatorState(
        {"0": client_token},
        control_token,
        party_id=0,
        party_count=2,
        peer_endpoints={"1": "http://127.0.0.1:1"},
        peer_tokens={"1": secrets.token_urlsafe(24)},
    )
    body, keys_json = _encode_arrays({"weight": np.array([1], dtype=np.int64)})
    message = b"round-one\n0\n" + keys_json.encode() + b"\n" + body
    state.submit_share(
        "round-one", "0", body, keys_json, _signature(client_token, message)
    )
    canonical = '["missing-client"]'
    signature = _signature(
        control_token, b"finalize\nround-one\n" + canonical.encode()
    )
    with pytest.raises(RuntimeError, match="missing client shares"):
        state.finalize_to_peers("round-one", ["missing-client"], signature)
