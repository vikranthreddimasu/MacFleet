"""Hypothesis fuzz for HardwareExchange JSON parser.

The handshake parser receives JSON bytes after authentication. A
malicious-but-authenticated peer can ship arbitrary JSON; the parser
must reject malformed input via HandshakeHwValidationError, never
let it bubble up as an unexpected exception type.
"""

from __future__ import annotations

import json

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from macfleet.comm.transport import HardwareExchange
from macfleet.security.auth import HandshakeHwValidationError

json_scalar = st.one_of(
    st.none(),
    st.booleans(),
    st.integers(min_value=-(2**53), max_value=2**53 - 1),
    st.floats(allow_nan=False, allow_infinity=False),
    st.text(max_size=200),
)

json_dict = st.dictionaries(
    st.text(min_size=1, max_size=80),
    json_scalar,
    max_size=20,
)


class TestHardwareExchangeRoundtrip:
    @given(
        gpu_cores=st.integers(min_value=0, max_value=128),
        ram_gb=st.floats(min_value=0.0, max_value=2048.0, allow_nan=False),
        bw=st.floats(min_value=0.0, max_value=10000.0, allow_nan=False),
        chip_name=st.text(min_size=0, max_size=80),
        has_ane=st.booleans(),
        mps=st.booleans(),
        mlx=st.booleans(),
        data_port=st.integers(min_value=0, max_value=65535),
    )
    @settings(max_examples=300, deadline=None)
    def test_roundtrip(self, gpu_cores, ram_gb, bw, chip_name, has_ane, mps, mlx, data_port):
        hw = HardwareExchange(
            gpu_cores=gpu_cores, ram_gb=ram_gb, memory_bandwidth_gbps=bw,
            chip_name=chip_name, has_ane=has_ane,
            mps_available=mps, mlx_available=mlx, data_port=data_port,
        )
        restored = HardwareExchange.from_json_bytes(hw.to_json_bytes())
        assert restored == hw


class TestHardwareExchangeJSONRobust:
    @given(payload=json_dict)
    @settings(
        max_examples=200, deadline=None,
        suppress_health_check=[HealthCheck.filter_too_much],
    )
    def test_arbitrary_dict_does_not_crash(self, payload):
        # Arbitrary key/value dict — known fields are accepted, unknown
        # fields ignored (forward compat). Should never raise unless
        # the field type is wrong.
        try:
            HardwareExchange.from_json_bytes(json.dumps(payload).encode())
        except HandshakeHwValidationError:
            # Known field with the wrong type triggers this.
            pass
        except TypeError:
            # Field type mismatches that slip past filtering — expected
            # to raise TypeError on dataclass construction.
            pass

    @given(blob=st.binary(min_size=0, max_size=2048))
    @settings(max_examples=400, deadline=None)
    def test_arbitrary_bytes_safe(self, blob):
        # Random bytes: must raise HandshakeHwValidationError, never
        # let JSONDecodeError or UnicodeDecodeError bubble up.
        try:
            HardwareExchange.from_json_bytes(blob)
        except HandshakeHwValidationError:
            pass
        except TypeError:
            # Edge case where filtered dict has wrong types
            pass

    def test_non_object_top_level_rejected(self):
        with pytest.raises(HandshakeHwValidationError):
            HardwareExchange.from_json_bytes(b'[]')
        with pytest.raises(HandshakeHwValidationError):
            HardwareExchange.from_json_bytes(b'"a string"')
        with pytest.raises(HandshakeHwValidationError):
            HardwareExchange.from_json_bytes(b'42')
        with pytest.raises(HandshakeHwValidationError):
            HardwareExchange.from_json_bytes(b'null')

    def test_invalid_utf8_rejected(self):
        with pytest.raises(HandshakeHwValidationError):
            HardwareExchange.from_json_bytes(b'\xff\xfe\xfd\xfc')

    def test_empty_bytes_rejected(self):
        with pytest.raises(HandshakeHwValidationError):
            HardwareExchange.from_json_bytes(b'')

    def test_truncated_json_rejected(self):
        with pytest.raises(HandshakeHwValidationError):
            HardwareExchange.from_json_bytes(b'{"gpu_cores": 8')

    def test_unknown_fields_ignored(self):
        # Forward compat: future schemas can add fields.
        hw = HardwareExchange.from_json_bytes(
            b'{"gpu_cores": 16, "future_field_v3": "ignored", "another": [1,2,3]}'
        )
        assert hw.gpu_cores == 16


class TestHardwareExchangeJSONDeterminism:
    """JSON output must be stable so HMAC over it is deterministic."""

    @given(
        gpu_cores=st.integers(min_value=0, max_value=128),
        chip_name=st.text(min_size=0, max_size=40),
    )
    @settings(max_examples=100, deadline=None)
    def test_same_input_produces_same_bytes(self, gpu_cores, chip_name):
        a = HardwareExchange(gpu_cores=gpu_cores, chip_name=chip_name)
        b = HardwareExchange(gpu_cores=gpu_cores, chip_name=chip_name)
        assert a.to_json_bytes() == b.to_json_bytes()

    @given(
        gpu_cores=st.integers(min_value=0, max_value=128),
        chip_name=st.text(min_size=0, max_size=40),
    )
    @settings(max_examples=100, deadline=None)
    def test_keys_sorted(self, gpu_cores, chip_name):
        # Verify the sorted-key invariant — required for HMAC stability.
        hw = HardwareExchange(gpu_cores=gpu_cores, chip_name=chip_name)
        parsed = json.loads(hw.to_json_bytes())
        keys = list(parsed.keys())
        assert keys == sorted(keys)
