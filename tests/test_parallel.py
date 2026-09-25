from business_entity_resolution.parallel import fork_available, parallel_imap, parallel_map


def _square(value: int) -> int:
    return value * value


def test_parallel_map_serial_and_fork():
    assert sorted(parallel_map(_square, [1, 2, 3, 4], workers=1)) == [1, 4, 9, 16]
    if fork_available():
        assert sorted(parallel_map(_square, [1, 2, 3, 4], workers=3)) == [1, 4, 9, 16]


def test_parallel_imap_bounds_in_flight():
    assert sorted(parallel_imap(_square, iter([1, 2, 3, 4]), workers=2, max_inflight=2)) == [1, 4, 9, 16]
