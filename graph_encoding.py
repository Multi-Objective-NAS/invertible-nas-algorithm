def encode(mat, ops):
    def _sanity_check(idx1, opidx, idx2):
        if idx1 not in range(7):
            return False
        if idx2 not in range(7):
            return False
        if opidx not in range(4):
            return False
        return True

    encoded = []
    for inbound, outbound in zip(*np.array(mat).nonzero()):
        op = ALLOWED_OPS.index(ops[outbound])
        assert _sanity_check(inbound, op, outbound)

        embed = [0] * (7 + 7 + len(ALLOWED_OPS))
        embed[inbound] = 1
        embed[7 + op] = 1
        embed[7 + len(ALLOWED_OPS) + outbound] = 1

        encoded += embed

    assert len(encoded) == 9 * (7 + 7 + len(ALLOWED_OPS))
    return encoded
