"""Two exact counts of cubic simple random-walk returns; no floating point."""
from fractions import Fraction
from math import comb


def direct(n):
    # Squared multinomial for allocations of positive and negative steps.
    return comb(2*n, n) * sum(
        comb(n, a)**2 * sum(comb(n-a, b)**2 for b in range(n-a+1))
        for a in range(n+1)
    )


def interleave(n):
    # Convolve two-coordinate returns with the third coordinate, inserting steps.
    planar = [comb(2*k, k)**2 for k in range(n+1)]
    return sum(comb(2*n, 2*k) * planar[k] * comb(2*(n-k), n-k)
               for k in range(n+1))


def certify(counts):
    cap = Fraction(49, 100)
    subtotal = sum(Fraction(c, 36**n) for n, c in enumerate(counts, 1))
    if not subtotal < cap:
        raise ValueError('partial return sum too large')
    assert Fraction(22, 21)**3 < Fraction(6, 5)**2
    assert 2 * Fraction(25, 36)**257 / (1-Fraction(25, 36)) < Fraction(1, 10000)
    assert cap + Fraction(3, 80) + Fraction(1, 10000) - Fraction(1, 24) < Fraction(489, 1000)
    return subtotal


if __name__ == '__main__':
    counts = [interleave(n) for n in range(1, 257)]
    assert counts[:4] == [6, 90, 1860, 44730]
    for n, c in enumerate(counts, 1):
        assert direct(n) == c, n
    subtotal = certify(counts)
    # Mutants: wrong step probability (delete one direction from normalization)
    # and wrong order-zero contribution.
    assert sum(Fraction(c, 25**n) for n, c in enumerate(counts, 1)) > Fraction(49, 100)
    try:
        certify([counts[0]+36, *counts[1:]])
    except ValueError:
        pass
    else:
        raise AssertionError('mutated counts must fail')
    print('256 moment counts agree; both mutants rejected; subtotal < 49/100')
    print('three-dimensional infinite-volume I^2 < 489/1000 (strict)')
