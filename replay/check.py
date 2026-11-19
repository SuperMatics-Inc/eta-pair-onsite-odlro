"""Four-site CAR checks of projected trace inequalities, not a test of 3D LRO."""
import os

os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'

import numpy as np


def adj(matrix):
    return matrix.conj().T


def annihilators(nsites):
    size = 1 << (2 * nsites)
    ops = np.zeros((2 * nsites, size, size))
    for mode, op in enumerate(ops):
        bit = 1 << mode
        for state in range(size):
            if state & bit:
                op[state ^ bit, state] = (-1) ** ((state & (bit - 1)).bit_count())
    return ops


def system(sites, bonds, ops, hopping, v, lam, real=0.):
    identity = np.eye(ops.shape[-1])
    occupations = [adj(a) @ a for a in ops]
    n = np.diagonal(np.array(occupations), axis1=1, axis2=2).real
    z = np.array([1 - n[2*x] - n[2*x+1] for x in range(len(sites))])
    p = [ops[2*x+1] @ ops[2*x] for x in range(len(sites))]
    xops = [a + adj(a) for a in p]
    yops = [1j * (adj(a) - a) for a in p]
    h0 = np.diag(lam * np.sum((n[::2] - .5) * (n[1::2] - .5), axis=0)).astype(complex)
    sources = []
    for x, y, j, sign in bonds:
        h0 += adj(p[x]) @ p[y] + adj(p[y]) @ p[x] + v * np.diag(z[x] * z[y])
        for spin in range(2):
            edge = adj(ops[2*x+spin]) @ ops[2*y+spin]
            coeff = real + 1j * hopping[j] * sign
            h0 += coeff * edge + coeff.conjugate() * adj(edge)
        sources.append((-1)**sum(sites[x]) * xops[x] - (-1)**sum(sites[y]) * xops[y])
    assert np.max(abs(h0 - adj(h0))) < 1e-13
    return h0, np.array(sources), n.sum(axis=0).astype(int), (identity, xops, yops, z, p)


def log_z(h0, a, beta, mask, field):
    matrix = h0 + .5 * np.einsum('j,jmn->mn', field, a)
    levels, u = np.linalg.eigh(matrix)
    projection = np.sum(abs(u[mask])**2, axis=0)
    return np.log(np.dot(np.exp(-beta * (levels-levels[0])), projection)) - beta * (levels[0] + .25*(field @ field))


def field_copy(bonds, reflection, left):
    """Rows map each oriented original bond to its copied field (possibly zero)."""
    indices = {(x, y): (i, 1.) for i, (x, y, _, _) in enumerate(bonds)}
    indices |= {(y, x): (i, -1.) for i, (x, y, _, _) in enumerate(bonds)}
    copy = np.zeros((len(bonds), len(bonds)))
    for i, (x, y, _, _) in enumerate(bonds):
        if x in left and y in left:
            copy[i, i] = 1
        elif x not in left and y not in left:
            j, sign = indices[reflection[x], reflection[y]]
            copy[i, j] = sign
    return copy


def curvature(h0, a, beta, number, mask, field):
    # Diagonalize separately in raw number sectors; exact degeneracies cannot mix sectors.
    vals = np.empty(len(number))
    vectors = np.zeros_like(h0)
    for k in np.unique(number):
        indices = np.flatnonzero(number == k)
        vals[indices], vectors[np.ix_(indices, indices)] = np.linalg.eigh(h0[np.ix_(indices, indices)])
    a_eig = adj(vectors) @ np.einsum('j,jmn->mn', field, a) @ vectors
    weights = np.exp(-beta * (vals[mask]-min(vals[mask])))
    weights = weights[:, None] / weights.sum() * abs(a_eig[mask])**2
    s = beta * (vals[None, :] - vals[mask, None])
    large = np.abs(s) > 1e-3
    f = np.empty_like(s)
    f[large] = 2 * (np.expm1(-s[large]) + s[large]) / s[large]**2
    t = s[~large]
    f[~large] = 1 - t/3 + t*t/12 - t*t*t/60
    g, b, c = weights.sum(), (weights*f).sum(), 2*(weights*s).sum()/beta
    assert g <= b + .5*np.sqrt(beta*b*max(c, 0)) + 1e-12
    assert max(-c, 0) <= 6*b/beta + 1e-12
    if mask.sum() != len(mask):
        assert np.max(abs(a_eig[np.ix_(mask, mask)])) < 1e-13
    hess = beta**2/4*b - beta/2*(field @ field)
    eps = 5e-4
    delta = (log_z(h0, a, beta, mask, eps*field) + log_z(h0, a, beta, mask, -eps*field)
             - 2*log_z(h0, a, beta, mask, field*0)) / eps**2
    assert abs(delta-hess) < 5e-6 and hess < 0
    return b, hess, delta


def spatial_reflection(matrix, perm):
    """Signed permutation of occupation vectors, composed with conjugation."""
    signs, image = [], []
    for state in range(len(matrix)):
        occ = [p for i, p in enumerate(perm) if state & (1 << i)]
        image.append(sum(1 << p for p in occ))
        signs.append((-1) ** sum(x > y for i, x in enumerate(occ) for y in occ[i+1:]))
    return matrix.conj()[np.ix_(image, image)] * np.outer(signs, signs)


def reflection_checks(sites, bonds, ops, cut, beta, hopping, v, lam):
    reflection, left = cut
    perm = [2*r+s for r in reflection for s in range(2)]
    transform = lambda a: spatial_reflection(a, perm)
    changed = np.array([1j**sum(c % 2 == 0 for c in site[1:]) *
                        (adj(ops[2*x+s]) if sum(site) % 2 else ops[2*x+s])
                        for x, site in enumerate(sites) for s in range(2)])
    h, _, number, _ = system(sites, bonds, changed, hopping, v, lam)
    _, _, _, (i, x, y, z, p) = system(sites, [], ops, hopping, 0, 0)
    mask = number == len(sites)
    proj = np.diag(mask).astype(float)
    assert np.array_equal(proj, transform(proj)) and np.max(abs(h - transform(h))) < 1e-13
    assert np.max(abs(h @ proj - proj @ h)) < 1e-13
    qleft = sum((-1)**sum(sites[l]) * z[l] for l in left)
    fourier = np.zeros_like(proj, dtype=complex)
    for k in range(2*len(sites)+1):
        phase = np.diag(np.exp(2j*np.pi*k/(2*len(sites)+1)*qleft))
        fourier += phase @ transform(phase) / (2*len(sites)+1)
    assert np.max(abs(fourier - proj)) < 1e-13
    vals, u = np.linalg.eigh(h)
    density = proj @ ((u * np.exp(-beta*(vals-vals[0]))) @ adj(u)) @ proj
    density /= np.trace(density)
    a, b = sorted(left)
    examples = [i, x[a], y[a], np.diag(z[a]), p[a], adj(p[a]), x[a] @ np.diag(z[b])]
    gram = np.array([[np.einsum('ij,ji->', density @ a, transform(b))
                      for b in examples] for a in examples])
    assert np.max(abs(gram-adj(gram))) < 1e-12
    smallest = np.linalg.eigvalsh(gram)[0]
    assert smallest >= -1e-12
    print(f'  projected even RP: min sampled Gram eigenvalue {smallest:.9g}')


def main():
    ops = annihilators(4)
    cases = [
        ('antiperiodic ring', [(-1,), (0,), (1,), (2,)],
         [(i,(i+1)%4,0,1 if i < 3 else -1) for i in range(4)],
         [([3,2,1,0], {0,1}), ([1,0,3,2], {1,2})]),
        ('open pi-flux plaquette', [(0,0), (1,0), (0,1), (1,1)],
         [(0,1,0,1), (2,3,0,1), (0,2,1,1), (1,3,1,-1)],
         [([1,0,3,2], {0,2}), ([2,3,0,1], {0,1})]),
    ]
    rng = np.random.default_rng(20261118)
    fields = rng.normal(size=(6, 4)) * np.array([.04, .15, .4, .75, 1., 1.5])[:,None]
    for name, sites, bonds, cuts in cases:
        print(name)
        reflection_checks(sites, bonds, ops, cuts[0], 1.3, (-.24,.37), .11, .23)
        for beta, hopping, v, lam in [(1.3, (-.24,.37), .11, .23),
                                      (1e4, (-1e-4,1e-4), 1e-4, 1e-4)]:
            h0, a, number, _ = system(sites, bonds, ops, hopping, v, lam)
            for label, mask in [('canonical',number == 4), ('grand canonical',number >= 0)]:
                logz = lambda h: log_z(h0, a, beta, mask, h)
                base = logz(np.zeros(4))
                gd = max(logz(h)-base for h in fields)
                chess = max(2*logz(h)-logz(field_copy(bonds,r,left) @ h)
                            - logz(field_copy(bonds,r,set(range(4))-left) @ h)
                            for r,left in cuts for h in fields)
                print(f'  {label}, beta={beta:g}: max GD residual {gd:.9g}; max chess residual {chess:.9g}')
                assert gd < 1e-8 and chess < 1e-8
                if beta == 1.3:
                    b, hess, delta = curvature(h0, a, beta, number, mask, fields[2])
                    print(f'    spectral B={b:.9g}; analytic curvature={hess:.9g}; finite diff={delta:.9g}')
    # One-bond ordinary-hopping negative control in transformed variables.
    ops = annihilators(2)
    ops[2:] = ops[2:].transpose(0,2,1).copy()
    h, _, _, (_, _, _, _, p) = system([(0,), (1,)], [(0,1,0,1)], ops, (.31,), 0, 0, real=.2)
    theta_h = spatial_reflection(h, [2,3,0,1])
    defect = np.max(abs(theta_h-h))
    assert abs(defect-.4) < 1e-13
    # p in system() is the pre-transformation annihilator mapped to the new basis.
    dstar = -adj(p[0]) @ p[1]
    vals,u = np.linalg.eigh(h)
    beta = .01
    imaginary = np.trace(dstar @ (u*np.exp(-beta*vals)) @ adj(u)).imag
    assert abs(imaginary/(2*beta**2*.2*.31)-1) < 1e-3
    print(f'ordinary hopping negative control: reflection defect {defect:.6g}; Im unnormalized trace {imaginary:.9g}')


if __name__ == '__main__':
    main()
