"""M2  Single-round quantum checks with full quantum side information.

Verified statements (v4_final):
  Prop. III.2   actual and virtual descriptions give the same test statistics and detection probability
  Lemma III.3   H(S | J_A E, det)_actual = H(D | E, det)_virtual                         (21)
  Eqs. (23)-(25), Lemma IV.1, Prop. V.1   H(D | E, det) >= r(e_j) and >= r(pooled e)    (25), (27)
  Prop. V.3     same bound with an arbitrary trusted local receiver unitary before Bob's analyzer (41)
  J_A role      H(S | E, det) = 2 for every attack (no-signalling); only J_A makes S decodable
  Counterexample  ZZ / XX are not valid secrecy tests (parity attack: e_ZZ = e_XX = 0, H(D|E) = 1)

Eve's single-round attack: an isometry V : Q -> B E with B = C^4 (+) |vac> (loss) followed by Bob's
trusted basis-independent detection filter (projection onto the 4-dimensional one-photon subspace).
E is the complete purification, so the bound is tested against unrestricted single-round quantum side
information conditioned on any fixed history (the GEAT step then handles memory across rounds).
"""
import itertools
import numpy as np
from core import BELL, BASES, C1, C2, PHI16, enc, r_rate, cond_entropy_cq, haar_isometry, gue

DB = 5  # Bob's space: 4 one-photon modes + vacuum (index 4)


# ------------------------------------------------------------------ attack families
def attack_haar_mix(f, dE2, rng):
    """V = sqrt(1-f) id (x) |0>_m  +  sqrt(f) W (x) |1>_m,  W Haar isometry Q -> B E2 (includes loss)."""
    W = haar_isometry(4, DB * dE2, rng).reshape(DB, dE2, 4)
    dE = 2 * dE2
    V = np.zeros((DB, dE, 4), complex)
    for k in range(4):
        V[k, 0, k] = np.sqrt(1 - f)                 # identity branch, marker 0, E2 index 0
    V[:, dE2:, :] = np.sqrt(f) * W                  # marker 1
    return V


def attack_coherent(theta, rng, loss_angle=0.0):
    """Coherent interaction U = exp(-i theta G) on Q (x) E_in (dim 4), then optional
    E-controlled loss: with amplitude sin(loss_angle) the photon is replaced by vacuum
    conditioned on Eve's register being in its first half (state-dependent loss)."""
    G = gue(16, rng)
    w, v = np.linalg.eigh(G)
    U = (v * np.exp(-1j * theta * w)) @ v.conj().T          # (Q E_out) x (Q E_in)
    U = U.reshape(4, 4, 4, 4)[:, :, :, 0]                   # E_in = |0>  -> V[b, e, q]
    V = np.zeros((DB, 8, 4), complex)
    c, s = np.cos(loss_angle), np.sin(loss_angle)
    for e in range(4):
        amp = s if e < 2 else 0.0
        V[:4, e, :] += (np.sqrt(1 - amp**2)) * U[:, e, :]
    # route vacuum amplitude: photon b with E state e (e<2) -> vacuum, Eve stores b in E[4+b]
    for e in range(2):
        for b in range(4):
            V[4, 4 + b, :] += s * U[b, e, :] / np.sqrt(2)
    # re-orthonormalise columns (keeps the isometry exact)
    M = V.reshape(DB * 8, 4)
    q, r = np.linalg.qr(M)
    return (q * (np.diag(r) / np.abs(np.diag(r)))).reshape(DB, 8, 4)


def attack_intercept(f, block_prob=None, basis=None):
    """Bell-basis intercept-resend on a fraction f of rounds; Eve keeps outcome d.
    block_prob[d]: probability that Eve replaces the resent photon by vacuum after outcome d
    (outcome-dependent loss, the most direct attempt to exploit loss postselection)."""
    block_prob = np.zeros(4) if block_prob is None else np.asarray(block_prob)
    basis = BELL if basis is None else basis
    dE = 5 + 4  # 0: no attack; 1..4: outcome d, photon resent; 5..8: outcome d, blocked
    V = np.zeros((DB, dE, 4), complex)
    for k in range(4):
        V[k, 0, k] = np.sqrt(1 - f)
    for d in range(4):
        P = np.outer(basis[d].conj(), basis[d])            # projector on Q (virtual basis -> conj)
        keep, blk = np.sqrt(1 - block_prob[d]), np.sqrt(block_prob[d])
        V[:4, 1 + d, :] += np.sqrt(f) * keep * P
        V[4, 5 + d, :] += np.sqrt(f) * blk * basis[d]
    return V


def attack_bell_dephasing(e):
    """Kraus operators sqrt(p_m) Z_m, Z_m = sum_d (-1)^{m.d} |d><d|, p = (1-e, e/3, e/3, e/3).
    Every Z_m (m != 0) is a fixed-point-free permutation of both C1 and C2, so e1 = e2 = e,
    and Eve's purification gives H(D|E) = 2 - H(p) = r(e): the pooled bound (27) is attained."""
    p = np.array([1 - e, e / 3, e / 3, e / 3])
    V = np.zeros((DB, 4, 4), complex)
    for m in range(4):
        Z = sum((-1) ** ((m >> 1) * (d >> 1) + (m & 1) * (d & 1)) * np.outer(BELL[d], BELL[d].conj()) for d in range(4))
        V[:4, m, :] = np.sqrt(p[m]) * Z
    return V


def attack_parity(basis="ZZ"):
    """Non-destructive parity measurement {span(H0,V1), span(H1,V0)} (basis 'ZZ'), or its
    Hadamard-rotated version (basis 'XX').  It commutes with the corresponding unmixed basis
    and reveals one bit of the Bell label."""
    Pe = np.diag([1, 0, 0, 1.0])
    if basis == "XX":
        Hd = np.array([[1, 1], [1, -1]]) / np.sqrt(2); HH = np.kron(Hd, Hd); Pe = HH @ Pe @ HH
    Po = np.eye(4) - Pe
    V = np.zeros((DB, 2, 4), complex)
    V[:4, 0, :] = Pe; V[:4, 1, :] = Po
    return V


# ------------------------------------------------------------------ evaluation
def virtual_state(V, receiver_U=None):
    """Psi[a, b, e] after Eve's isometry on Q of |Phi16>_{AQ}, postselected on det (b < 4)."""
    psi = np.einsum("beq,aq->abe", V, PHI16.reshape(4, 4))
    psi = psi[:, :4, :]
    pdet = np.linalg.norm(psi) ** 2
    psi = psi / np.sqrt(pdet)
    if receiver_U is not None:
        psi = np.einsum("cb,abe->ace", receiver_U, psi)
    return psi, pdet


def H_D_given_E(psi, basis=BELL):
    blocks = []
    for d in basis:
        phi = np.einsum("a,abe->be", d.conj(), psi)
        blocks.append(phi.T @ phi.conj())
    return cond_entropy_cq(blocks)


def mismatch(psi, basisA, basisB=None):
    basisB = basisA if basisB is None else basisB
    amp = np.einsum("ka,lb,abe->kle", basisA.conj(), basisB.conj(), psi)
    P = (np.abs(amp) ** 2).sum(2)
    return float(1 - np.trace(P) / P.sum())


def actual_protocol(V):
    """Actual protocol of Sec. II: message rounds with encoding on A, control rounds with C1/C2 product states."""
    res = {}
    # message: H(S | J_A E, det), H(S | E, det), detection probability
    blocksSJ, blocksS, pdets = [], [], []
    for a, w in itertools.product((0, 1), repeat=2):
        st = (np.kron(enc(a, w), np.eye(4)) @ PHI16).reshape(4, 4)
        psi = np.einsum("beq,aq->abe", V, st)[:, :4, :]
        pdets.append(np.linalg.norm(psi) ** 2)
        accS = 0
        for j in range(4):
            phi = np.einsum("a,abe->be", BELL[j].conj(), psi)
            rhoE = phi.T @ phi.conj()
            blocksSJ.append((j, rhoE)); accS = accS + rhoE
        blocksS.append(accS)
    pdet = np.mean(pdets)
    # H(S | J E) = H(S J E) - H(J E) with S uniform (weight 1/4)
    def norm(b): return [x / (4 * pdet) for x in b]
    HSJE = cond_entropy_cq(norm([b for _, b in blocksSJ]))           # H(SJ | E)
    HJE = cond_entropy_cq(norm([sum(b for j2, b in blocksSJ if j2 == j) for j in range(4)]))
    res["H_S_given_JE"] = HSJE - HJE
    res["H_S_given_E"] = cond_entropy_cq(norm(blocksS))
    res["pdet_msg_spread"] = float(np.ptp(pdets))
    res["pdet_msg"] = float(pdet)
    # control rounds: Alice sends |c_k>, Bob measures the same basis after the filter
    for name, Cb in (("C1", C1), ("C2", C2)):
        err, det = 0.0, 0.0
        for k in range(4):
            out = np.einsum("beq,q->be", V, Cb[k])[:4, :]
            det += np.linalg.norm(out) ** 2 / 4
            amp = np.einsum("lb,be->le", Cb.conj(), out)
            P = (np.abs(amp) ** 2).sum(1)
            err += (P.sum() - P[k]) / 4
        res[f"e_{name}"] = float(err / det)
        res[f"pdet_{name}"] = float(det)
    return res


def evaluate(V, receiver_U=None, lam=0.5):
    psi, pdet = virtual_state(V, receiver_U)
    e1, e2 = mismatch(psi, C1), mismatch(psi, C2)
    eZZ, eXX = mismatch(psi, BASES["ZZ"]), mismatch(psi, BASES["XX"])
    H = H_D_given_E(psi)
    e_pool = lam * e1 + (1 - lam) * e2
    return dict(pdet=pdet, e1=e1, e2=e2, eZZ=eZZ, eXX=eXX, e_pool=e_pool, H=H,
                gap_pool=H - float(r_rate(e_pool)), gap_best=H - float(r_rate(min(e1, e2))),
                gap_ZZ=H - float(r_rate(eZZ)), gap_XX=H - float(r_rate(eXX)))


def run(n_random=3000, seed=7):
    rng = np.random.default_rng(seed)
    rows = []
    # random families
    for _ in range(n_random):
        fam = rng.integers(3)
        if fam == 0:
            f = rng.uniform(0, 1) ** 2; V = attack_haar_mix(f, int(rng.integers(1, 5)), rng); tag = "haar_mix"
        elif fam == 1:
            V = attack_coherent(rng.uniform(0, 3), rng, rng.uniform(0, 1.2)); tag = "coherent"
        else:
            V = attack_intercept(rng.uniform(0, 1), rng.uniform(0, 1, 4)); tag = "intercept+loss"
        U = None
        if rng.random() < 0.3:                                   # Prop. V.3: receiver misalignment
            w, v = np.linalg.eigh(gue(4, rng)); U = (v * np.exp(-1j * rng.uniform(0, 0.6) * w)) @ v.conj().T
            tag += "+rx"
        ev = evaluate(V, U); ev["family"] = tag
        if U is None:                                            # actual vs virtual
            act = actual_protocol(V)
            ev["dev_H_actual_virtual"] = abs(act["H_S_given_JE"] - ev["H"])
            ev["dev_e_actual_virtual"] = max(abs(act["e_C1"] - ev["e1"]), abs(act["e_C2"] - ev["e2"]))
            ev["dev_pdet"] = max(abs(act["pdet_C1"] - ev["pdet"]), abs(act["pdet_C2"] - ev["pdet"]),
                                 abs(act["pdet_msg"] - ev["pdet"]), act["pdet_msg_spread"])
            ev["H_S_given_E"] = act["H_S_given_E"]
        rows.append(ev)
    # structured sweeps for the figure
    sweep = []
    for f in np.linspace(0, 1, 21):
        ev = evaluate(attack_intercept(f)); ev["f"] = f; ev["family"] = "intercept"; sweep.append(ev)
    for e in np.linspace(0, 0.75, 16):
        ev = evaluate(attack_bell_dephasing(e)); ev["f"] = e; ev["family"] = "bell_dephasing"; sweep.append(ev)
    parity = {b: evaluate(attack_parity(b)) for b in ("ZZ", "XX")}
    return rows, sweep, parity


if __name__ == "__main__":
    rows, sweep, parity = run(400)
    import pandas as pd
    df = pd.DataFrame(rows)
    print(df[["gap_pool", "gap_best", "dev_H_actual_virtual", "dev_e_actual_virtual", "dev_pdet", "H_S_given_E"]].describe())
    print("parity:", parity)
