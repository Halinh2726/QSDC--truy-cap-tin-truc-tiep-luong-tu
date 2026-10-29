"""M1  Optical core: Eqs. (6)-(11), (15)-(20), Lemma III.1, Appendix A, Remark II.1.  Exact linear algebra."""
import itertools
import numpy as np
from core import BELL, BASES, PHI16, enc, S2


def run():
    out = {}
    # App. A: column matrix unitary, adjoint equals (17)
    Bcol = BELL.T
    out["A_unitary"] = float(np.abs(Bcol.conj().T @ Bcol - np.eye(4)).max())
    U17 = S2 * np.array([[1, 0, 0, 1], [0, 1, 1, 0], [1, 0, 0, -1], [0, 1, -1, 0]])
    out["eq17_equals_adjoint"] = float(np.abs(U17 - Bcol.conj().T).max())
    # Eq. (19) and the non-MUB claim
    ov = {}
    for b in ["ZZ", "ZX", "XZ", "XX"]:
        ov[b] = sorted(set(np.round(np.abs(BELL @ BASES[b].T) ** 2, 12).ravel().tolist()))
    out["overlaps"] = ov
    ov12 = sorted(set(np.round(np.abs(BASES["ZX"] @ BASES["XZ"].T) ** 2, 12).ravel().tolist()))
    out["overlap_C1_C2"] = ov12
    # Lemma III.1, Eq. (18), and detector relation (3)
    err18, rel3 = 0.0, True
    for a, w in itertools.product((0, 1), repeat=2):
        lhs = np.kron(enc(a, w), np.eye(4)) @ PHI16
        rhs = sum(0.5 * (-1) ** (w * (u ^ a)) * np.kron(BELL[2 * u + v], BELL[2 * (u ^ a) + (v ^ w)])
                  for u in (0, 1) for v in (0, 1))
        err18 = max(err18, float(np.abs(lhs - rhs).max()))
        # joint detector statistics: P(J_A=(u,v), J_B=(u',v'))
        P = np.abs(np.kron(BELL, BELL) @ lhs) ** 2
        for idx in np.nonzero(P > 1e-12)[0]:
            ja, jb = divmod(idx, 4)
            u, v, u2, v2 = ja >> 1, ja & 1, jb >> 1, jb & 1
            rel3 &= (a == u ^ u2) and (w == v ^ v2)
        out.setdefault("JA_uniform", []).append(np.round(P.reshape(4, 4).sum(1), 12).tolist())
    out["eq18_maxerr"] = err18
    out["eq3_relation_holds"] = bool(rel3)
    # Eq. (20): conditional state of photon B given J_A is |d_{J_A xor S}> (up to phase)
    ok20 = True
    for a, w in itertools.product((0, 1), repeat=2):
        psi = (np.kron(enc(a, w), np.eye(4)) @ PHI16).reshape(4, 4)
        for j in range(4):
            phiB = BELL[j].conj() @ psi
            phiB /= np.linalg.norm(phiB)
            D = j ^ (2 * a + w)
            ok20 &= abs(abs(np.vdot(BELL[D], phiB)) - 1) < 1e-12
    out["eq20_D_equals_JA_xor_S"] = bool(ok20)
    # Sec. II-C / Remark II.1: average transmitted states
    rhoB_msg = [np.einsum("ab,ac->bc", m := (np.kron(enc(a, w), np.eye(4)) @ PHI16).reshape(4, 4), m.conj())
                for a, w in itertools.product((0, 1), repeat=2)]
    out["msg_reduced_state_dev"] = float(max(np.abs(r - np.eye(4) / 4).max() for r in rhoB_msg))
    avg16 = sum(np.outer(v, v) for b in ["ZZ", "ZX", "XZ", "XX"] for v in BASES[b]) / 16
    avg8 = sum(np.outer(v, v) for b in ["ZX", "XZ"] for v in BASES[b]) / 8
    out["ctrl16_avg_dev"] = float(np.abs(avg16 - np.eye(4) / 4).max())
    out["ctrl8_avg_dev"] = float(np.abs(avg8 - np.eye(4) / 4).max())
    # Bell-basis intercept-resend error per control basis (used in M9)
    out["IR_error"] = {b: float(1 - np.mean(np.diag((np.abs(BASES[b] @ BELL.T) ** 2) @ (np.abs(BELL @ BASES[b].T) ** 2))))
                       for b in ["ZZ", "ZX", "XZ", "XX"]}
    return out


if __name__ == "__main__":
    import json; print(json.dumps(run(), indent=1))
