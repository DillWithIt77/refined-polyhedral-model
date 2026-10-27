"""
For each problem, mirrors the basis-recovery approach used in the main
pipeline's `get_edge_start_info`:

  1. Get x_feasible via P.find_feasible_solution(basis=False).
  2. Recover the basis/tableau *for that exact vertex* by building a
     synthetic cost (huge on the variables that are zero/active at
     x_feasible, zero on the rest) and solving that LP to optimality --
     since x_feasible is (generically) the unique minimizer of that
     synthetic cost, the final basis/tableau _linprog_simplex lands on
     corresponds to x_feasible itself.
  3. Warm-start `_linprog_simplex_warmstart` from that basis with the
     REAL objective, and let it run to completion (or maxiter).
  4. Record whether the warm-started solve actually finishes (reaches
     optimality) within budget, and if not, why.

This intentionally reuses the same recovery trick as the main driver
script rather than re-deriving a basis independently, so the vertex
being warm-started from is guaranteed to be the same x_feasible your
SDAC comparisons use.
"""

import os
import time
import numpy as np
import pandas as pd

from mps_reader_preprocessor import read_mps_preprocess
from custom_simplex_warmstart import _linprog_simplex, _linprog_simplex_warmstart
from refined_polyhedron import PolyhedronStandard

# =====================================================================
# CONFIG
# =====================================================================

PROBLEM_DIR = 'netlib_lp_subset'

PROBLEMS = [
    'adlittle', 'afiro', 'agg2', 'agg3', 'agg', 'bandm', 'beaconfd', 'blend', 'bnl1',
    'boeing1', 'boeing2', 'bore3d', 'brandy', 'capri', 'degen2', 'degen3', 'e226',
    'etamacro', 'fffff800', 'finnis', 'fit1d', 'fit1p', 'forplan', 'ganges', 'gfrd-pnc',
    'grow15', 'grow22', 'grow7', 'israel', 'kb2', 'ken-07', 'lotfi', 'maros',
    'modszk1', 'network_ex', 'network_ex_2', 'pds-02', 'perold', 'pilot4', 'pilotnov',
    'qap08', 'recipe', 'sc105', 'sc205', 'sc50a', 'sc50b', 'scagr25', 'scagr7',
    'scfxm1', 'scfxm2', 'scfxm3', 'scorpion', 'scrs8', 'scsd1', 'scsd6', 'scsd8',
    'sctap1', 'sctap2', 'sctap3', 'seba', 'share1b', 'share2b', 'shell', 'ship04l',
    'ship04s', 'ship08l', 'ship08s', 'ship12l', 'ship12s', 'sierra', 'stair',
    'standata', 'standgub', 'standmps', 'stocfor1', 'stocfor2', 'tuff', 'vtp_base',
    'wood1p', 'cre-c', 'angela_test_prob',
]

TOL = 1e-5
BLAND = False
RECOVER_MAXITER = 5000     # budget for recovering the basis at x_feasible
WARMSTART_MAXITER = 5000   # budget for the warm-started solve itself

RESULTS_DIR = './results'
OUT_CSV = os.path.join(RESULTS_DIR, f'warmstart_check_{WARMSTART_MAXITER}_tol{TOL}.csv')

STATUS_LABELS = {
    0: 'Success',
    1: 'Iteration limit',
    2: 'Infeasible',
    3: 'Unbounded',
    4: 'Numerical difficulties',
}

# =====================================================================
# Helpers
# =====================================================================


def mps_to_standard(filepath):
    c, G, h, A, b = read_mps_preprocess(filepath)
    n = A.shape[1]
    m_A = A.shape[0]
    m_G = G.shape[0]
    new_A = np.block([
        [A, -A, np.zeros((m_A, m_G))],
        [G, -G, np.identity(m_G)],
    ])
    new_b = np.concatenate((b, h), axis=0)
    new_c = np.concatenate((c, -c, np.zeros(m_G)), axis=0)
    return new_c, new_A, new_b


def make_list_callback():
    log = []
    def list_callback(info):
        log.append(info)
    return list_callback, log


def recover_basis_for_x(x, P, maxiter=RECOVER_MAXITER):
    """
    Same trick as get_edge_start_info in the main pipeline: build a
    synthetic cost that's huge on the variables active (== 0) at x and
    zero elsewhere, then solve that LP to optimality. The final basis
    is read off the last callback entry.
    """
    active_inds = P.get_active_constraints(x)

    c_recover = P.c.copy()
    c_recover[:] = 1e12
    n = P.A.shape[1]
    mask = np.ones(n, dtype=bool)
    mask[active_inds] = False
    non_active_inds = np.where(mask)[0]
    c_recover[non_active_inds] = 0.0

    list_callback, callback_log = make_list_callback()
    _linprog_simplex(
        c=c_recover, c0=0, A=P.A, b=P.b,
        callback=list_callback, postsolve_args=None,
        maxiter=maxiter, tol=TOL, bland=BLAND,
    )

    if not callback_log:
        raise RuntimeError("callback_log is empty during basis recovery.")

    last_entry = callback_log[-1]
    return {
        'basis': last_entry['basis'],
        'T': last_entry['T'],
        'x_recovered': last_entry['x'],
        'nit': last_entry['nit'],
    }


def run_warmstart_check(problem, tol=TOL, bland=BLAND,
                         recover_maxiter=RECOVER_MAXITER,
                         warmstart_maxiter=WARMSTART_MAXITER):
    row = {
        'problem': problem, 'tol': tol, 'bland': bland,
        'recover_iterations': None, 'recover_time': None,
        'x_match': None,
        'finished': None, 'status': None, 'objective': None,
        'warmstart_iterations': None, 'warmstart_time': None,
        'total_iterations': None, 'total_time': None,
        'message': None,
    }

    path = os.path.join(PROBLEM_DIR, problem)
    try:
        c, A, b = mps_to_standard(path)
        P = PolyhedronStandard(A, b, c)

        print('  Finding x_feasible...')
        x_feasible = P.find_feasible_solution(basis=False)

        print('  Recovering basis at x_feasible...')
        t0 = time.time()
        recovery = recover_basis_for_x(x_feasible, P, maxiter=recover_maxiter)
        recover_time = time.time() - t0

        row['recover_iterations'] = recovery['nit']
        row['recover_time'] = recover_time
        row['x_match'] = bool(np.allclose(recovery['x_recovered'], x_feasible, atol=1e-6))
        if not row['x_match']:
            print(f"  WARNING: recovered vertex doesn't match x_feasible for {problem} "
                  f"(likely degeneracy) -- warm-starting from the recovered vertex anyway.")

        # Guard against artificial-variable indices leaking into the basis
        # (same check as the main pipeline does before using a basis for
        # warm-starting).
        m = P.A.shape[1]
        basis_arr = np.asarray(recovery['basis'])
        if np.any(basis_arr >= m):
            raise RuntimeError(
                f"Recovered basis for {problem} still contains artificial-variable "
                f"indices (>= {m}): {basis_arr[basis_arr >= m]}"
            )

        print('  Warm-starting with real objective...')
        t0 = time.time()
        try:
            x_ws, status, message, nit = _linprog_simplex_warmstart(
                c=P.c, c0=0.0, A=P.A, b=P.b,
                T=recovery['T'], basis=recovery['basis'],
                callback=None, postsolve_args=None,
                maxiter=warmstart_maxiter, tol=tol, bland=bland,
            )
        except Exception as e:
            warmstart_time = time.time() - t0
            row['finished'] = False
            row['status'] = 'Error'
            row['message'] = f'Warm-started solve raised an exception: {e}'
            row['warmstart_time'] = warmstart_time
            row['total_iterations'] = row['recover_iterations']
            row['total_time'] = row['recover_time'] + warmstart_time
            return row

        warmstart_time = time.time() - t0
        finished = (status == 0)

        row.update({
            'finished': finished,
            'status': STATUS_LABELS.get(status, f'Unknown ({status})'),
            'objective': float(P.c @ x_ws) if x_ws is not None else None,
            'warmstart_iterations': nit,
            'warmstart_time': warmstart_time,
            'total_iterations': row['recover_iterations'] + nit,
            'total_time': row['recover_time'] + warmstart_time,
            'message': message,
        })

    except Exception as e:
        row['finished'] = False
        row['status'] = 'Error'
        row['message'] = str(e)

    return row


# =====================================================================
# Main loop
# =====================================================================

def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    records = []

    for i, problem in enumerate(PROBLEMS, start=1):
        print(f'\n[{i}/{len(PROBLEMS)}] Checking warm-started solve for {problem}...')
        row = run_warmstart_check(problem)
        tag = 'FINISHED' if row['finished'] else 'DID NOT FINISH'
        print(f"  -> {tag} (status={row['status']}, "
              f"recover_iters={row['recover_iterations']}, "
              f"warmstart_iters={row['warmstart_iterations']})")
        records.append(row)

        # Save after every problem so a crash doesn't lose progress.
        pd.DataFrame(records).to_csv(OUT_CSV)

    df = pd.DataFrame(records)
    n_finished = df['finished'].sum()
    print(f"\n{n_finished}/{len(df)} problems finished after warm-starting.")
    if n_finished < len(df):
        print("Did not finish:", df.loc[~df['finished'].astype(bool), 'problem'].tolist())
    print(f"Saved results to {OUT_CSV}")


if __name__ == '__main__':
    main()