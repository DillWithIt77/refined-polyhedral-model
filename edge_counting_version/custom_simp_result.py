"""
Reproduces results5000_tol1e-05.csv by running `_linprog_simplex`
(from custom_simplex_warmstart.py) on each problem in PROBLEMS, with a
callback that:
  - splits iteration count into Phase 1 / Phase 2 buckets
  - splits wall-clock time into Phase 1 / Phase 2 buckets
  - counts "edge steps": non-degenerate objective changes between
    consecutive iterations *within the same phase* (mirrors the
    edge-step logic in the original driver script)

Dependencies you still need to provide:
  - mps_reader_preprocessor.py  (must expose read_mps_preprocess)
  - a directory of .mps files named after each problem (PROBLEM_DIR)
  - custom_simplex_warmstart.py (already provided; must be importable)

NOTE on the "Infeasible" rows:
When Phase 1 fails to find a feasible point, `_linprog_simplex` never
calls Phase 2 -- every iteration that happened, happened in Phase 1.
This script attributes all of those iterations/time/edge-steps to the
"P1" columns for infeasible problems. The CSV you're trying to
reproduce instead shows P1 = 0 and P2 = <all iterations> for infeasible
rows, which looks like a labeling bug in whatever script produced it.
If you find the original script and it turns out that behavior was
intentional, flip INFEASIBLE_ITERS_COUNT_AS_PHASE below.
"""

import os
import time
import numpy as np
import pandas as pd

from mps_reader_preprocessor import read_mps_preprocess
from custom_simplex_warmstart import _linprog_simplex

# =====================================================================
# CONFIG
# =====================================================================

PROBLEM_DIR = 'netlib_lp_subset'

# Superset of Netlib LP problems (matches the ALL_PROBLEMS list commented
# out in the original driver script). Trim/edit this to match exactly
# which problems you want in the CSV.
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
MAXITER = 5000

EDGE_TOL = 1e-9  # threshold for a "non-degenerate" objective change

RESULTS_DIR = './results'
OUT_CSV = os.path.join(RESULTS_DIR, f'results{MAXITER}_tol{TOL:.0e}.csv')

# See note in the module docstring.
INFEASIBLE_ITERS_COUNT_AS_PHASE = 1

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
    """Minimize c @ x s.t. A @ x == b, x >= 0 (converted from mixed
    equality/inequality/free-variable MPS form)."""
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


def make_tracking_callback():
    """
    Callback for _linprog_simplex that records, per iteration:
      nit, phase, objective, wall-clock timestamp
    Kept deliberately lightweight (no x/basis/T storage) since we only
    need iteration counts, phase, objective, and timing to reconstruct
    every CSV column.
    """
    log = []

    def callback(info):
        log.append({
            'nit': info['nit'],
            'phase': info['phase'],
            'objective': info['objective'],
            't': time.time(),
        })

    return callback, log


def summarize_run(log, status, total_iterations):
    """
    Turns the raw per-iteration callback log into the CSV columns:
    P1/P2 iterations, edge steps, P1/P2 steps, time, P1/P2 time.
    """
    if not log:
        return dict(
            p1_iterations=0, p2_iterations=0,
            edge_steps=0, p1_steps=0, p2_steps=0,
            total_time=0.0, p1_time=0.0, p2_time=0.0,
        )

    phase1_log = [row for row in log if row['phase'] == 1]
    phase2_log = [row for row in log if row['phase'] == 2]

    if status == 2 and INFEASIBLE_ITERS_COUNT_AS_PHASE == 1:
        # Phase 2 never ran; everything recorded is genuinely Phase 1.
        # (phase2_log should already be empty in this case.)
        pass

    p1_iterations = len(phase1_log)
    p2_iterations = len(phase2_log)

    def count_edge_steps(phase_log):
        steps = 0
        for i in range(1, len(phase_log)):
            if abs(phase_log[i]['objective'] - phase_log[i - 1]['objective']) > EDGE_TOL:
                steps += 1
        return steps

    p1_steps = count_edge_steps(phase1_log)
    p2_steps = count_edge_steps(phase2_log)
    edge_steps = p1_steps + p2_steps

    def phase_elapsed(phase_log):
        if len(phase_log) < 2:
            return 0.0
        return phase_log[-1]['t'] - phase_log[0]['t']

    p1_time = phase_elapsed(phase1_log)
    p2_time = phase_elapsed(phase2_log)
    # Include the gap between phase1's last iteration and phase2's
    # first iteration in p2_time, so p1_time + p2_time == total_time.
    if phase1_log and phase2_log:
        p2_time += phase2_log[0]['t'] - phase1_log[-1]['t']
    total_time = p1_time + p2_time

    return dict(
        p1_iterations=p1_iterations, p2_iterations=p2_iterations,
        edge_steps=edge_steps, p1_steps=p1_steps, p2_steps=p2_steps,
        total_time=total_time, p1_time=p1_time, p2_time=p2_time,
    )


def solve_one(problem):
    path = os.path.join(PROBLEM_DIR, problem)
    c, A, b = mps_to_standard(path)

    callback, log = make_tracking_callback()

    t_start = time.time()
    x, status, message, nit = _linprog_simplex(
        c=c, c0=0.0, A=A, b=b,
        callback=callback, postsolve_args=None,
        maxiter=MAXITER, tol=TOL, bland=BLAND,
    )
    t_wall = time.time() - t_start

    stats = summarize_run(log, status, nit)

    # Objective reported against the *original* c (matches what the
    # solver returns directly since c already includes the split for
    # free variables via mps_to_standard).
    objective = float(c @ x)

    return dict(
        problem=problem,
        tol=TOL,
        bland=BLAND,
        status=STATUS_LABELS.get(status, f'Unknown ({status})'),
        objective=objective,
        iterations=nit,
        **{'P1 iterations': stats['p1_iterations']},
        **{'P2 iterations': stats['p2_iterations']},
        **{'edge steps': stats['edge_steps']},
        **{'P1 steps': stats['p1_steps']},
        **{'P2 steps': stats['p2_steps']},
        time=t_wall,
        **{'P1 time': stats['p1_time']},
        **{'P2 time': stats['p2_time']},
        message=message,
    )


# =====================================================================
# Main loop
# =====================================================================

def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    records = []
    for i, problem in enumerate(PROBLEMS, start=1):
        print(f'[{i}/{len(PROBLEMS)}] Solving {problem}...')
        try:
            record = solve_one(problem)
        except Exception as exc:
            print(f'  -> FAILED: {exc}')
            record = dict(
                problem=problem, tol=TOL, bland=BLAND, status='Error',
                objective=np.nan, iterations=np.nan,
                **{'P1 iterations': np.nan, 'P2 iterations': np.nan,
                   'edge steps': np.nan, 'P1 steps': np.nan, 'P2 steps': np.nan},
                time=np.nan, **{'P1 time': np.nan, 'P2 time': np.nan},
                message=str(exc),
            )
        records.append(record)

    df = pd.DataFrame.from_records(records, columns=[
        'problem', 'tol', 'bland', 'status', 'objective', 'iterations',
        'P1 iterations', 'P2 iterations', 'edge steps', 'P1 steps',
        'P2 steps', 'time', 'P1 time', 'P2 time', 'message',
    ])
    df.to_csv(OUT_CSV)
    print(f'\nSaved results to {OUT_CSV}')


if __name__ == '__main__':
    main()