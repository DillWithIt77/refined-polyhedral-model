import os
import sys
import numpy as np
import pandas as pd
from scipy.sparse import issparse
from scipy.sparse.linalg import splu  # needed for the rank-estimation branch below
import time

from mps_reader_preprocessor import read_mps_preprocess
from custom_simplex_warmstart import _linprog_simplex, _linprog_simplex_warmstart
from refined_polyhedron import PolyhedronStandard
from refined_sdac import refined_sdac
from utils import result

from datetime import datetime

# =====================================================================
# CONFIG -- everything you'd normally comment/uncomment lives here.
# =====================================================================

PROBLEM_DIR = 'netlib_lp_subset'
###All netlib problems for reference
# ALL_PROBLEMS = [
#     'adlittle', 'afiro', 'agg2', 'agg3', 'agg', 'bandm', 'beaconfd', 'blend', 'bnl1',
#     'boeing1', 'boeing2', 'bore3d', 'brandy', 'capri', 'degen2', 'degen3', 'e226',
#     'etamacro', 'fffff800', 'finnis', 'fit1d', 'fit1p', 'forplan', 'ganges', 'gfrd-pnc',
#     'grow15', 'grow22', 'grow7', 'kb2', 'ken-07', 'lotfi', 'maros', 'pds-02', 'perold',
#     'pilotnov', 'qap08', 'recipe', 'sc105', 'sc205', 'sc50a', 'sc50b', 'scagr25',
#     'scagr7', 'scfxm1', 'scfxm2', 'scfxm3', 'scorpion', 'scrs8', 'scsd1', 'scsd6',
#     'scsd8', 'sctap1', 'sctap2', 'sctap3', 'seba', 'share1b', 'share2b', 'shell',
#     'ship04l', 'ship04s', 'ship08l', 'ship08s', 'ship12l', 'ship12s', 'sierra', 'stair',
#     'standata', 'standgub', 'standmps', 'stocfor1', 'stocfor2', 'tuff', 'vtp_base',
#     'wood1p', 'cre-c', 'modszk1',
# ]

###Problems that work for trivial and edge warm-starts for reference
WS_PROBLEMS = [
    'adlittle', 'afiro', 'agg', 'agg2', 'agg3',
    'beaconfd', 'blend', 'capri', 'e226', 'gfrd-pnc',
    'israel', 'kb2', 'lotfi', 'sc105', 'sc205',
    'sc50a', 'sc50b', 'scagr7', 'scrs8', 'scsd1',
    'sctap1', 'share1b', 'share2b', 'ship04l', 'ship04s',
    'ship08s', 'ship12s', 'standata', 'standmps', 'stocfor1'
]
PROBLEMS = WS_PROBLEMS

###use (sub)set of problems to run
# PROBLEMS = ['etamacro']

RESULTS_DIR = './results'

# Baseline solves. Flip these off instead of commenting out the calls below.
RUN_LP_BASELINE = True       # Gurobi-based P.solve_lp
RUN_SCIPY_BASELINE = True    # scipy simplex baseline

# Common facet-restriction settings shared by every SDAC variant below.
# Override per-variant by putting the key in that variant's dict instead.
DEFAULT_SDAC_KWARGS = dict(
    facet_restriction=False,
    facet_rest_type='active',
    percent=10,
    verbose=False,
)

# Each entry is one refined_sdac run. To turn a variant on/off, just flip
# 'enabled' -- nothing else in the file needs to change.
#
#   warm_start=None      -> no warm start, x0 = x_feasible
#   warm_start='trivial'  -> x0 = x_feasible, first_warm_start = basis1
#   warm_start='edge'     -> x0 = x1, x2 = x2, first/second_warm_start = basis1/basis2
RUN_CONFIGS = [
    # dict(name='sdac', enabled=True, warm_start=None),
    dict(name='sdac_rest', enabled=True, warm_start=None,facet_restriction=True),
    # dict(name='sdac_triv', enabled=True, warm_start='trivial', verbose = False),
    # dict(name='sdac_edge', enabled=True, warm_start='edge', verbose = False) ,
    # dict(name='sdac_triv_rest', enabled=True, warm_start='trivial', facet_restriction=True, verbose = False),
    # dict(name='sdac_edge_rest', enabled=True, warm_start='edge', facet_restriction=True, verbose = False)
    # Example of an easy-to-add extra variant, off by default:
    # dict(name='plain_no_restriction', enabled=False, warm_start=None,
    #      facet_restriction=False),
]

# =====================================================================
# Logging setup
# =====================================================================

os.makedirs('./logs', exist_ok=True)
log_path = f"./logs/run_log_{datetime.now():%Y%m%d_%H%M%S}.txt"
log_file = open(log_path, 'w')


class Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for s in self.streams:
            s.write(data)

    def flush(self):
        for s in self.streams:
            s.flush()


sys.stdout = Tee(sys.stdout, log_file)
sys.stderr = Tee(sys.stderr, log_file)

# =====================================================================
# Helpers
# =====================================================================


def mps_to_standard(filepath):
    # Minimize:: c @ x
    # Subject to:: A @ x == b, G @ x <= h
    c, G, h, A, b = read_mps_preprocess(filepath)

    n = A.shape[1]
    m_A = A.shape[0]
    m_G = G.shape[0]

    # Top blocks:    [A, -A, 0]
    # Bottom blocks: [G, -G, I]
    new_A = np.block([
        [A, -A, np.zeros((m_A, m_G))],
        [G, -G, np.identity(m_G)]
    ])
    new_b = np.concatenate((b, h), axis=0)
    new_c = np.concatenate((c, -c, np.zeros(m_G)), axis=0)

    print(f"[DEBUG inside mps_to_standard] new_c shape: {new_c.shape}, new_A shape: {new_A.shape}")
    return new_c, new_A, new_b


# Tolerance for detecting an edge step
EDGE_TOL = 1e-9


class StopWhenFeasible(Exception):
    def __init__(self, data):
        self.data = data


def vertex_basis_callback(info):
    """Stop after finding a feasible solution (end of Phase I)."""
    if info.get('phase') == 2:
        raise StopWhenFeasible(info)


def make_edge_basis_callback():
    """
    Returns a fresh callback + its state container. Stops after taking the
    first edge step in Phase II. Using a closure instead of module-level
    globals means each problem gets its own clean state.
    """
    state = {'info_list': [], 'current_obj_val': 0, 'current_phase': 1}

    def edge_basis_callback(info):
        if info.get('phase') != 2:
            return
        # print(f"iteration {info.get('nit')} (phase {info.get('phase')}) \t --> \t obj = {info.get('objective')}")
        if info.get('phase') - state['current_phase'] != 0:
            # First iteration of Phase II
            print("  -> first Phase II iteration")
            state['info_list'].append(info)
            state['current_phase'] = info.get('phase')
            state['current_obj_val'] = info.get('objective')
        elif abs(info.get('objective') - state['current_obj_val']) > EDGE_TOL:
            # First (non-degenerate) edge step
            print(f"  -> non-degenerate edge step (change in obj = {info.get('objective') - state['current_obj_val']:.3e}), stopping")
            state['info_list'].append(info)
            raise StopWhenFeasible(state['info_list'])
        else:
            # Degenerate iteration
            print("  -> degenerate iteration")
            state['info_list'].append(info)
            state['current_obj_val'] = info.get('objective')

    return edge_basis_callback, state

def make_list_callback():
    log = []
    def list_callback(info):
        print(f"iteration {info.get('nit')} (phase {info.get('phase')}) \t --> \t obj = {info.get('objective')}")
        log.append(info)
    return list_callback, log


def recover_start_vertex(x, P):
    """
    Simplex run with a big cost on every variable that is zero at x. It ends at the
    vertex whose zero pattern matches x (exactly x, if x is a vertex). Returns
    (x_rec, basis, T): an exact tableau vertex, its basis, and the tableau.
    Call ONCE per problem; every variant and the scipy baseline start from x_rec.
    """
    x = np.asarray(x, dtype=float)
    n = P.A.shape[1]
    active_inds = P.get_active_constraints(x)
    c_recover = np.zeros(n)
    c_recover[active_inds] = 1e12   # keep large: tol=1e-5 on reduced costs is absolute
    log = []
    # _linprog_simplex flips rows with b<0 in place, so hand it copies
    _linprog_simplex(c=c_recover, c0=0.0, A=P.A.copy(), b=P.b.copy(),
                     callback=log.append, postsolve_args=None, maxiter=5000)
    if not log:
        raise RuntimeError('callback_log is empty after start-vertex recovery.')
    last = log[-1]
    x_rec = np.asarray(last['x'], dtype=float)[:n].copy()
    basis = np.asarray(last['basis']).copy()
    T = np.asarray(last['T']).copy()

    resid = np.max(np.abs(P.A @ x_rec - P.b)) if P.A.shape[0] else 0.0
    if resid > 1e-6 * max(1.0, np.max(np.abs(P.b))):
        raise RuntimeError(f'Recovered vertex is not feasible (max residual {resid:.2e}).')
    leak = np.max(np.abs(x_rec[active_inds])) if len(active_inds) else 0.0
    if leak > 1e-8:
        print(f'  !! WARNING: recovery stopped early; active vars not all zero (max {leak:.2e}).')
    return x_rec, basis, T


def get_warmstart_info(rec, P):
    """
    rec = (x_rec, basis, T) from recover_start_vertex. Runs simplex from that vertex until
    the first edge step of Phase II and returns the two vertices bracketing that edge
    (x1/basis1, x2/basis2), needed for both the 'trivial' and 'edge' warm-start variants.
    """
    x_rec, last_basis, tableau = rec
    x_init = x_rec.copy()
    last_entry = {'x': x_rec, 'basis': last_basis, 'T': tableau}
    last_basis = last_basis.copy()   # _linprog_simplex_warmstart mutates basis in place

    print(f"[DEBUG] callback_log last entry keys: {list(last_entry.keys())}")
    print(f"[DEBUG] last_basis is None? {last_basis is None}, tableau is None? {tableau is None}")

    callback, state = make_edge_basis_callback()
    try:
        x_ws, status, message, nit = _linprog_simplex_warmstart(
            c=P.c, c0=0.0, A=P.A, b=P.b, T=tableau, basis=last_basis,
            callback=callback, postsolve_args=None, maxiter=5000,
            )
    except StopWhenFeasible as e:
        info_list = e.data
        print(f"[DEBUG] Warmstart caught StopWhenFeasible with len(info_list) = {len(info_list)}")
        info1, info2 = info_list[-2], info_list[-1]
    else:
        print('[DEBUG] Got to this else statement (warmstart completed normally)')
        if status == 1:
            raise RuntimeError('Edge step was never reached within maxiter (cycling?).')
        info_list = state['info_list']
        # info1 = info_list[0]
        # info2 = info_list[-1]   # same as info1 if len == 1
        print(f"[DEBUG] state['info_list'] length: {len(info_list)}")
        if len(info_list) > 0:
            info1 = info_list[0]
            info2 = info_list[-1]   # same as info1 if len == 1
        else:
            raise RuntimeError("state['info_list'] is empty after warmstart.")


    print(f'length of x1: {len(info1["x"])}')
    print(f'length of x_feasible: {len(x_init)}')
    print(f"[DEBUG] info1 basis is None? {info1.get('basis') is None}")
    print(f"[DEBUG] info2 basis is None? {info2.get('basis') is None}")
    return {
        'x1': info1['x'], 'obj1': info1['objective'],
        'iteration1': info1['nit'], 'basis1': info1['basis'],
        'x2': info2['x'], 'obj2': info2['objective'],
        'iteration2': info2['nit'], 'basis2': info2['basis'],
        'degenerate': (info1['x'] is info2['x']) or np.allclose(info1['x'], info2['x']),
        'x_start': x_init,
        }


def make_list_callback_withorgcomparison(P, c_original, verbose=False):
    log = []
    def list_callback_withorgcomparison(info):
        x_ = info["x"]
        obj_original = P.c @ x_[:len(c_original)]
        if verbose:
            print(
                f"Iter {info['nit']:4d} | "
                f"Phase {info['phase']} | "
                f"Dummy obj = {info['objective']:12.6f} | "
                f"Original obj = {obj_original:12.6f}"
            )
        info["obj_original"] = obj_original
        log.append(info)
    return list_callback_withorgcomparison, log

def estimate_rank(A):
    if A.shape[0] == 0:
        return None
    if issparse(A):
        try:
            A_inner = (A @ A.T).tocsc()
            splu(A_inner)
            return A.shape[0]
        except Exception:
            print("Matrix is rank deficient. Using randomized estimate...")
            from sklearn.utils.extmath import randomized_svd
            _, s, _ = randomized_svd(A, n_components=min(A.shape[0], 500))
            return int(np.sum(s > 1e-10))
    return np.linalg.matrix_rank(A)


def run_variant(cfg, P, rankA, x_feasible, edge_info):
    """
    Dispatches a single refined_sdac run according to its config dict.
    Returns the result object with rankA attached.
    """
    warm_start = cfg.get('warm_start')

    kwargs = dict(DEFAULT_SDAC_KWARGS)
    kwargs.update({k: v for k, v in cfg.items() if k not in ('name', 'enabled', 'warm_start')})
    kwargs['c'] = P.c
    kwargs['rankA'] = rankA
    kwargs['warm_start'] = warm_start

    if warm_start is None:
        x0 = np.array(x_feasible, dtype=float)   # private copy; SDAC updates x in place
    elif warm_start == 'trivial':
        x0 = np.array(x_feasible, dtype=float)   # private copy; SDAC updates x in place
        print(f'basis1 length: {len(edge_info["basis1"])}')
        kwargs['first_warm_start'] = edge_info['basis1']
    elif warm_start == 'edge':
        x0 = np.array(x_feasible, dtype=float)
        if not np.allclose(np.asarray(edge_info['x1'])[:len(x0)], x0, rtol=1e-7, atol=1e-7):
            print('  !! WARNING: edge x1 differs from the shared start vertex')
        kwargs['x2'] = edge_info['x2'].tolist()
        kwargs['first_warm_start'] = edge_info['basis1']
        kwargs['second_warm_start'] = edge_info['basis2']
    else:
        raise ValueError(f"Unknown warm_start type in config {cfg['name']!r}: {warm_start!r}")

    res = refined_sdac(P=P.copy(), x=x0, **kwargs)
    res.rankA = rankA
    return res


# =====================================================================
# Main loop
# =====================================================================

os.makedirs(RESULTS_DIR, exist_ok=True)

needs_warmstart_info = any(
    cfg['enabled'] and cfg.get('warm_start') in ('trivial', 'edge')
    for cfg in RUN_CONFIGS
)

edge_step_records = []  # per-problem edge-step info, if computed

for i, problem in enumerate(PROBLEMS, start=1):
    path = os.path.join(PROBLEM_DIR, problem)

    print(f'\nSolving: {problem}')
    info_list = []
    callback_log = []
    print(f'Working on problem {i} of {len(PROBLEMS)}')

    c, A, b = mps_to_standard(path)
    print(f"Standard form dimensions -> c: {c.shape}, A: {A.shape}")

    print('Building polyhedron in standard form...')
    P = PolyhedronStandard(A, b, c)
    rankA = estimate_rank(A) if P.m_A != 0 else None
    print(f'rank(A): {rankA}')

    print('Finding feasible solution for polyhedron in standard form...')
    x_gurobi = np.asarray(P.find_feasible_solution(basis=False), dtype=float)
    # Canonical start for EVERY variant and the scipy baseline: the exact tableau vertex
    # with the same zero pattern as Gurobi's point (see recover_start_vertex).
    rec = recover_start_vertex(x_gurobi, P)
    x_feasible = rec[0].copy()
    start_vert = x_feasible.copy()
    print(f'max |x_gurobi - x_start| = {np.max(np.abs(x_gurobi - x_feasible)):.3e}')

    warmstart_info = None
    if needs_warmstart_info:
        warmstart_info = get_warmstart_info(rec, P.copy())

        # --- validate recovered basis before it's used for any warm start ---
        m = P.A.shape[1]
        basis1_arr = np.asarray(warmstart_info['basis1'])
        if np.any(basis1_arr >= m):
            raise RuntimeError(
                f"Recovered basis1 for {problem} still contains artificial-variable "
                f"indices (>= {m}): {basis1_arr[basis1_arr >= m]}"
                )

        # if you're also using the 'edge' warm-start variant, check basis2 too
        if warmstart_info.get('basis2') is not None:
            basis2_arr = np.asarray(warmstart_info['basis2'])
            if np.any(basis2_arr >= m):
                raise RuntimeError(
                    f"Recovered basis2 for {problem} still contains artificial-variable "
                    f"indices (>= {m}): {basis2_arr[basis2_arr >= m]}"
                    )
        edge_step_records.append({'problem': problem, **{
            k: warmstart_info[k] for k in
            ('obj1', 'obj2', 'iteration1', 'iteration2')
        }})

    prefix = os.path.join(RESULTS_DIR, problem)

    if RUN_LP_BASELINE:
        print('\nSolving with simplex method (Gurobi)...')
        lp_result = P.solve_lp(verbose=False, record_objs=True)
        lp_result.save(prefix + '_lp.p')

    if RUN_SCIPY_BASELINE:
        print('\nSolving with scipy simplex...')
        # reuse the SAME recovered vertex/tableau every other variant starts from
        _, last_basis, tableau = rec
        last_basis = last_basis.copy()   # _linprog_simplex_warmstart mutates basis in place
        callback, callback_log = make_list_callback_withorgcomparison(P, P.c, verbose=False)

        scipy_start = time.time()
        _linprog_simplex_warmstart(
            c=P.c, c0=0.0, A=P.A, b=P.b, T=tableau, basis=last_basis,
            callback=callback, postsolve_args=None, maxiter=5000,
            )
        scipy_runtime = time.time() - scipy_start

        ###pull out only edge step information
        edge_tol = 1e-9
        edge_iterations = []
        edge_obj_values = [callback_log[0]['objective']]   # include the start vertex

        for i in range(1, len(callback_log)):
            obj_prev = callback_log[i - 1]["objective"]
            obj_curr = callback_log[i]["objective"]

            if abs(obj_curr - obj_prev) > edge_tol:
                edge_iterations.append(callback_log[i]["nit"])
                edge_obj_values.append(obj_curr)

        print(f"Edge steps: {len(edge_iterations)}")
        print("Occurred at iterations:", edge_iterations)


        scipy_result = result(
            0, x=callback_log[-1]['x'], obj=callback_log[-1]['objective'], obj_values = edge_obj_values,
            n_iters=len(edge_obj_values) - 1, solve_time = scipy_runtime,
        )
        print(f"Optimal objective after scipy simplex is: {callback_log[-1]['objective']}")
        scipy_result.save(prefix + '_scipy_lp.p')

    for cfg in RUN_CONFIGS:
        if not cfg['enabled']:
            continue
        print(f"\nSolving {problem}: variant '{cfg['name']}'...")
        res = run_variant(cfg, P, rankA, x_feasible, warmstart_info)
        ovals = getattr(res, 'obj_values', None)
        start_obj = float(P.c @ x_feasible)
        if ovals is not None and len(ovals) > 0 and not np.isclose(ovals[0], start_obj, rtol=1e-9, atol=1e-8):
            print(f"  !! WARNING: '{cfg['name']}' starts at obj {ovals[0]} != shared start obj {start_obj}")
        res.save(prefix + f"_{cfg['name']}_sd.p")

if edge_step_records:
    pd.DataFrame(edge_step_records).to_csv(
        os.path.join(RESULTS_DIR, 'edge_step_info.csv'), index=False
    )