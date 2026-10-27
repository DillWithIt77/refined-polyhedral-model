"""
refined_sdac.py

IMPORTANT ASSUMPTION: `refined_sdac` assumes the input polyhedron P is
already given in standard form, i.e.

    P = { x : P.A @ x = P.b, x >= 0 }

`P.A`, `P.b`, and `P.c` are passed directly into `PolyhedronStandard`
with no conversion step, so if P is not already in standard form
(e.g. it still has inequality constraints, free variables, or bounds
other than x >= 0), the results of this function are not meaningful.
Convert P to standard form before calling this function.
"""

import numpy as np
import gurobipy as gp
import math
import sympy
import time
import pulp
### import from other .py files for steepest-descent implementation 
from utils import result, METHODS, INF, EPS
from refined_polyhedral_model import PolyhedralModelStandard
from refined_polyhedron import PolyhedronStandard
from custom_simplex_warmstart import _linprog_simplex, _linprog_simplex_warmstart

class StopWhenFeasible(Exception):
    def __init__(self, data):
        self.data = data

def basis_callback(info):
    print(f"iteration {info.get('nit')} (phase {info.get('phase')}) \t --> \t obj = {info.get('objective')}")
    if info.get('phase') == 2:
        # Stop after finding a feasible solution
        raise StopWhenFeasible(info)

def refined_sdac(P, x, c, x2=None, warm_start=None, first_warm_start=None, second_warm_start=None,
               verbose=False, method='dual_simplex', reset=False, max_time=1200,
               rankA=None, facet_restriction=True, facet_rest_type=None, percent=100,
               save_first_steps=0, problem_name=''):
    """
    Parameters
    ----------
    P : Polyhedron
        Original problem (used for P.c in objective bookkeeping / vertex checks).
        Assumed to already be in standard form (P.A @ x = P.b, x >= 0) --
        see the module docstring for details.
    x : ndarray
        Starting point (feasible solution of P_new).
    c : ndarray or None
        Objective to use on P_new (P_new.set_objective(c) if not None).
    x2 : ndarray, optional
        Second point, only needed when warm_start='edge'.
    warm_start : {None, 'custom', 'trivial', 'edge'}, optional
        Which warm-start mechanism to use on the polyhedral model:
          - None      : no warm start (default steepest-descent behavior)
          - 'trivial' : pm.set_trivial_basis(x=x, basis_array=first_warm_start)
          - 'edge'    : pm.set_edge_basis(x=x, x2=x2,
                        basis_array1=first_warm_start, basis_array2=second_warm_start)
                        (requires both x2 and second_warm_start)
    first_warm_start : ndarray, optional
        Basis-column array (or solution, for 'custom') used to warm-start.
    second_warm_start : ndarray, optional
        Second basis-column array, only used when warm_start='edge'.
    facet_restriction : bool, optional
        If False, the facet-restriction state machine never activates,
        regardless of `percent`/`facet_rest_type`. Default True (matches the
        original functions' behavior, gated internally by `percent`).
    facet_rest_type : {'initial', 'active', None}, optional
        How restriction indices are chosen once inside a restriction episode.
    percent : as before.
        `percent` controls how many active constraints must be lost before
        entering a restriction episode (used only if facet_restriction=True).
    Other parameters are unchanged from the originals.
    """
    ###collects all active inds throughout the run
    all_active_inds = []

    P_new = PolyhedronStandard(P.A, P.b, P.c)
    if c is not None:
        P_new.set_objective(c)

    t0 = time.time()
    x_current = np.array(x, dtype=float)   # take_maximal_step updates in place; never alias the caller's x
    if save_first_steps:
        np.save('solutions/{}_0.npy'.format(problem_name), x_current)
    active_inds = P_new.get_active_constraints(x_current)
    all_active_inds.append(active_inds)

    pm = P_new.build_polyhedral_model(active_inds=active_inds, method=method, verbose=verbose)

    ###warm-start dispatch
    ws_start = time.time()
    if warm_start == 'trivial' and first_warm_start is not None:
        print('Using trivial warm start')
        pm.set_trivial_basis(x=x_current, basis_array=first_warm_start)
    elif warm_start == 'edge' and first_warm_start is not None and second_warm_start is not None:
        print('Using edge warm start')
        print('x1==x2:', np.allclose(x_current, x2),
            'basis1 len:', len(first_warm_start), 'basis2 len:', len(second_warm_start),
            'basis1:', first_warm_start, 'basis2:', second_warm_start)
        pm.set_edge_basis(x=x_current, x2=x2, basis_array1=first_warm_start, basis_array2=second_warm_start)
    elif warm_start is not None:
        raise ValueError(
            "Invalid warm_start configuration: warm_start={!r} requires the matching "
            "first_warm_start (and second_warm_start + x2 for 'edge') to be provided."
            .format(warm_start)
        )
    ws_runtime = time.time() - ws_start

    t1 = time.time()
    build_time = (t1 - t0) - ws_runtime
    print('Polyhedral model build time: {}'.format(build_time))

    sub_times = {'sd': [], 'step': [], 'solve': [], 'phase_times': []}
    descent_circuits = []
    obj_values = []
    step_sizes = []
    iter_times = []
    simplex_iters = []
    steeps = []
    iteration = 0
    obj_value = P_new.c.dot(x_current)
    obj_values.append(obj_value)
    t2 = time.time()
    iter_times.append(t2 - t1)

    # compute steepest-descent direction
    y_pos, y_neg, steepness, num_steps, solve_time, phase_times = pm.compute_sd_direction(verbose=verbose)
    simplex_iters.append(num_steps)
    sub_times['solve'].append(solve_time)
    sub_times['phase_times'].append(phase_times)
    steeps.append(steepness)

    t3 = time.time()
    sub_times['sd'].append(t3 - t2)

    ###inds for facet restriction
    rest_inds = None
    ###whether we are currently inside a facet-restriction episode
    in_restriction = False
    ###check if already captured initial inds for the current facet restriction episode
    set_init_inds = False
    ###restriction step tracker (kept for logging / downstream plotting use)
    rest_track = [None]
    ###initial number of active inds:
    start_act_inds = len(active_inds)
    ###step indices (aligned with obj_values) for when facet restriction starts/ends:
    ###  fr_entry_step: the step whose landing point triggered the entry condition
    ###                 (first restricted move is fr_entry_step + 1)
    ###  fr_exit_step:  last step before custom simplex takes over
    ###  Both stay None if facet restriction never fires.
    fr_entry_step = None
    fr_exit_step = None
    ###timer stuff
    post_fr_runtime = 0
    end_simp_runtime = 0

    while abs(steepness) > EPS:

        t3 = time.time()
        if reset:
            pm.reset()

        # take maximal step
        x_current, alpha, active_inds = P_new.take_maximal_step(y_pos, y_neg)
        all_active_inds.append(active_inds)

        obj_value = P_new.c.dot(x_current)
        obj_values.append(obj_value)
        if iteration % 50 == 0 or iteration == 1:
            print('\nIteration {}'.format(iteration))
            print('Objective: {}'.format(obj_value))
            print('Steepness: {}'.format(steepness))
            print('Step length: {}'.format(alpha))

        t4 = time.time()
        iter_times.append(t4 - t1)
        sub_times['step'].append(t4 - t3)
        descent_direction = y_pos - y_neg
        descent_circuits.append(descent_direction)
        step_sizes.append(alpha)

        if math.isinf(alpha):
            # problem is unbounded
            return result(status=1, circuits=descent_circuits, steps=step_sizes)

        ### ----- facet restriction state machine -----
        # ENTRY: only evaluated while NOT currently restricted, and only if
        # facet restriction is enabled at all.
        if facet_restriction and not in_restriction:
            should_enter = (len(active_inds) <= start_act_inds * (1 - (percent) / 100))
            if should_enter:
                fr_entry_step = len(obj_values) - 1
                print(f'------ Entered Facet Restriction at step {fr_entry_step}! ------')
                in_restriction = True
                set_init_inds = False  # fresh episode -> re-anchor 'initial' mode

        # Once inside a restriction episode, (re)compute which inds to restrict to.
        if in_restriction:
            if facet_rest_type == 'initial' and not set_init_inds:
                rest_inds = active_inds
                set_init_inds = True
            elif facet_rest_type == 'active':
                rest_inds = active_inds
            pm.set_active_inds(active_inds, rest_inds=rest_inds)
        else:
            rest_inds = None
            pm.set_active_inds(active_inds)

        rest_track.append(rest_inds if in_restriction else None)

        ###compute next circuit step (restricted, if in_restriction)
        y_pos, y_neg, steepness, num_steps, solve_time, phase_times = pm.compute_sd_direction(verbose=verbose)
        steeps.append(steepness)

        # EXIT: only evaluated while currently restricted. Triggered when the
        # restricted steepest-descent direction itself is ~0, i.e. this face is exhausted.
        if in_restriction and abs(steepness) <= EPS:
            fr_exit_step = len(obj_values) - 1
            print(f'++++++ Exited Facet Restriction at step {fr_exit_step}! ++++++')
            post_fr_start = time.time()
            ##check if the current solution x_current is a vertex of the original polyhedron (not polyhedral model)
            if verbose:
                is_vertex = P_new.is_vertex()
                print(f'Is our current solution a vertex?: {is_vertex}')

            ###custom simplex warm-start
            ###run to get info necessary for warm-start
            callback_log = []

            def list_callback(info):
                callback_log.append(info)

            def list_callback_withorgcomparison(info):
                x_ = info["x"]
                obj_original = P.c @ x_[:len(c_original)]
                if verbose:
                    print(f"Iter {info['nit']:4d} | "f"Phase {info['phase']} | "f"Dummy obj = {info['objective']:12.6f} | "f"Original obj = {obj_original:12.6f}")
                info["obj_original"] = obj_original
                callback_log.append(info)

            c_original = P_new.c.copy()
            c_recover = P_new.c.copy()
            c_recover[:] = 1e12
            n = P_new.A.shape[1]
            mask = np.ones(n, dtype=bool)
            mask[active_inds] = False
            non_active_inds = np.where(mask)[0]
            c_recover[non_active_inds] = 0.0

            try:
                _linprog_simplex(
                    c=c_recover,
                    c0=0.0,
                    A=P_new.A,
                    b=P_new.b,
                    callback=list_callback,
                    postsolve_args=None,
                    maxiter=5000
                )
            except StopWhenFeasible as e:
                info = e.data

                iteration = info['nit']
                x = info['x']
                obj = info['objective']
                basis = info['basis']

                temp_results.append({
                    'problem': problem,
                    'objective': obj,
                    'x': x,
                    'iterations': iteration,
                    'basis': basis,
                    'T': T
                })
            post_fr_runtime = time.time() - post_fr_start

            ###warm-starting using info
            last_basis = callback_log[-1]["basis"]
            tableau = callback_log[-1]["T"]
            callback_log = []

            end_simp_start = time.time()
            try:
                _linprog_simplex_warmstart(
                    c=P_new.c,
                    c0=0.0,
                    A=P_new.A,
                    b=P_new.b,
                    T=tableau,
                    basis=last_basis,
                    callback=list_callback_withorgcomparison,
                    postsolve_args=None,
                    maxiter=5000
                )
            except StopWhenFeasible as e:
                info = e.data

                iteration = info['nit']
                x = info['x']
                obj = info['objective']
                basis = info['basis']

                results.append({
                    'problem': problem,
                    'objective': obj,
                    'x': x,
                    'iterations': iteration,
                    'basis': basis,
                    'T': T
                })

            end_simp_runtime = time.time() - end_simp_start
            ###counting edge steps (probs need to save this info like I was for gurobi)
            edge_tol = 1e-9
            edge_iterations = []
            edge_obj_values = []

            for i in range(1, len(callback_log)):
                obj_prev = callback_log[i - 1]["objective"]
                obj_curr = callback_log[i]["objective"]

                if abs(obj_curr - obj_prev) > edge_tol:
                    edge_iterations.append(callback_log[i]["nit"])
                    edge_obj_values.append(obj_curr)

            if verbose:
                print(f"Edge steps: {len(edge_iterations)}")
                print("Occurred at iterations:", edge_iterations)
            obj_values.extend(edge_obj_values)
            rest_track.extend(['simplex'] * len(edge_obj_values))

            break

        if in_restriction and verbose:
            print(f'After computing sd direction...')
            print(f'facet restriction on: {rest_inds is not None}, steepness value: {abs(steepness)}')

        t5 = time.time()
        sub_times['sd'].append(t5 - t4)
        sub_times['solve'].append(solve_time)
        sub_times['phase_times'].append(phase_times)
        simplex_iters.append(num_steps)

        iteration += 1
        current_time = t5 - t1
        ###if we are taking longer than we should be, exit sdac
        if current_time > max_time:
            print('Reached maximum time')
            return result(status=2)
        if iteration <= save_first_steps:
            np.save('solutions/{}_{}.npy'.format(problem_name, iteration), x_current)

    t6 = time.time()
    total_time = t6 - t1 - post_fr_runtime
    print('Total time for steepest-descent scheme: {}'.format(total_time))
    print(f'Final sdac objective value: {obj_values[-1]}')
    print(f'Final steepness value: {steepness}')

    return result(status=0, x=x_current,
                  obj=P.c.dot(x_current), n_iters=len(step_sizes), solve_time=total_time,
                  iter_times=iter_times, alg_type='steepest-descent',
                  circuits=descent_circuits, steps=step_sizes,
                  simplex_iters=simplex_iters, solve_times=sub_times['solve'],
                  sub_times=sub_times, obj_values=obj_values,
                  step_type=['simplex' if r == 'simplex' else ('facet' if r is not None else None) for r in rest_track],
                  active_inds=all_active_inds, steepness=steeps,
                  fr_entry_step=fr_entry_step, fr_exit_step=fr_exit_step)