import os
import sys
import csv
import pickle
import numpy as np
from matplotlib import pyplot as plt
from datetime import datetime

# =====================================================================
# CONFIG -- flip 'enabled' instead of commenting/uncommenting lines.
# =====================================================================

RESULTS_DIR = 'results'
FIGURE_DIR = 'figures'
ACTIVE_INDS_CSV_DIR = os.path.join(os.getcwd(), 'active_inds_csv')

# Which output(s) to produce this run.
RUN_STEP_PLOTS = True
RUN_ACTIVE_INDS_CSV = False
RUN_COMPARISON_TABLE = False
RUN_LP_SCIPY_COMPARISON = False
RUN_START_FINAL_COMPARISON = False
RUN_TIME_TABLE = True
RUN_STEP_TABLE = True   # step counts, ms/step, and when facet restriction fires

# All netlib problems, for reference -- copy names into PROBLEMS below.
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

# Subset actually run/plotted -- should match PROBLEMS in refined_netlib_runs.py.
# PROBLEMS = ['angela_test_prob', 'etamacro', 'network_ex', 'network_ex_2', 'recipe', 'shell']

### Which set of problems to view: 'netlib' or 'circulation'. Set this (and the circulation
### settings below) to match what you ran in refined_circ_runs.py.
PROBLEM_SET = 'circulation'

### Circulation instances (made by circulation.py). Results are saved under the file name without '.mps'.
CIRC_N_VALUES = [50]          # which n to view, e.g. [50, 100]
CIRC_PER_N = 10               # how many instances per n (1..10)
CIRC_PROBLEMS = [f'circ_n{n}_{k:02d}' for n in CIRC_N_VALUES for k in range(1, CIRC_PER_N + 1)]
###use (sub)set of circulation problems to view
# CIRC_PROBLEMS = ['circ_n50_01']

if PROBLEM_SET == 'circulation':
    PROBLEMS = CIRC_PROBLEMS
elif PROBLEM_SET != 'netlib':
    raise ValueError(f"PROBLEM_SET must be 'netlib' or 'circulation', not {PROBLEM_SET!r}")

# Every SDAC variant you might want to plot. 'key' must match the
# 'name' used in refined_netlib_runs.py's RUN_CONFIGS (filenames are
# '{problem}_{key}_sd.p'). Flip 'enabled' per variant; nothing else
# needs to change to add/remove a line from the plots/CSVs/table.
SD_METHODS = [
    dict(key='sdac', enabled=True, color='blue', linestyle='-',
         label='steepest-descent'),
    # dict(key='sdac_edge', enabled=True, color='red', linestyle='-',
         # label='edge warm-start'),
    # dict(key='sdac_triv', enabled=True, color='purple', linestyle='-',
         # label='trivial warm-start'),
    dict(key='sdac_rest', enabled=True, color='pink', linestyle='-',
         label='facet rest'),
    # dict(key='sdac_triv_rest', enabled=True, color='royalblue', linestyle='-',
    #      label='trivial warm-start facet rest'),
    # dict(key='sdac_edge_rest', enabled=True, color='darkorange', linestyle='-',
    #      label='edge warm-start facet rest'),
]

# =====================================================================
# Logging setup
# =====================================================================

os.makedirs('./logs', exist_ok=True)
log_path = f"./logs/active_inds_{datetime.now():%Y%m%d_%H%M%S}.txt"
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

# =====================================================================
# Helpers
# =====================================================================


def enabled_methods():
    return [m for m in SD_METHODS if m['enabled']]


def load_pickle(path):
    if not os.path.exists(path):
        return None
    with open(path, 'rb') as f:
        return pickle.load(f)


def load_lp(problem):
    return load_pickle(os.path.join(RESULTS_DIR, f'{problem}_lp.p'))

def load_scipy(problem):
    return load_pickle(os.path.join(RESULTS_DIR, f'{problem}_scipy_lp.p'))


def load_sd(problem, method):
    """method is one of the dicts in SD_METHODS."""
    return load_pickle(os.path.join(RESULTS_DIR, f"{problem}_{method['key']}_sd.p"))


def print_summary_rows(columns, label_width=12, col_width=12, fmt='.1f', geo=True):
    """Print Mean / Median / Geo mean rows under a table.

    columns : list of lists, one list of per-problem values per table column
              (None entries = missing data, skipped).
    Geo mean uses only values > 0 (e.g. Gurobi's 0.0000 times are skipped).
    Prints a note if columns were averaged over different numbers of problems,
    since their averages are then not directly comparable.
    """
    def _stat(vals, kind):
        vals = [v for v in vals if v is not None]
        if kind == 'geo':
            vals = [v for v in vals if v > 0]
        if not vals:
            return None
        if kind == 'mean':
            return float(np.mean(vals))
        if kind == 'median':
            return float(np.median(vals))
        return float(np.exp(np.mean(np.log(vals))))

    print('-' * (label_width + (col_width + 1) * len(columns)))
    kinds = [('Mean', 'mean'), ('Median', 'median')] + ([('Geo mean', 'geo')] if geo else [])
    for label, kind in kinds:
        row = f'{label:<{label_width}}'
        for vals in columns:
            v = _stat(vals, kind)
            row += f' {"n/a":>{col_width}}' if v is None else f' {v:>{col_width}{fmt}}'
        print(row)

    counts = [sum(v is not None for v in vals) for vals in columns]
    geo_counts = [sum(v is not None and v > 0 for v in vals) for vals in columns]
    if len(set(counts)) > 1:
        print(f'  note: columns have different numbers of problems with data: {counts}')
    if geo and geo_counts != counts:
        print(f'  note: geo mean skips values <= 0; problems used per column: {geo_counts}')


# =====================================================================
# 1. Objective-vs-step plots
# =====================================================================

if RUN_STEP_PLOTS:
    os.makedirs(FIGURE_DIR, exist_ok=True)

    for problem in PROBLEMS:
        ###gurobi lp objective values
        # lp_data = load_lp(problem)
        # if lp_data is None:
        #     continue
        # lp_obj = lp_data['obj']

        # fig, ax = plt.subplots(figsize=(6, 4))
        # ax.set_title(f'Objective vs. step for {problem}')
        # ax.axhline(y=lp_obj, color='black', linestyle='--', linewidth=1, label='LP optimal')

        # lp_objs = lp_data['obj_values']
        # ax.plot(range(len(lp_objs)), lp_objs, color='green', marker='o',
        #         markersize=3, label='simplex')

        ###scipy lp objective values
        scipy_data = load_scipy(problem)
        if scipy_data is None:
            continue
        scipy_obj = scipy_data['obj']

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.set_title(f'Objective vs. step for {problem}')
        ax.axhline(y=scipy_obj, color='black', linestyle='--', linewidth=1, label='LP optimal')

        scipy_objs = scipy_data['obj_values']
        ax.plot(range(len(scipy_objs)), scipy_objs, color='green', marker='o',
                markersize=3, label='custom simplex')

        for method in enabled_methods():
            sd_data = load_sd(problem, method)
            if sd_data is None:
                continue
            sd_objs = sd_data['obj_values']
            ax.plot(range(len(sd_objs)), sd_objs, color=method['color'],
                     linestyle=method['linestyle'], marker='o', markersize=3,
                     label=method['label'])

        ax.set_xlabel('Step')
        ax.set_ylabel('Obj Value')
        ax.legend()
        plt.tight_layout()
        plt.savefig(f'{FIGURE_DIR}/step_comparison_facet_rest_{problem}.png', dpi=150)
        plt.close(fig)

# =====================================================================
# 2. Active-indices CSVs
# =====================================================================

if RUN_ACTIVE_INDS_CSV:
    os.makedirs(ACTIVE_INDS_CSV_DIR, exist_ok=True)

    for problem in PROBLEMS:
        for method in enabled_methods():
            sd_data = load_sd(problem, method)
            if sd_data is None:
                continue

            history_raw = sd_data.get('active_inds', [])
            step_types = sd_data.get('step_type', [])
            steepness = sd_data.get('steepness', [])
            obj_values = sd_data.get('obj_values', [])
            step_sizes = sd_data.get('steps', [])
            rankA = sd_data.get('rankA', None)
            size = len(sd_data['x']) if sd_data.get('x') is not None else None

            history = [set(s) for s in history_raw]  # may be shorter than obj_values
            n = len(obj_values)  # drive the loop off the longest/most-complete list
            if not n:
                continue

            safe_label = method['label'].replace(' ', '_').replace('/', '-')
            csv_fn = os.path.join(ACTIVE_INDS_CSV_DIR, f'{problem}_{safe_label}_active_inds.csv')
            with open(csv_fn, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(['step', 'step_type', 'size', 'step_size', 'steepness',
                                  'obj_value', 'rankA', 'total', 'added_inds', 'removed_inds'])
                for i in range(n):
                    stype = step_types[i] if i < len(step_types) else None
                    stype = 'restricted' if stype == 'facet' else stype
                    step_sz = step_sizes[i] if i < len(step_sizes) else None
                    steep = steepness[i] if i < len(steepness) else None
                    obj_val = obj_values[i] if i < len(obj_values) else None

                    if i < len(history):
                        s = history[i]
                        added = sorted(s - history[i - 1]) if i > 0 else []
                        removed = sorted(history[i - 1] - s) if i > 0 else []
                        total = len(s)
                    else:
                        # no new active-inds data for this step (e.g. simplex-finish rows)
                        s = added = removed = total = None

                    writer.writerow([i, stype, size, step_sz, steep, obj_val, rankA,
                                      total, added, removed])
            print(f'Saved: {csv_fn}')

# =====================================================================
# 3. Objective-value comparison table
# =====================================================================

if RUN_COMPARISON_TABLE:
    print(f'\n{"Problem":<15} {"Method":<45} {"Final Obj":>15} {"LP Obj":>15} {"Match?":>8}')
    print('-' * 105)

    for problem in PROBLEMS:
        lp_data = load_lp(problem)
        if lp_data is None:
            continue
        lp_obj = lp_data['obj']

        for method in enabled_methods():
            sd_data = load_sd(problem, method)
            if sd_data is None:
                continue

            sd_obj = sd_data['obj_values'][-1]
            match = np.isclose(sd_obj, lp_obj, rtol=1e-5, atol=1e-8)
            match_str = 'YES' if match else 'NO'

            print(f"{problem:<15} {method['label']:<45} {sd_obj:>15.6f} {lp_obj:>15.6f} {match_str:>8}")

# =====================================================================
# 4. LP vs scipy objective comparison
# =====================================================================

if RUN_LP_SCIPY_COMPARISON:
    print(f'\n{"Problem":<15} {"LP Obj":>15} {"Scipy Obj":>15} {"Match?":>8}')
    print('-' * 60)

    for problem in PROBLEMS:
        lp_data = load_lp(problem)
        scipy_data = load_scipy(problem)

        if lp_data is None or scipy_data is None:
            print(f"{problem:<15} {'missing data':>15}")
            continue

        lp_obj = lp_data['obj']
        scipy_obj = scipy_data['obj']
        match = np.isclose(sd_obj := scipy_obj, lp_obj, rtol=1e-5, atol=1e-8)
        match_str = 'YES' if match else 'NO'

        print(f"{problem:<15} {lp_obj:>15.6f} {scipy_obj:>15.6f} {match_str:>8}")

# =====================================================================
# 5. SDAC vs scipy: starting and final objective values
# =====================================================================

if RUN_START_FINAL_COMPARISON:
    def _fmt(v):
        return f'{v:>15.6f}' if v is not None else f'{"n/a":>15}'

    def _diff(a, b):
        return a - b if (a is not None and b is not None) else None

    def _close(a, b):
        if a is None or b is None:
            return 'n/a'
        return 'YES' if np.isclose(a, b, rtol=1e-5, atol=1e-8) else 'NO'

    for method in enabled_methods():
        print(f"\n=== Start/final objective: {method['label']} vs. scipy ===")
        header = (f'{"Problem":<12} {"Scipy Start":>15} {"SD Start":>15} {"Diff":>12} {"Same?":>6}'
                  f' | {"Scipy Final":>15} {"SD Final":>15} {"Diff":>12} {"Same?":>6}')
        print(header)
        print('-' * len(header))

        for problem in PROBLEMS:
            scipy_data = load_scipy(problem)
            sd_data = load_sd(problem, method)
            if scipy_data is None or sd_data is None:
                print(f'{problem:<12} {"missing data":>15}')
                continue

            scipy_objs = scipy_data.get('obj_values', [])
            sd_objs = sd_data.get('obj_values', [])

            scipy_start = scipy_objs[0] if len(scipy_objs) else None
            sd_start = sd_objs[0] if len(sd_objs) else None
            scipy_final = scipy_data.get('obj', scipy_objs[-1] if len(scipy_objs) else None)
            sd_final = sd_objs[-1] if len(sd_objs) else None

            d_start = _diff(sd_start, scipy_start)
            d_final = _diff(sd_final, scipy_final)
            d_start_s = f'{d_start:>12.4e}' if d_start is not None else f'{"n/a":>12}'
            d_final_s = f'{d_final:>12.4e}' if d_final is not None else f'{"n/a":>12}'

            print(f'{problem:<12} {_fmt(scipy_start)} {_fmt(sd_start)} {d_start_s} '
                  f'{_close(sd_start, scipy_start):>6}'
                  f' | {_fmt(scipy_final)} {_fmt(sd_final)} {d_final_s} '
                  f'{_close(sd_final, scipy_final):>6}')

# =====================================================================
# 6. Total solve-time table
# =====================================================================

if RUN_TIME_TABLE:
    def _t(data):
        if data is None or data.get('solve_time_total') is None:
            return f'{"n/a":>14}'
        return f"{data['solve_time_total']:>14.4f}"

    methods = enabled_methods()
    header = f'{"Problem":<12} {"Gurobi LP":>14} {"Scipy":>14}' + ''.join(
        f" {m['key']:>14}" for m in methods)
    print('\n=== Total solve time (s) ===')
    print(header)
    print('-' * len(header))

    def _tv(data):
        return None if data is None else data.get('solve_time_total')

    time_cols = [[] for _ in range(2 + len(methods))]
    for problem in PROBLEMS:
        lp_d, sp_d = load_lp(problem), load_scipy(problem)
        sd_ds = [load_sd(problem, method) for method in methods]
        row = f'{problem:<12} {_t(lp_d)} {_t(sp_d)}'
        for d in sd_ds:
            row += f' {_t(d)}'
        print(row)
        for col, d in zip(time_cols, [lp_d, sp_d] + sd_ds):
            col.append(_tv(d))
    print_summary_rows(time_cols, label_width=12, col_width=14, fmt='.4f')


# =====================================================================
# 7. Step-count table (matches the x-axis of the objective-vs-step plots)
# =====================================================================

if RUN_STEP_TABLE:
    def _n_steps(data):
        """Steps taken = len(obj_values) - 1, because obj_values[0] is the
        starting point. This equals the last x-value in the step plots."""
        if data is None:
            return None
        objs = data.get('obj_values')
        if objs is None or len(objs) == 0:
            return None
        return len(objs) - 1

    def _ms_per_step(data, n):
        if data is None or not n or data.get('solve_time_total') is None:
            return None
        return 1000.0 * data['solve_time_total'] / n

    def _restricted_info(data):
        """(entry step, exit step) for facet restriction, aligned with obj_values.
        Uses fr_entry_step / fr_exit_step if the pickle has them; otherwise
        (older pickles) infers them from step_type: entry = first 'facet'
        index, exit = last 'facet' index.
        Returns (-1, -1) if restriction never fired, (None, None) if no data."""
        if data is None:
            return None, None
        if 'fr_entry_step' in data:
            entry, exit_ = data['fr_entry_step'], data['fr_exit_step']
            return (-1, -1) if entry is None else (entry, exit_)
        if not data.get('step_type'):
            return None, None
        idx = [i for i, t in enumerate(data['step_type']) if t == 'facet']
        return (idx[0], idx[-1]) if idx else (-1, -1)

    def _cell(v, fmt='d', width=12):
        if v is None:
            return f'{"n/a":>{width}}'
        if fmt == 'd' and v == -1:
            return f'{"never":>{width}}'
        return f'{v:>{width}{fmt}}'

    methods = enabled_methods()

    # --- Table A: number of steps ---
    header = f'{"Problem":<12} {"Scipy":>12}' + ''.join(f" {m['key']:>12}" for m in methods)
    print('\n=== Number of steps (last x-value in the step plots) ===')
    print(header)
    print('-' * len(header))

    per_problem = {}
    for problem in PROBLEMS:
        scipy_data = load_scipy(problem)
        entries = [('scipy', scipy_data)] + [(m['key'], load_sd(problem, m)) for m in methods]
        per_problem[problem] = entries

        row = f'{problem:<12}'
        for key, data in entries:
            n = _n_steps(data)
            row += f' {_cell(n)}'
        print(row)

    print_summary_rows([[_n_steps(entries[j][1]) for entries in per_problem.values()]
                        for j in range(1 + len(methods))], fmt='.1f')

    # --- Table B: time per step ---
    print('\n=== Time per step (ms) = solve_time_total / steps ===')
    print(header)
    print('-' * len(header))
    for problem, entries in per_problem.items():
        row = f'{problem:<12}'
        for key, data in entries:
            row += f' {_cell(_ms_per_step(data, _n_steps(data)), ".1f")}'
        print(row)

    print_summary_rows([[_ms_per_step(entries[j][1], _n_steps(entries[j][1]))
                         for entries in per_problem.values()]
                        for j in range(1 + len(methods))], fmt='.1f')

    # --- Table C: facet-restriction switch points (SD methods only) ---
    # entry     = step whose landing point triggered restriction
    #             (first restricted move is entry + 1)
    # moves     = restricted moves actually taken (exit - entry)
    # simplex   = custom-simplex steps after exit (non-degenerate pivots only)
    print('\n=== Facet restriction: entry step / restricted moves / simplex steps after exit ===')
    for m_i, method in enumerate(methods, start=1):
        rows = []
        for problem, entries in per_problem.items():
            data = entries[m_i][1]
            entry, exit_ = _restricted_info(data)
            n = _n_steps(data)
            if entry is None:
                rows.append((problem, None, None, None, None))
            elif entry == -1:
                rows.append((problem, -1, None, None, n))
            else:
                simp = (n - exit_) if (n is not None and exit_ is not None) else None
                moves = (exit_ - entry) if exit_ is not None else None
                rows.append((problem, entry, moves, simp, n))

        if not any(r[1] is not None and r[1] >= 0 for r in rows):
            print(f"\n[{method['key']}] facet restriction never fired")
            continue

        hdr = f'{"Problem":<12} {"entry":>8} {"moves":>8} {"simplex":>8} {"total":>8}'
        print(f"\n[{method['key']}]")
        print(hdr)
        print('-' * len(hdr))
        for problem, entry, moves, simp, n in rows:
            print(f'{problem:<12} {_cell(entry, width=8)} {_cell(moves, width=8)} '
                  f'{_cell(simp, width=8)} {_cell(n, width=8)}')

        fired = [r for r in rows if r[1] is not None and r[1] >= 0]
        print(f'  (summary over the {len(fired)} problems where restriction fired)')
        print_summary_rows([[r[k] for r in fired] for k in range(1, 5)],
                           col_width=8, fmt='.1f', geo=False)