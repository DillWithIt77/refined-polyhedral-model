import numpy as np
from scipy.sparse import issparse
import sympy
import gurobipy as gp
import contextlib
import time
from fractions import Fraction

from refined_polyhedral_model import PolyhedralModelStandard
from utils import result, EPS, INF, METHODS


# class for representing a standard form polyhedron:
# P = {x in R^n : Ax = b, x >=0 }, with objective c
class PolyhedronStandard:

    # initiallize with matrices and vectors given by numpy arrays
    def __init__(self, A=None, b=None, c=None):
        self.A = A
        self.b = b
        self.c = c

        self.m_A, self.n = self.A.shape
        self.model = None
        print('Problem size: n = {},  m_A = {}'.format(self.n, self.m_A))

    # construct polyhedral model for computing circuits
    def build_polyhedral_model(self, active_inds=[], primal=True, method='dual_simplex', alt=False, verbose=False):
        pm = PolyhedralModelStandard(A=self.A, c=self.c, active_inds=active_inds, primal=primal,
                                      method=method, alt=alt, verbose=verbose)
        return pm

    # set current problem solution and get active constraints
    def get_active_constraints(self, x, alt=False):
        inds = []
        for i in range(self.n):
            if x[i] <= EPS:
                inds.append(i)
        self.x_current = x

        # an array of booleans it seems
        self.active_inds = [False] * self.n
        for i in inds:
            self.active_inds[i] = True
        return inds

    # use saved information about active facets to compute maximal step size along given direction
    def take_maximal_step(self, y_pos, y_neg):
        assert hasattr(self, 'x_current') and hasattr(self, 'active_inds')
        g = y_pos - y_neg
        alpha = float('inf')
        stopping_inds = []
        for i in range(self.n):
            if abs(g[i]) <= EPS:
                continue
            elif g[i] < 0:
                if self.active_inds[i]:
                    continue
                a = abs(self.x_current[i] / g[i])
                if abs(alpha - a) < EPS:
                    self.active_inds[i] = True
                    stopping_inds.append(i)
                elif a < alpha:
                    alpha = a
                    for j in stopping_inds:
                        self.active_inds[j] = False
                    stopping_inds = [i]
                    self.active_inds[i] = True
            elif g[i] > 0:
                self.active_inds[i] = False

        # take step with size alpha
        if alpha < EPS:
            print('Degenerate step computed. Changing active facets...')
            for i in stopping_inds:
                self.active_inds[i] = True
        self.x_current += alpha * g

        # return solution, step size, and list of active constraints
        return self.x_current, alpha, [i for i in range(self.n) if self.active_inds[i]]

    # build a gurobi LP for the polyhedron
    def build_gurobi_model(self, c=None, verbose=False, method='primal_simplex'):
        if c is None:
            c = self.c
        assert c is not None, 'Provide an objective function'

        self.model = gp.Model()
        self.model.setParam('OutputFlag', 0 if not verbose else 1)
        self.model.setParam(gp.GRB.Param.Presolve, 0)
        self.x = []
        for i in range(self.n):
            self.x.append(self.model.addVar(lb=0, name='x_{}'.format(i)))

        if issparse(self.A):
            for i in range(self.m_A):
                # Extract the nonzero elements in row `i`
                start_idx, end_idx = self.A.indptr[i], self.A.indptr[i + 1]

                # Get the nonzero values and corresponding column indices
                row_values = self.A.data[start_idx:end_idx]
                row_columns = self.A.indices[start_idx:end_idx]

                expr = gp.LinExpr()
                expr.addTerms(row_values, [self.x[j] for j in row_columns])

                self.model.addConstr(expr == self.b[i], name='A_{}'.format(i))
        else:
            for i in range(self.m_A):
                self.model.addConstr(gp.LinExpr(self.A[i], self.x) == self.b[i], name='A_{}'.format(i))

        self.set_objective(c)
        self.set_verbose(verbose)
        self.set_method(method)
        self.model.update()

    # (re)set objective function
    def set_objective(self, c):
        self.c = c
        if self.model is not None:
            self.model.setObjective(gp.LinExpr(self.c, self.x))

    # change model verbose settings
    def set_verbose(self, verbose):
        flag = 1 if verbose else 0
        self.model.setParam('OutputFlag', flag)
        # Also disable the log file to be safe
        self.model.setParam('LogFile', "")

    # set lp solve method
    def set_method(self, method):
        self.method = method
        with contextlib.redirect_stdout(None):
            self.model.Params.method = METHODS[method]

    # find a feasible solution in the polyhedron
    def find_feasible_solution(self, basis=False, verbose=False):
        c_orig = np.copy(self.c)
        c = np.zeros(self.n)
        if self.model is None:
            self.build_gurobi_model(c, verbose)
        else:
            self.set_objective(c)

        self.model.optimize()

        if self.model.status != gp.GRB.Status.OPTIMAL:
            raise RuntimeError('Failed to find feasible solution.')

        self.set_objective(c_orig)
        x_feasible = self.model.getAttr('x', self.x)
        vbasis = self.model.getAttr(gp.GRB.Attr.VBasis)
        cbasis = self.model.getAttr(gp.GRB.Attr.CBasis)
        if basis:
            return x_feasible, vbasis, cbasis
        else:
            return x_feasible

    # solve linear program using the given method
    def solve_lp(self, c=None, verbose=False, record_objs=True, method='primal_simplex', pstart=None):

        if c is None:
            assert self.c is not None, 'Need objective function'
            c = self.c

        if self.model is None:
            self.build_gurobi_model(c=c)
        self.set_objective(c)
        self.set_method(method)
        self.model.setParam(gp.GRB.Param.Presolve, 0)
        self.model.setParam('OutputFlag', 1 if verbose else 0)
        self.model.update()

        if pstart is not None:
            self.model.setParam('LPWarmStart', 1)
            self.set_solution(pstart, pstart=True)

        t0 = time.time()

        if self.model.SolCount > 0:
            obj_values = [c.dot(np.array(self.model.getAttr('x', self.x)))]
        else:
            obj_values = []
        iter_times = [0.0]
        iter_counts = [0]

        # count actual edge steps
        self.model._trueiterations = 0
        self.model._prevobj = None

        def obj_callback(model, where):
            if where == gp.GRB.Callback.SIMPLEX:
                obj = model.cbGet(gp.GRB.Callback.SPX_OBJVAL)
                obj_values.append(obj)

                if self.model._prevobj is not None and self.model._prevobj != obj:
                    self.model._trueiterations += 1
                    if verbose:
                        print("Improving pivot {}: {} -> {}".format(
                            self.model._trueiterations, self.model._prevobj, obj))

                self.model._prevobj = obj

                iter_times.append(time.time() - t0)
                iter_counts.append(model.cbGet(gp.GRB.Callback.SPX_ITRCNT))

        if record_objs:
            self.model.optimize(obj_callback)
        else:
            self.model.optimize()

        print(f'Model Status: {self.model.status}')
        if self.model.status != gp.GRB.Status.OPTIMAL:
            raise RuntimeError('Model failed to solve')

        x_optimal = self.model.getAttr('x', self.x)
        obj_optimal = self.model.objVal
        num_steps = self.model.getAttr('IterCount')

        if self.model._prevobj != obj_optimal:
            obj_values.append(obj_optimal)
            num_steps = num_steps + 1
            iter_times.append(iter_times[-1])
            iter_counts.append(iter_counts[-1])

        solve_time = time.time() - t0
        output = result(0, x=x_optimal, obj=obj_optimal, n_iters=num_steps, solve_time=solve_time,
                         iter_times=iter_times, obj_values=obj_values, iter_counts=iter_counts)
        print('\nFinal objective value for simplex:', obj_optimal)
        return output

    # warm start the model with the provided solution
    def set_solution(self, x, pstart=False, verbose=False):

        if pstart:
            for i in range(self.n):
                self.x[i].PStart = x[i]
            self.model.update()

        else:
            for i in range(self.n):
                self.x[i].lb = x[i]
                self.x[i].ub = x[i]

            self.model.setParam(gp.GRB.Param.Presolve, 0)
            self.model.setParam(gp.GRB.Param.FeasibilityTol, 1e-4)
            self.model.optimize()
            if self.model.status != gp.GRB.Status.OPTIMAL:
                raise RuntimeError('Failed to find feasible solution.')
            if verbose:
                print('Feasible solution found with objective {}'.format(self.model.objVal))

            # reset the model to its original state
            for i in range(self.n):
                self.x[i].lb = -INF
                self.x[i].ub = INF
            c_orig = np.copy(self.c)
            self.set_objective(np.zeros(self.n))
            self.model.optimize()
            if self.model.status != gp.GRB.Status.OPTIMAL:
                raise RuntimeError('Failed to set feasible solution.')
            self.set_objective(c_orig)
            self.model.setParam(gp.GRB.Param.FeasibilityTol, 1e-6)
            vbasis = self.model.getAttr(gp.GRB.Attr.VBasis)
            cbasis = self.model.getAttr(gp.GRB.Attr.CBasis)

            return vbasis, cbasis

    def is_vertex(self):
        assert hasattr(self, 'x_current') and hasattr(self, 'active_inds'), \
            'Call get_active_constraints or take_maximal_step first'

        # active_inds tracks which x_i = 0 (nonnegativity constraints that are tight)
        # build the corresponding identity rows for those indices
        nonnegativity_active = np.eye(self.n)[self.active_inds]

        if self.A is not None:
            active = np.vstack([self.A, nonnegativity_active])
        else:
            active = nonnegativity_active

        if active.shape[0] < self.n:
            return False

        sv = np.linalg.svd(active, compute_uv=False)
        is_vert = sv[self.n - 1] > 1e-4
        if not is_vert:
            rank = np.sum(sv > 1e-4)
            print(f'Not a vertex. Rank of active matrix: {rank}, n: {self.n}')
        return is_vert

    def copy(self):
        b_copy = self.b.copy() if self.b is not None else None
        P_copy = PolyhedronStandard(self.A.copy(), b_copy, self.c.copy())
        if self.model is not None:
            P_copy.build_gurobi_model(c=P_copy.c)
        return P_copy