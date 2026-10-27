import numpy as np
import gurobipy as gp
import time

from utils import METHODS, INF, EPS


class PolyhedralModelStandard():

    # Given matrix A (and optional inds argument / objective function),
    # builds a polyhedral model for computing steepest-descent circuits
    # as a gurobi linear program model

    def __init__(self, A=None, active_inds=[], c=None, primal=True, method='dual_simplex', alt=False, verbose=False):

        print('Building polyhedral model for standard form in standard form. Solve method: {}'.format(method))

        self.model = gp.Model()
        self.model.setParam('OutputFlag', 0 if not verbose else 1)
        self.model.setParam(gp.GRB.Param.Presolve, 0)
        self.primal = primal
        self.verbose = verbose

        # add variables and constraints to the model
        self.m_A, self.n = A.shape
        self.y_pos = []
        self.y_neg = []
        self.s = []
        self.t = []

        for i in range(self.n):
            self.y_pos.append(self.model.addVar(lb=0.0, name='y_pos_{}'.format(i)))

        for i in range(self.n):
            self.y_neg.append(self.model.addVar(lb=0.0, name='y_neg_{}'.format(i)))

        for i in range(self.n):
            self.s.append(self.model.addVar(lb=0.0, name='s_{}'.format(i)))

        for i in range(self.n):
            self.t.append(self.model.addVar(lb=0.0, name='t_{}'.format(i)))

        for i in range(self.m_A):
            self.model.addConstr(gp.LinExpr(list(A[i]) + list(-A[i]), self.y_pos + self.y_neg) == 0, name='A_{}'.format(i))

        self.model.addConstr(gp.LinExpr([1] * (2 * self.n), self.y_pos + self.y_neg) == 1, name='1_norm')

        if c is not None:
            self.set_objective(c)

        # a common form of y_i^- + s_i = 1, dynamically updated as needed
        for i in range(self.n):
            self.model.addConstr(self.y_neg[i] + self.s[i] == 1, name='slack_{}'.format(i))

        for i in range(self.n):
            self.model.addConstr(self.y_pos[i] + self.t[i] == 1, name='restrict_{}'.format(i))

        self.model.update()

        self.set_active_inds(active_inds)

        self.set_method(method)

        self.model.update()

        print('Polyhedral model built!')

    def set_objective(self, c):
        self.c_new = np.concatenate((c, -c))
        self.model.setObjective(gp.LinExpr(c, self.y_pos) - gp.LinExpr(c, self.y_neg))

    def set_active_inds(self, active_inds, **kwargs):
        rest_inds = kwargs.get('rest_inds', None)
        self.active_inds = active_inds

        # reset everything to slack 1
        for i in range(self.n):
            constr = self.model.getConstrByName('slack_{}'.format(i))
            constr.RHS = 1
            constr_rest = self.model.getConstrByName('restrict_{}'.format(i))
            constr_rest.RHS = 1

        for i in self.active_inds:
            constr = self.model.getConstrByName('slack_{}'.format(i))
            constr.RHS = 0

        ###facet restriction
        if rest_inds is not None:
            for i in rest_inds:
                constr = self.model.getConstrByName('slack_{}'.format(i))
                constr.RHS = 0
                constr_rest = self.model.getConstrByName('restrict_{}'.format(i))
                constr_rest.RHS = 0

        self.model.update()

    def set_method(self, method):
        self.method = method
        self.model.Params.method = METHODS[method]

    def compute_sd_direction(self, verbose=False):
        flag = 1 if verbose else 0
        self.model.setParam(gp.GRB.Param.OutputFlag, flag)

        t0 = time.time()
        self.model._phase1_time = None
        self.model._is_dualinf = True

        def dualinf_callback(model, where):
            if where == gp.GRB.Callback.SIMPLEX:
                if model._is_dualinf:
                    dualinf = model.cbGet(gp.GRB.Callback.SPX_DUALINF)
                    if dualinf < EPS:
                        model._phase1_time = time.time() - t0
                        model._is_dualinf = False

        self.model.optimize(dualinf_callback)

        if self.model.status != gp.GRB.Status.OPTIMAL:
            raise RuntimeError('Failed to find steepst-descent direction.')

        phase2_time = time.time() - self.model._phase1_time
        phase_times = (self.model._phase1_time, phase2_time)
        y_pos = self.model.getAttr('x', self.y_pos)
        y_neg = self.model.getAttr('x', self.y_neg)
        steepness = self.model.objVal
        num_steps = self.model.getAttr('IterCount')
        solve_time = self.model.getAttr('Runtime')

        return np.asarray(y_pos), np.asarray(y_neg), steepness, num_steps, solve_time, phase_times

    # find a feasible solution for the polyhedral model
    def find_feasible_solution(self):
        c_orig = np.copy(self.c)
        c = np.zeros(self.n)
        self.set_objective(c)

        self.model.optimize()
        if self.model.status != gp.GRB.Status.OPTIMAL:
            raise RuntimeError('Failed to find feasible solution.')

        print(f"Objective value: {self.model.ObjVal}")

        self.set_objective(c_orig)

        y_pos = np.asarray(self.model.getAttr('X', self.y_pos))
        y_neg = np.asarray(self.model.getAttr('X', self.y_neg))
        x_feasible = y_pos - y_neg
        return x_feasible

    # should still work
    def reset(self):
        self.model.reset()

    # routine to set trivial basis
    def set_trivial_basis(self, x, basis_array):
        t_start = time.time()

        if self.verbose:
            print('entering trivial basis setting routine')

        variables_warm = self.model.getVars()
        constraints_warm = self.model.getConstrs()

        n_poly = len(variables_warm)
        m_poly = len(constraints_warm)
        n_org = n_poly // 4  # this should be the original dimension

        vbasis = [-1] * n_poly      # nonbasic at lower bound
        cbasis = [-1] * m_poly      # slack nonbasic

        if self.verbose:
            print(f'{n_poly} variables and {m_poly} constraints being set now for the polyhedral model')
            print(f'original had {n_org} variables, confirming {len(x)} length of vector passed')

        # set all slack and restrict variables to be basic
        for col in range(0, 2 * n_org):
            vbasis[col + 2 * n_org] = 0

        # set the y_i^+ basic
        for col in basis_array:
            vbasis[col] = 0

        # set one y_i^- basic
        for col in basis_array:
            if x[col] > 0:
                vbasis[col + n_org] = 0
                break

        self.model.setAttr(gp.GRB.Attr.VBasis, variables_warm, vbasis)
        self.model.setAttr(gp.GRB.Attr.CBasis, constraints_warm, cbasis)
        self.model.update()

        if self.verbose:
            vb_check = self.model.getAttr(gp.GRB.Attr.VBasis, self.model.getVars())
            cb_check = self.model.getAttr(gp.GRB.Attr.CBasis, self.model.getConstrs())
            print(f'vbasis mismatches after set: {sum(a != b for a, b in zip(vb_check, vbasis))}')
            print(f'cbasis mismatches after set: {sum(a != b for a, b in zip(cb_check, cbasis))}')

        vb_before = self.model.getAttr(gp.GRB.Attr.VBasis, self.model.getVars())
        cb_before = self.model.getAttr(gp.GRB.Attr.CBasis, self.model.getConstrs())

        self.model.Params.IterationLimit = 0
        self.model.optimize()

        if self.verbose:
            vb_after = self.model.getAttr(gp.GRB.Attr.VBasis, self.model.getVars())
            added = [i for i, (b0, b1) in enumerate(zip(vb_before, vb_after)) if b0 != 0 and b1 == 0]
            dropped = [i for i, (b0, b1) in enumerate(zip(vb_before, vb_after)) if b0 == 0 and b1 != 0]
            print(f'Added: {len(added)}')
            print(f'Dropped: {len(dropped)}')

        self.model.setParam('IterationLimit', float('inf'))
        self.model.update()

        return time.time() - t_start

    # routine to set edge basis
    def set_edge_basis(self, x, x2, basis_array1, basis_array2):
        t_start = time.time()

        if self.verbose:
            print('entering edge basis setting routine')

        variables_warm = self.model.getVars()
        constraints_warm = self.model.getConstrs()

        n_poly = len(variables_warm)
        m_poly = len(constraints_warm)
        n_org = n_poly // 4  # this should be the original dimension

        vbasis = [-1] * n_poly      # nonbasic at lower bound
        cbasis = [-1] * m_poly      # slack nonbasic

        if self.verbose:
            print(f'{n_poly} variables and {m_poly} constraints being set now for the polyhedral model')
            print(f'original had {n_org} variables, confirming {len(x)} length of vector passed')

        # set all slack and restrict variables to be basic
        for col in range(0, 2 * n_org):
            vbasis[col + 2 * n_org] = 0

        # set the y_i^+ and y_i^- basic
        for col in basis_array2:
            if x2[col] not in x:
                vbasis[col] = 0
            elif x2[col] >= x[col]:
                # check if x does not contain it
                vbasis[col] = 0
            else:
                vbasis[col + n_org] = 0

        # # set one additional y_i^- basic
        # for col in basis_array1:
        #     if x[col] not in x2:
        #         vbasis[col + n_org] = 0
        #         break
        
        # set one additional y_i^- basic
        for col in basis_array1:
            if col not in basis_array2:
                vbasis[col + n_org] = 0
                break

        self.model.setAttr(gp.GRB.Attr.VBasis, variables_warm, vbasis)
        self.model.setAttr(gp.GRB.Attr.CBasis, constraints_warm, cbasis)
        self.model.update()

        if self.verbose:
            vb_check = self.model.getAttr(gp.GRB.Attr.VBasis, self.model.getVars())
            cb_check = self.model.getAttr(gp.GRB.Attr.CBasis, self.model.getConstrs())
            print(f'vbasis mismatches after set: {sum(a != b for a, b in zip(vb_check, vbasis))}')
            print(f'cbasis mismatches after set: {sum(a != b for a, b in zip(cb_check, cbasis))}')

        vb_before = self.model.getAttr(gp.GRB.Attr.VBasis, self.model.getVars())
        cb_before = self.model.getAttr(gp.GRB.Attr.CBasis, self.model.getConstrs())

        self.model.Params.IterationLimit = 0
        self.model.optimize()

        if self.verbose:
            vb_after = self.model.getAttr(gp.GRB.Attr.VBasis, self.model.getVars())
            added = [i for i, (b0, b1) in enumerate(zip(vb_before, vb_after)) if b0 != 0 and b1 == 0]
            dropped = [i for i, (b0, b1) in enumerate(zip(vb_before, vb_after)) if b0 == 0 and b1 != 0]
            print(f'Added: {len(added)}')
            print(f'Dropped: {len(dropped)}')

        self.model.setParam('IterationLimit', float('inf'))
        self.model.update()

        return time.time() - t_start