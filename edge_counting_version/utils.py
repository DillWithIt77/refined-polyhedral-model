import pickle
import random
import numpy as np

METHODS = {'auto': -1,
           'primal_simplex': 0,
           'dual_simplex': 1,
           'barrier': 2,
           'concurrent': 3,
           'deterministic_concurrent': 4,
           'deterministic_concurrent_simplex': 5,}

INF = 10e100
EPS = 10e-8

def avg(x):
    return float(sum(x)) / float(len(x))


class result:  
    def __init__(self, status, x=None, obj=None, n_iters=None, solve_time=None, iter_times=[], alg_type='simplex',
                 circuits=[], steps=[], simplex_iters=[], solve_times=[], sub_times=None,
                 obj_values=[], iter_counts=[], step_type = [], active_inds = [], steepness = [], rankA = None,
                 fr_entry_step=None, fr_exit_step=None):
        self.status = status
        self.x = x
        self.obj = obj
        self.n_iters = n_iters
        self.iter_times = iter_times
        self.solve_time = solve_time
        self.iter_counts = iter_counts
        self.alg_type = alg_type
        
        self.circuits = circuits
        self.steps = steps
        self.simplex_iters = simplex_iters
        self.solve_times = solve_times
        
        self.sub_times = sub_times
        self.obj_values = obj_values

        self.step_type = step_type
        self.active_inds = active_inds
        self.steepness = steepness
        self.rankA = rankA

        # facet-restriction switch points (step indices aligned with obj_values; None = never fired)
        self.fr_entry_step = fr_entry_step
        self.fr_exit_step = fr_exit_step
          
    def __str__(self):
        if self.status == 1:
            return ('Problem is unbounded.'
                        + '\nSteepest descent unbounded circuit: ' + str(self.circuits[-1].T)
                    )
        elif self.status == 0:
            output = ('\nOptimal objective: {}'.format(self.obj)
                   + '\nTotal solve time: {}'.format(self.solve_time)
                   + '\nNumber of iterations: {}'.format(self.n_iters)
            )
            if self.alg_type == 'steepest-descent':
                output += ('\nFirst simplex iterations {}'.format(self.simplex_iters[0])
                       + '\nAverage num simplex iterations {}'.format(sum(self.simplex_iters)/len(self.simplex_iters))
                       + '\nTotal simplex iterations: {}'.format(sum(self.simplex_iters))
                       + '\nFirst solve time {}'.format(self.solve_times[0])
                       + '\nAverage solve time {}'.format(sum(self.solve_times)/len(self.solve_times))
                       + '\nTotal solve time: {}'.format(sum(self.solve_times))
                       )
            return output
        else:
            return 'Problem unsolved'
        
    def save(self, fn):
        results = {'obj': self.obj, 
                   'x': self.x,
                   'obj_values': self.obj_values,
                   'n_iters': self.n_iters, 
                   'solve_time_total': self.solve_time,
                   'iter_times': self.iter_times,
                   'iter_counts': self.iter_counts,
                   'alg_type': self.alg_type,
                   'rankA': self.rankA}
        if self.alg_type == 'steepest-descent':
            results['simplex_iters'] = self.simplex_iters
            results['solve_times'] = self.solve_times
            results['sub_times'] = self.sub_times

            results['step_type'] = self.step_type
            results['active_inds'] = self.active_inds
            results['steepness'] = self.steepness
            results['steps'] = self.steps
            results['fr_entry_step'] = self.fr_entry_step
            results['fr_exit_step'] = self.fr_exit_step
            
        with open(fn, 'wb') as f:
            pickle.dump(results, f)