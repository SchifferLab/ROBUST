"""
Give any 1d timeseries this class calculates the block averaged standard error
See:
    Flyvbjerg, Henrik, and Henrik Gordon Petersen.
    "Error estimates on averages of correlated data."
    The Journal of Chemical Physics 91.1 (1989): 461-466.

    Grossfield, Alan, and Daniel M. Zuckerman.
    "Quantifying uncertainty and sampling quality in biomolecular simulations."
     Annual reports in computational chemistry 5 (2009): 23-48.
"""

from __future__ import print_function, division

import numpy as np
import scipy as sp

from multiprocessing import Pool


def block_averages(x, l):
    """
    Given a vector x return a vector x' of the block averages .
    """

    if l == 1:
        return x

    # If the array x is not a multiple of l drop the first x values so that it becomes one
    if len(x) % l != 0:
        x = x[int(len(x) % l):]

    xp = []
    for i in range(len(x) // int(l)):
        xp.append(np.mean(x[l * i:l + l * i]))

    return np.array(xp)


def ste(x):
    return np.std(x)/np.sqrt(len(x))

def get_bse(x, min_blocks=3):

    steps = np.max((1,len(x)//100))
    stop = len(x)//min_blocks+steps

    bse = []

    for l in range(1, stop, steps):
        xp = block_averages(x, l)
        bse.append(ste(xp))

    # Fit simple exponential to determine plateau
    def model_func(x, p0, p1):
        return p0 * (1 - np.exp(-p1 * x))

    opt_parms, parm_cov = sp.optimize.curve_fit(model_func, np.arange(len(bse)), bse, (np.mean(bse), 0.1))
    return opt_parms[0]


def run(self):

    while True:
        data = self.in_queue.get()
        # poison pill
        if data == 'STOP':
            break
        i, data_array = data
        # blocksize 1 => full run
        block_length = 2
        blocks = np.array([1, ])
        block_stderr = [np.std(data_array) / np.sqrt(len(data_array))]
        # Transform the series until nblocks = 2
        while len(data_array) // block_length >= self.min_m:
            b = self.transform(data_array, block_length)
            block_stderr.append(np.std(b) / np.sqrt(len(b)))
            block_length += 1
            if len(data_array) // block_length < self.min_m:
                blocks = np.arange(block_length - 1) + 1



        # Fit curve
        opt_parms, parm_cov = sp.optimize.curve_fit(model_func, blocks, block_stderr, maxfev=2000)
        error_estimate = opt_parms[0]
        while True:
            if self.out_queue.full():
                time.sleep(5)
            else:
                self.out_queue.put([i, error_estimate])
                break
    return