import numpy as np
import numpy.polynomial.polynomial as nppoly
import scipy.special
import torch as tr
import sys
import math
import warnings
import re as restring
from math import log2, floor, ceil, sqrt, factorial
import matplotlib.pyplot as plt

tr.set_printoptions(precision=8)

################################### aux functions ##############################

#### interval intersection
def interval_intersection(set1, set2):
    # realizes cap
    a = set1[0]
    b = set1[1]

    c = set2[0]
    d = set2[1]

    assert a < b and c < d

    # realize cap 
    lo = max(a,c)
    hi = min(b,d)
    
    intersection = []
    if lo < hi:
        intersection = [lo, hi]
    if lo == hi:
        intersection = [lo]

    return intersection
   

#### interval union
def interval_union(set1, set2):
    # merge intervals [a,b], [c,d]
    a = set1[0]
    b = set1[1]

    c = set2[0]
    d = set2[1]

    assert a < b and c < d

    union_set = []
    # if the intersection is empty set1 and set2 are distinct
    if len(interval_intersection(set1, set2)) == 0:
        union_set.append([a,b])
        union_set.append([c,d])
    # if the intersection is not empty
    else:
        union_set = [min(a,c), max(b,d)]

    return union_set


#### union of a set of intervals
def interval_big_union(list_intervals):

    # nothing to do
    if len(list_intervals) == 1 or len(list_intervals) == 0:
        return list_intervals

    final_list = [[None]]
    # search for nonempty intersections of the first interval with the others
    A = list_intervals[0]
    union_flag = 0
    for j in range(1, len(list_intervals)):
        intersec_temp = interval_intersection(A, list_intervals[j])
        # if it is not empty build the union
        if len(intersec_temp) > 0:
            union_flag = 1
            A = interval_union(A, list_intervals[j])
        # if it is empty add interval to the final list
        else:
            final_list.append(list_intervals[j])

    # add the union to the final list
    final_list[0] = A

    # if there was at least one union, we have to check whether there are others
    # with this unioned set
    if union_flag:
        final_list = interval_big_union(final_list)
    # if not then A is distinct from other intervals and it is in the final list
    # move on with the rest of the interval list
    else:
        rest = interval_big_union(list_intervals[1:])
        final_list = [A, *rest]


    return final_list


#### union of a ordered set of intervals in pytorch 
def ordered_big_union(intervals):
    # Input: intervals: ordered intervals as Tensor of type int, shape=(N,2)
    #        ordered means, intervals[0,0] <= intervals[1,0] <= ...
    # Output: cup_intervals: the union of the intervals
    
    N = intervals.shape[0]
    # more than one interval
    if N > 1:
        # intervals intersect
        if intervals[0,1] >= intervals[1,0]-1:
            intervals[1,0] = intervals[0,0]
            intervals      = intervals[1:,:]

            # call the function with the remainder
            intervals = ordered_big_union(intervals)

        # intervals do not intersect
        else:
            # call the function with the remainder
            intervals = tr.cat((intervals[0:1,:], \
                                ordered_big_union(intervals[1:,:])), dim=0)
            

    return intervals


#### union for periodiced ordered intervals
def per_ordered_big_union(intervals, level):
    # Input: intervals: ordered intervals as Tensor of type int, shape=(N,2)
    #        ordered means, intervals[0,0] <= intervals[1,0] <= ...
    # Output: cup_intervals: the union of the intervals

    # get data type and device
    dtype = intervals.dtype
    device = intervals.device
 
    # build the union
    intervals = ordered_big_union(intervals)

    # flags for first and last interval
    first = False
    last  = False

    # divide the intervals if needed
    if intervals[0,0] < 0: 
        first     = True
        # redefine first interval
        temp      = intervals[0,0]
        intervals = tr.cat((tr.tensor([[temp, -1]], dtype=dtype, device=device),\
                            intervals), dim=0)
        intervals[1,0] = 0

    if intervals[-1,-1] > 2**(level)-1:
        last      = True
        # redefine last interval
        temp      = intervals[-1,-1]
        intervals = tr.cat((intervals,\
                            tr.tensor([[2**(level),temp]], dtype=dtype, \
                                                           device=device)), \
                           dim=0)
        intervals[-2,-1] = 2**(level)-1

    # order the interval again
    if first and last:
        intervals = tr.cat((intervals[-1:,:], \
                            intervals[1:-1,:], \
                            intervals[0:1,:]), dim=0)

    elif first and not last:        
        intervals = tr.cat((intervals[1:,:], \
                            intervals[0:1,:]), dim=0)
    elif last and not first:        
        intervals = tr.cat((intervals[-1:,:], \
                            intervals[:-1,:]), dim=0)

    # build the remainder of all intervals to project into 0,...,2**(j+1)-1
    intervals = tr.remainder(intervals, int(2**(level)))
    # call big_cup again
    intervals = ordered_big_union(intervals)

    return intervals


# lexicographically sort a 2D-tensor
def torch_lexsort(A, \
                  dim=0, \
                  return_indices=False, \
                  return_unique =False, \
                  return_counts =False, \
                  return_inverse=False):

    # must be at most a 2D tensor
    assert A.ndim <= 2

    # make unique and sort lexicographically
    A_uniq, p_inv, counts = tr.unique(A, \
                                      dim=dim, \
                                      sorted=True, \
                                      return_inverse=True, \
                                      return_counts=return_counts)

    arg_sort_p_inv = tr.argsort(p_inv)

    # sort tensor A
    return_tupel = (A[arg_sort_p_inv],)

    if return_indices is True:
        # return indices and lexicographically sorted tensor 
        return_tupel = return_tupel + (arg_sort_p_inv,)
    if return_unique is True:
        # return unique lexicographically sorted tensor 
        return_tupel = return_tupel + (A_uniq,)
    if return_counts is True:
        # make counts int32
        counts = counts.int()
        # return counts for each element in A_uniq
        return_tupel = return_tupel + (counts,)
    if return_inverse is True:
        # return inverse indices for each element in A_uniq
        return_tupel = return_tupel + (p_inv,)

    return return_tupel


#### transform a tupel of numbers interpreted in basis b into number in basis 10
def basis_10_transform(numbs_b, \
                       b, \
                       dtype=tr.int32, \
                       device=tr.device("cpu")):
    # must be 2d tensor
    assert numbs_b.ndim == 2

    numbs_10 = tr.zeros(numbs_b.shape[0], dtype=dtype, device=device)
    for i in range(numbs_b.shape[1]):
        numbs_10 = numbs_10 + numbs_b[:,-1-i]*b**i

    return numbs_10


def inv_basis_10_transform(numbs_10, \
                           b, \
                           dim, \
                           dtype=tr.int32, \
                           device=tr.device("cpu")):
    # must be 1d tensor
    assert numbs_10.ndim == 1

    if dim == 1:
        return numbs_10.reshape(-1,1)

    numbs_b = tr.zeros(numbs_10.shape[0], dim, dtype=dtype, device=device)
    for i in range(dim):
        numbs_b[:,-1-i] = tr.floor(numbs_10/b**i)-b*tr.floor(numbs_10/b**(i+1))

    return numbs_b


def inv_basis_10_transform_i(numbs_10, \
                             b, \
                             ith, \
                             dtype=tr.int32, \
                             device=tr.device("cpu")):
    # must be 1d tensor
    assert numbs_10.ndim == 1

    numbs_b_i = tr.floor(numbs_10/b**ith)-b*tr.floor(numbs_10/b**(ith+1))

    return numbs_b_i



#### function for computing binoms within mask calculation
def binomd(n, m):

    n = int(n)
    m = int(m)

    if m>n:
        return 0.

    if m==0:
        return 1.

    prod = 1.

    for j in range(m+1, n+1):
        prod *= j

    for j in range(1, n-m+1):
        prod /= j

    return prod

#### some aux functions for fwt which gives the order of wavelet coeffs in nD
def make_zeros(x):
    n = x.shape[0]
    x[:n//2,...] = 0
    return x

def make_ones(x):
    n = x.shape[0]
    x[n//2:,...] = 1
    return x

def make_zero_ones(x):
    x = make_ones(x)
    x = make_zeros(x) 

    n = x.shape[0]
    if n >= 2:
        x[:n//2,:-1] = make_zero_ones(x[:n//2,:-1])
        x[n//2:,:-1] = make_zero_ones(x[n//2:,:-1])

    return x



############################### end aux functions ##############################


