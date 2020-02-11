import os
import numpy as np
import random
import math
from scipy.spatial.distance import euclidean, pdist, squareform
import time
import scipy.stats as sts
import multiprocessing as mp
import argparse
import sys
from ResDepth.Solvate import Solvate, run_iteration, output_atom_depth, output_res_depth

parser = argparse.ArgumentParser(prog="ResidueDepth", description="A python implementation of the Residue Depth calculation by Tan et al.")
parser.add_argument('-pdb_file',required=True, nargs=1,
                    help="pdb to run residue depth calculation on")
parser.add_argument('-ncpus', nargs=1,
                    help="Number of cpus to use for calculation (default: number of available cpus will be used)")
parser.add_argument('-iterations', nargs=1,
                    help="number of iterations to run (default 25)")
parser.add_argument('-neighbours', nargs=1,
                    help="number of waters in hydration shell to be considered bulk water (default 2)")
parser.add_argument('-hydration_shell', nargs=1,
                    help="radius of hydration shell (default 4.2)")
parser.add_argument('-water_box', nargs=1,
                    help="specify different water model (default water.pdb)")
parser.add_argument('-clash_dist', nargs=1,
                    help="distance specifying where water clashes with protein atoms (default 2.6)")
parser.add_argument('-prefix', nargs=1,
                    help="prefix for output files (default pdb file name without .pdb)")

args = parser.parse_args()

if args.pdb_file:
    pdb_file = args.pdb_file[0]
else:
    print("Please specify -pdb_file")

if args.ncpus:
    ncpus = int(args.ncpus[0])
else:
    ncpus = os.cpu_count()

if args.iterations:
    iterations = int(args.iterations[0])
else:
    iterations = 25

if args.neighbours:
    neighbours = int(args.neighbours[0])
else:
    neighbours = 2

if args.hydration_shell:
    hydration_shell = float(args.hydration_shell[0])
else:
    hydration_shell = float(4.2)

if args.water_box:
    water_box = args.water_box[0]
else:
    water_box = "water.pdb"

if args.clash_dist:
    clash_dist = float(args.clash_dist[0])
else:
    clash_dist = 2.6

if args.prefix:
    prefix = args.prefix
else:
    prefix = pdb_file.split('.pdb')[0]


#####################################################################

if __name__ == "__main__":
    start_tot = time.time()
    solv = Solvate(pdb_file ,neighbours=neighbours, ncpus=ncpus, hydration_shell=hydration_shell, water_box=water_box, clash_dist=clash_dist)

    print("Running depth calculation on %s\n" % pdb_file)

    runs = []

    for run in range(0,iterations):
        print("Running iteration %s" % str(run+1))
        runs.append(run_iteration(solv))

    d_mean, d_std = output_atom_depth(prefix, runs, solv.res_names)

    res_depths = solv.get_res_distances(d_mean)

    output_res_depth(prefix, res_depths, solv.res_names)
    end_tot = time.time()
    print("Whole run completed in: %s s" % str(end_tot-start_tot))