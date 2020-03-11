import os
import numpy as np
import random
import math
from scipy.spatial.distance import euclidean, pdist, squareform
from scipy.spatial import KDTree
import scipy.stats as sts
import multiprocessing as mp
import time

ATOMIC_WEIGHTS = {'H':1.008, 'HE':4.002602, 'LI':6.94, 'BE':9.012182,
       'B':10.81, 'C':12.011, 'N':14.007, 'O':15.999, 'F':18.9984032,
       'NE':20.1797, 'NA':22.98976928, 'MG':24.305, 'AL':26.9815386,
       'SI':28.085, 'P':30.973762, 'S':32.06, 'CL':35.45, 'AR':39.948,
       'K':39.0983, 'CA':40.078, 'SC':44.955912, 'TI':47.867, 'V':50.9415,
       'CR':51.9961, 'MN':54.938045, 'FE':55.845, 'CO':58.933195,
       'NI':58.6934, 'CU':63.546, 'ZN':65.38, 'GA':69.723, 'GE':72.630,
       'AS':74.92160, 'SE':78.96, 'BR':79.904, 'RB':85.4678, 'SR':87.62,
       'Y':88.90585, 'ZR':91.224, 'NB':92.90638, 'MO':95.96, 'TC':98,
       'RU':101.07, 'RH':102.90550, 'PD':106.42, 'AG':107.8682, 'CD':112.411,
       'IN':114.818, 'SN':118.710, 'SB':121.760, 'TE':127.60, 'I':126.90447,
       'XE':131.293, 'CS':132.9054519, 'BA':137.327, 'LA':138.90547,
       'CE':140.116, 'PR':140.90765, 'ND':144.242, 'PM':145, 'SM':150.36,
       'EU':151.964, 'GD':157.25, 'TB':158.92535, 'DY':162.500, 'HO':164.93032,
       'ER':167.259, 'TM':168.93421, 'YB':173.054, 'LU':174.9668, 'HF':178.49,
       'TA':180.94788, 'W':183.84, 'RE':186.207, 'OS':190.23, 'IR':192.217,
       'PT':195.084, 'AU':196.966569, 'HG':200.592, 'TL':204.38, 'PB':207.2,
       'BI':208.98040, 'PO':209, 'AT':210, 'RN':222, 'FR':223, 'RA':226,
       'AC':227, 'TH':232.03806, 'PA':231.03588, 'U':238.02891, 'NP':237,
       'PU':244, 'AM':243, 'CM':247, 'BK':247, 'CF':251, 'ES':252, 'FM':257,
       'MD':258, 'NO':259, 'LR':262, 'RF':267, 'DB':268, 'SG':269, 'BH':270,
       'HS':269, 'MT':278, 'DS':281, 'RG':281, 'CN':285, 'UUT':286, 'FL':289,
       'UUP':288, 'LV':293, 'UUS':294}


class Solvate:
    def __init__(self,pdb_file, water_box="water.pdb", neighbours=2, ncpus = os.cpu_count(), clash_dist=2.6, hydration_shell=4.2):
        self.pdb_file = self.read_pdb(pdb_file,record_res=True)
        self.pdb_file = self.translate_coords(self.pdb_file[0],self.pdb_file[1])
        self.water_box = self.read_pdb(water_box)
        self.neighbours = neighbours
        self.ncpus = ncpus
        self.clash_dist = clash_dist
        self.hydration_shell = hydration_shell
        
    def read_pdb(self, pdb, include='ATOM,HETATM', record_res=False):
        """
        Reads in pdb file
        Returns
            coordinates (dictionary): dictionary containing coordinates for atoms in pdb
            Center of Mass (tuple): tuple containing coordinates of the center of mass
        """
        com = self.center_of_mass(pdb)
        include = tuple(include.split(','))
        coordinates = {}
        res_names = {}
        with open(pdb) as f:
            g = f.read().splitlines()
            for line in [i for i in g if i.startswith(include)]:
                resnum = int(line[22:27].strip())
                chain = line[21]
                atom_name = line[13:17].strip()
                res_name = line[17:21]
                res_names["%s:%s" % (chain,resnum)] = res_name

                if "%s:%s" % (chain,resnum) not in coordinates.keys():
                    coordinates["%s:%s" % (chain,resnum)] = {}
                
                coordinates["%s:%s" % (chain,resnum)][atom_name] = [float(line[30:38]),    # x_coord
                                    float(line[38:46]),    # y_coord
                                    float(line[46:54])     # z_coord
                                       ]
            f.close()
        
        if record_res:
            self.res_names = res_names
        #print(resnum)
        return coordinates, com
            
    def center_of_mass(self, pdb, include='ATOM,HETATM'):
        """
        Calculates center of mass of a protein and/or ligand structure.
        Returns:
            center (list): List of float coordinates [x,y,z] that represent the
            center of mass (precision 3).
        """

        center = [None, None, None]
        include = tuple(include.split(','))

        with open(pdb, 'r') as pdb:

            # extract coordinates [ [x1,y1,z1], [x2,y2,z2], ... ]
            coordinates = []
            masses = []    
            for line in pdb:
                if line.startswith(include):
                    coordinates.append([float(line[30:38]),    # x_coord
                                        float(line[38:46]),    # y_coord
                                        float(line[46:54])     # z_coord
                                       ])
                    element_name = line[76:].strip()
                    if element_name not in ATOMIC_WEIGHTS:
                        element_name = line.split()[2].strip()[0]
                    masses.append(ATOMIC_WEIGHTS[element_name])

            assert len(coordinates) == len(masses)

            # calculate relative weight of every atomic mass
            total_mass = sum(masses)
            weights = [float(atom_mass/total_mass) for atom_mass in masses]

            # calculate center of mass
            center = [sum([coordinates[i][j] * weights[i]
                  for i in range(len(weights))]) for j in range(3)]
            center_rounded = [round(center[i], 3) for i in range(3)]
            return center_rounded
        
    def translate_coords(self,coords, com, new_center=(0,0,0)):
        """
        Translates coordinates to a new center of mass (default: 0,0,0)
        Returns:
            coords (dictionary): dictionary containing translated coordinates
        """
        
        new_coords = {}
        shift = np.array(new_center) - np.array(com)
        #print(shift)
        
        for k,v in coords.items():
            new_coords[k] = {}
            for k1,v1 in v.items():
                new_coords[k][k1] = [round(i,3) for i in list(np.array(v1) + shift)]
        
        return (new_coords,new_center)
    
    def random_rotation(self):
        """
        Applies a random rotation to the protein coordinates to self.pdb_file
        Returns:
            rotated (dictionary): dictionary containing rotated coordinates
        """
        angle = math.radians(random.uniform(0,360))
        
        choices = ['x','y','z']
        choice = random.choice(choices)
        
        rotated = {}
        
        ox,oy,oz = self.pdb_file[1]
        
        for k,v in self.pdb_file[0].items():
            rotated[k] = {}
            for k1,v1 in v.items():
                px, py, pz = v1
                
                if choice == 'x':
                    p_rot_x = px-ox
                    p_rot_y = (py-oy)*math.cos(angle) - (pz-oz)*math.sin(angle)
                    p_rot_z = (pz-oz)*math.cos(angle) + (py-oy)*math.sin(angle)
                elif choice == 'y':
                    p_rot_y = py-oy
                    p_rot_x = (px-ox)*math.cos(angle) - (pz-oz)*math.sin(angle)
                    p_rot_z = (pz-oz)*math.cos(angle) + (px-ox)*math.sin(angle)
                else:
                    p_rot_z = pz-oz
                    p_rot_y = (py-oy)*math.cos(angle) - (px-ox)*math.sin(angle)
                    p_rot_x = (px-ox)*math.cos(angle) + (py-oy)*math.sin(angle)
                
                rotated[k][k1] = [p_rot_x,p_rot_y,p_rot_z]
        
#         rotated_centered = self.translate_coords()
                
        return rotated
    
    def remove_clash_waters_process(self,w):
#         #print(w)
        for a in self.atoms:
            w = np.array(w)
            a = np.array(a)
            dist = np.linalg.norm(w - a)
            x = False
            if dist < self.clash_dist:
                x = True
                break
        if not x:
#             #print(w)
#             self.water_no_clash.append(w.tolist())
            return w.tolist()

    def remove_clash_kdtree_process(self,w):
        dist, _ = self.atom_tree.query(w)
        if dist < self.clash_dist:
            return w
                
        
    def remove_clash_waters(self, coords):
        """
        Returns new water dictionary removing clashes with protein
        """
        water_clash = self.trim_box().tolist()
        self.atoms = []
#         self.water_no_clash = []
#         #print(type(water_no_clash))
        for kp,vp in coords.items():
#             #print(kp)
            for vp1 in vp.values():
                self.atoms.append(vp1)
            
        pool = mp.Pool(self.ncpus)
        result = pool.map(self.remove_clash_waters_process, water_clash)
        pool.close()
        pool.join()

    def remove_clash_waters_kdtree(self, coords):
        """
        Returns new water dictionary removing clashes with protein
        """
        water_clash = self.trim_box().tolist()
        atoms = []
#         self.water_no_clash = []
#         #print(type(water_no_clash))
        for kp,vp in coords.items():
#             #print(kp)
            for vp1 in vp.values():
                atoms.append(vp1)

        self.atom_tree = KDTree(atoms)
            
        pool = mp.Pool(self.ncpus)
        result = pool.map(self.remove_clash_kdtree_process, water_clash)
        pool.close()
        pool.join()
            
#             break
#         #print(result)
                            
        return [i for i in result if i]
    
    def water_box_to_array(self):
        """
        Transfers self.water_box to numpy array containing only oxygen coordinates
        """
        arr = []
        for v in self.water_box[0].values():
            arr.append(v['OW'])
        return np.array(arr)
    
    def furthest_point(self):
        """
        Calculates largest distance from protein center in a PDB structure.
        Returns:
            Largest distance (Float)
        """
        coords = self.pdb_file[0]
        com  = self.pdb_file[1]
        
        max_dist = 0
        
        for v in coords.values():
            for v1 in v.values():
                dist = euclidean(com,v1)
                if dist > max_dist:
                    max_dist = dist
                    
        return max_dist
    
    def trim_box(self, pad=1):
        """
        Returns reduced numpy array water box with a radius of max distance + pad
        """
        wat = self.water_box_to_array()
        wat_trim = []
        
        com = np.array([0,0,0])
        
        max_dist = self.furthest_point() + pad
        for w in [i for i in wat if np.linalg.norm(com-np.array(i)) < max_dist]:
            wat_trim.append(w)
        
        return np.array(wat_trim)
    
    def remove_bulk_water(self,water):
        """
        Returns list with non-bulk water removed
        """
        dists = [1 if len([j for j in i if j < self.hydration_shell]) > self.neighbours else 0 for i in squareform(pdist(water))]
        
        return [i for i,j in zip(water,dists) if j]
    
    def multi_run_wrapper(self,args):
        return self.get_distances_kdtree_process(*args)
    
    def get_distances_process(self,k,v,k1,v1,water):
        x = 0
        for w in water:
            x+=1
            w = np.array(w)
            v1 = np.array(v1)
            dist = np.linalg.norm(w - v1)
            if x==1:
                min_dist = dist
            else:
                if dist < min_dist:
                    min_dist = dist
        return (k,k1,min_dist)

    def get_distances_kdtree_process(self,k,v,k1,v1,water):
        dist, _ = water.query(v1)
        return (k,k1,dist)
    
    def get_distances(self, coords, water):
        """
        Returns a dictionary with atomic distances to nearest bulk water
        """
        d_dists = {}
        args = []
        for k,v in coords.items():
            d_dists[k] = {}
            for k1,v1 in v.items():
                args.append((k,v,k1,v1,water))

        pool = mp.Pool(self.ncpus)
        result = pool.map(self.multi_run_wrapper, args)
        pool.close()
        pool.join()
        
        for r in result:
            if r[0] not in d_dists:
                d_dists[r[0]] = {r[1]:r[2]}
            else:
                d_dists[r[0]][r[1]] = r[2]
                
        return d_dists

    def get_distances_kdtree(self, coords, water):
        """
        Returns a dictionary with atomic distances to nearest bulk water
        """
        water = KDTree(water)
        d_dists = {}
        args = []
        for k,v in coords.items():
            d_dists[k] = {}
            for k1,v1 in v.items():
                args.append((k,v,k1,v1,water))

        pool = mp.Pool(self.ncpus)
        result = pool.map(self.multi_run_wrapper, args)
        pool.close()
        pool.join()
        
        for r in result:
            if r[0] not in d_dists:
                d_dists[r[0]] = {r[1]:r[2]}
            else:
                d_dists[r[0]][r[1]] = r[2]
                
        return d_dists
    
    def get_res_distances(self, dists):
        """
        Returns a dictionary with average depth for residues (averaged over atoms for each residue)
        """

        mc = ['N','CA','C','O']
        pol = ('N','O','S','M')
        
        all_mean = {k:np.mean(list(v.values())) for k,v in dists.items()}
        mc_mean = {}
        sc_mean = {}
        sc_pol_mean = {}
        sc_nopol_mean = {}

        all_std = {k:np.std(list(v.values())) for k,v in dists.items()}
        mc_std = {}
        sc_std = {}
        sc_pol_std = {}
        sc_nopol_std = {}

        for k,v in dists.items():
            mc_tmp = []
            sc_tmp = []
            sc_pol_tmp = []
            sc_nopol_tmp = []

            for k1,v1 in v.items():
                if k1 in mc:
                    mc_tmp.append(v1)
                else:
                    sc_tmp.append(v1)
                    if k1.startswith(pol):
                        sc_pol_tmp.append(v1)
                    else:
                        sc_nopol_tmp.append(v1)

            mean_d = [mc_mean,sc_mean,sc_pol_mean,sc_nopol_mean]
            std_d = [mc_std,sc_std,sc_pol_std,sc_nopol_std]
            tmp_d = [mc_tmp,sc_tmp,sc_pol_tmp,sc_nopol_tmp]
            for mean,std,tmp in zip(mean_d, std_d, tmp_d):
                mean[k] = np.mean(tmp)
                std[k] = np.std(tmp)

        return all_mean, all_std, mc_mean, mc_std, sc_mean, sc_std, sc_pol_mean, sc_pol_std, sc_nopol_mean, sc_nopol_std

def run_iteration(solv):
    start_tot = time.time()
    d_rot = solv.random_rotation()
    start = time.time()
    wat = solv.remove_clash_waters_kdtree(d_rot)
    bulk_wat = solv.remove_bulk_water(wat)
    end = time.time()
    print("Water removed in: %s s" % str(end-start))
    start = time.time()
    dists = solv.get_distances_kdtree(d_rot,bulk_wat)
    end = time.time()
    print("Distances calculated in: %s s" % str(end - start))
    end_tot = time.time()

    print("Total time: %s s\n" % str(end_tot-start_tot))
    
    return dists

def output_atom_depth(prefix, runs, res_names):
    d_mean = {}
    d_std = {}

    for k in runs[0].keys():
        for k1 in runs[0][k]:
            atom = []
            for r in runs:
                atom.append(r[k][k1])
            mean_atom = np.mean(atom)
            std_atom = np.std(atom)
            
            if k not in d_mean.keys():
                d_mean[k] = {k1:mean_atom}
                d_std[k] = {k1:std_atom}
            else:
                d_mean[k][k1] = mean_atom
                d_std[k][k1] = std_atom

    with open("%s-atom.depth" % prefix,"w+") as f:
        f.write("Residue\tResType\tAtom\tMeanDepth\tSTDDepth\n")
        for k in d_mean:
            for k1 in d_mean[k]:
                f.write("%s\t%s\t%s\t%s\t%s\n" % (k,res_names[k],k1,d_mean[k][k1],d_std[k][k1]))
        f.close()

    return d_mean, d_std

def output_res_depth(prefix, res_depths, res_names):
    all_mean, all_std, mc_mean, mc_std, sc_mean, sc_std, sc_pol_mean, sc_pol_std, sc_nopol_mean,sc_nopol_std = res_depths

    with open("%s-residue.depth" % prefix,"w+") as f:
        f.write("Residue\tResType\tAllAtomMean\tAllAtomSTD\tMainChainMean\tMainChainSTD\tSideChainMean\tSideChainSTD\tSideChainPolarMean\tSideChainPolarMean\tSideChainPolarSTD\tSideChainNonPolarMean\tSideChainNonPolarSTD\n")
        for k in all_mean:
            write_string = "\t".join([k,str(res_names[k]),str(all_mean[k]),str(all_std[k]),str(mc_mean[k]),str(mc_std[k]),str(sc_mean[k]),str(sc_std[k]),str(sc_pol_mean[k]),str(sc_pol_std[k]),str(sc_nopol_mean[k]),str(sc_nopol_std[k])])
            f.write("%s\n" % write_string)
        f.close()