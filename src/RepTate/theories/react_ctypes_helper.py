# RepTate: Rheology of Entangled Polymers: Toolkit for the Analysis of Theory and Experiments
# --------------------------------------------------------------------------------------------------------
#
# Authors:
#     Jorge Ramirez, jorge.ramirez@upm.es
#     Victor Boudara, victor.boudara@gmail.com
#
# Useful links:
#     http://blogs.upm.es/compsoftmatter/software/reptate/
#     https://github.com/jorge-ramirez-upm/RepTate
#     http://reptate.readthedocs.io
#
# --------------------------------------------------------------------------------------------------------
#
# Copyright (2017-2023): Jorge Ramirez, Victor Boudara, Universidad Politécnica de Madrid, University of Leeds
#
# This file is part of RepTate.
#
# RepTate is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# RepTate is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with RepTate.  If not, see <http://www.gnu.org/licenses/>.
#
# --------------------------------------------------------------------------------------------------------
"""
Define the C-variables and functions from the C-files that are needed in Python
"""
import numpy as np
import ctypes as ct
import sys
import os

from RepTate.core.ctypes_loader import load_ctypes_library

dir_path = os.path.dirname(
    os.path.realpath(__file__)
)  # get the directory path of current file
if sys.maxsize > 2 ** 32:
    # 64-bit system
    lib_path = os.path.join(dir_path, "react_lib_%s.so" % (sys.platform))
else:
    # 32-bit system
    lib_path = os.path.join(dir_path, "react_lib_%s_i686.so" % (sys.platform))
react_lib = load_ctypes_library(lib_path, "React CH library")

###############
# polybits.c
###############


# struct
class polybits_global_const(ct.Structure):
    """Global constants for polymer reaction simulations.

    This structure contains maximum limits and configuration constants used throughout
    the polymer reaction simulation framework. These constants define memory allocation
    limits for various data structures in the C library.

    Attributes:
        maxbobbins (int): Maximum number of BoB (Branch-on-Branch) bins for molecular weight distribution.
        maxmwdbins (int): Maximum number of molecular weight distribution bins.
        maxarm (int): Maximum number of polymer arms that can be stored in memory.
        maxpol (int): Maximum number of polymer molecules that can be stored.
        maxreact (int): Maximum number of reaction distributions that can be tracked.
        MAX_RLEVEL (int): Maximum recursion level for polymer architecture analysis.
        MAX_NBR (int): Maximum number of branch points per polymer molecule.
    """
    _fields_ = [
        ("maxbobbins", ct.c_int),
        ("maxmwdbins", ct.c_int),
        ("maxarm", ct.c_int),
        ("maxpol", ct.c_int),
        ("maxreact", ct.c_int),
        ("MAX_RLEVEL", ct.c_int),
        ("MAX_NBR", ct.c_int),
    ]


class polybits_global(ct.Structure):
    """Global state variables for polymer reaction memory pool management.

    This structure tracks the current state of the polymer reaction simulation memory pools,
    including availability of resources and pointers to the first available records in each pool.

    Attributes:
        first_in_pool (int): Index of the first available arm record in the pool.
        first_poly_in_pool (int): Index of the first available polymer record in the pool.
        first_dist_in_pool (int): Index of the first available distribution record in the pool.
        mmax (int): Current maximum number of monomers.
        num_react (int): Current number of active reaction distributions.
        arms_left (int): Number of arm records remaining in memory.
        react_pool_initialised (bool): Whether the reaction pool has been initialized.
        react_pool_declared (bool): Whether the reaction pool has been declared.
        arms_avail (bool): Whether arm records are available for allocation.
        polys_avail (bool): Whether polymer records are available for allocation.
        dists_avail (bool): Whether distribution records are available for allocation.
    """
    _fields_ = [
        ("first_in_pool", ct.c_int),
        ("first_poly_in_pool", ct.c_int),
        ("first_dist_in_pool", ct.c_int),
        ("mmax", ct.c_int),
        ("num_react", ct.c_int),
        ("arms_left", ct.c_int),
        ("react_pool_initialised", ct.c_bool),
        ("react_pool_declared", ct.c_bool),
        ("arms_avail", ct.c_bool),
        ("polys_avail", ct.c_bool),
        ("dists_avail", ct.c_bool),
    ]


class arm(ct.Structure):
    """Representation of a polymer chain arm.

    This structure contains all properties and connectivity information for a single
    polymer arm within a branched polymer molecule. Arms are connected through branching
    points and form the tree-like structure of the polymer.

    Attributes:
        arm_len (float): Length of the arm in monomer units.
        arm_conv (float): Conversion fraction at which this arm was created.
        arm_time (float): Time at which this arm was created during polymerization.
        arm_tm (float): Monomer addition time for this arm.
        arm_tddb (float): Time for double bond formation.
        L1 (int): Index of first left-connected arm (branch point).
        L2 (int): Index of second left-connected arm (branch point).
        R1 (int): Index of first right-connected arm (branch point).
        R2 (int): Index of second right-connected arm (branch point).
        up (int): Index of parent arm (toward polymer backbone).
        down (int): Index of child arm (away from backbone).
        armnum (int): Unique identifier for this arm.
        armcat (int): Catalyst type that created this arm.
        ended (int): Flag indicating if arm growth has terminated.
        endfin (int): Flag indicating final termination state.
        scission (int): Flag indicating if arm underwent chain scission.
        senio (int): Seniority level (generation number from first monomer).
        prio (int): Priority level (distance from chain end).
    """
    _fields_ = [
        ("arm_len", ct.c_double),
        ("arm_conv", ct.c_double),
        ("arm_time", ct.c_double),
        ("arm_tm", ct.c_double),
        ("arm_tddb", ct.c_double),
        ("L1", ct.c_int),
        ("L2", ct.c_int),
        ("R1", ct.c_int),
        ("R2", ct.c_int),
        ("up", ct.c_int),
        ("down", ct.c_int),
        ("armnum", ct.c_int),
        ("armcat", ct.c_int),
        ("ended", ct.c_int),
        ("endfin", ct.c_int),
        ("scission", ct.c_int),
        ("senio", ct.c_int),
        ("prio", ct.c_int),
    ]


class polymer(ct.Structure):
    """Representation of a complete polymer molecule.

    This structure represents a complete polymer molecule composed of multiple arms
    connected through branch points. It tracks the molecule's topology, molecular weight,
    and structural characteristics.

    Attributes:
        first_end (int): Index of the first chain end (arm) in this polymer.
        num_br (int): Number of branch points in this polymer molecule.
        bin (int): Molecular weight bin assignment for this polymer.
        num_sat (int): Number of saturated (terminated) chain ends.
        num_unsat (int): Number of unsaturated (active) chain ends.
        armnum (int): Index of the primary arm of this polymer.
        nextpoly (int): Index of the next polymer in linked list.
        tot_len (float): Total length of all arms in monomer units (molecular weight proxy).
        gfactor (float): G-factor (radius of gyration ratio) for this polymer architecture.
        saved (bool): Whether this polymer has been saved to output.
        max_senio (int): Maximum seniority level found in this polymer.
        max_prio (int): Maximum priority level found in this polymer.
    """
    _fields_ = [
        ("first_end", ct.c_int),
        ("num_br", ct.c_int),
        ("bin", ct.c_int),
        ("num_sat", ct.c_int),
        ("num_unsat", ct.c_int),
        ("armnum", ct.c_int),
        ("nextpoly", ct.c_int),
        ("tot_len", ct.c_double),
        ("gfactor", ct.c_double),
        ("saved", ct.c_bool),
        ("max_senio", ct.c_int),
        ("max_prio", ct.c_int),
    ]


class reactresults(ct.Structure):
    """Complete results from a polymer reaction simulation.

    This structure stores all computed properties and distributions from a polymerization
    reaction simulation, including molecular weight distributions, branching statistics,
    and architecture classification results.

    Attributes:
        wt (POINTER(float)): Pointer to weight fraction array for molecular weight bins.
        avbr (POINTER(float)): Pointer to average branching per bin array.
        wmass (POINTER(float)): Pointer to weight-average mass per bin array.
        avg (POINTER(float)): Pointer to average g-factor per bin array.
        lgmid (POINTER(float)): Pointer to log10 of bin center molecular weights.
        numinbin (POINTER(int)): Pointer to number of polymers in each MW bin.
        numin_armwt_bin (POINTER(int)): Pointer to number of arms in each arm weight bin.
        numin_num_br_bin (POINTER(int)): Pointer to number of polymers by branch point count.
        num_armwt_bin (int): Number of bins used for arm weight distribution.
        max_num_br (int): Maximum number of branch points found in any polymer.
        monmass (float): Monomer molecular weight (g/mol).
        M_e (float): Entanglement molecular weight (g/mol).
        N_e (float): Number of monomers per entanglement.
        boblgmin (float): Log10 of minimum molecular weight for BoB binning.
        boblgmax (float): Log10 of maximum molecular weight for BoB binning.
        m_w (float): Weight-average molecular weight (g/mol).
        m_n (float): Number-average molecular weight (g/mol).
        brav (float): Average number of branches per 1000 carbon atoms.
        m_z (float): Z-average molecular weight (g/mol).
        m_zp1 (float): Z+1 average molecular weight (g/mol).
        m_zp2 (float): Z+2 average molecular weight (g/mol).
        first_poly (int): Index of first polymer in linked list.
        next (int): Index of next available polymer slot.
        nummwdbins (int): Number of molecular weight distribution bins.
        numbobbins (int): Number of BoB (Branch-on-Branch) bins.
        bobbinmax (int): Maximum number of polymers per BoB bin.
        nsaved (int): Number of polymer configurations saved for BoB analysis.
        npoly (int): Total number of polymers generated in simulation.
        simnumber (int): Unique simulation identifier.
        polysaved (bool): Whether polymer configurations have been saved.
        name (bytes): Name/identifier for this simulation result.
        wlin (float): Total weight of linear polymers.
        wstar (float): Total weight of star polymers.
        wH (float): Total weight of H-shaped polymers.
        w7arm (float): Total weight of 7-arm polymers.
        wcomb (float): Total weight of comb polymers.
        wother (float): Total weight of other architectures.
        nlin (int): Number of linear polymers.
        nstar (int): Number of star polymers.
        nH (int): Number of H-shaped polymers.
        n7arm (int): Number of 7-arm polymers.
        ncomb (int): Number of comb polymers.
        nother (int): Number of other architectures.
        nsaved_arch (int): Number of polymers saved for architecture analysis.
        arch_minwt (float): Minimum molecular weight for architecture classification.
        arch_maxwt (float): Maximum molecular weight for architecture classification.
    """
    _fields_ = [
        ("wt", ct.POINTER(ct.c_double)),
        ("avbr", ct.POINTER(ct.c_double)),
        ("wmass", ct.POINTER(ct.c_double)),
        ("avg", ct.POINTER(ct.c_double)),
        ("lgmid", ct.POINTER(ct.c_double)),
        ("numinbin", ct.POINTER(ct.c_int)),
        ("numin_armwt_bin", ct.POINTER(ct.c_int)),
        ("numin_num_br_bin", ct.POINTER(ct.c_int)),
        ("num_armwt_bin", ct.c_int),
        ("max_num_br", ct.c_int),
        ("monmass", ct.c_double),
        ("M_e", ct.c_double),
        ("N_e", ct.c_double),
        ("boblgmin", ct.c_double),
        ("boblgmax", ct.c_double),
        ("m_w", ct.c_double),
        ("m_n", ct.c_double),
        ("brav", ct.c_double),
        ("m_z", ct.c_double),
        ("m_zp1", ct.c_double),
        ("m_zp2", ct.c_double),
        ("first_poly", ct.c_int),
        ("next", ct.c_int),
        ("nummwdbins", ct.c_int),
        ("numbobbins", ct.c_int),
        ("bobbinmax", ct.c_int),
        ("nsaved", ct.c_int),
        ("npoly", ct.c_int),
        ("simnumber", ct.c_int),
        ("polysaved", ct.c_bool),
        ("name", ct.c_char_p),
        ("wlin", ct.c_double),
        ("wstar", ct.c_double),
        ("wH", ct.c_double),
        ("w7arm", ct.c_double),
        ("wcomb", ct.c_double),
        ("wother", ct.c_double),
        ("nlin", ct.c_int),
        ("nstar", ct.c_int),
        ("nH", ct.c_int),
        ("n7arm", ct.c_int),
        ("ncomb", ct.c_int),
        ("nother", ct.c_int),
        ("nsaved_arch", ct.c_int),
        ("arch_minwt", ct.c_double),
        ("arch_maxwt", ct.c_double),
    ]


# global variable
pb_global_const = polybits_global_const.in_dll(react_lib, "pb_global_const")
pb_global = polybits_global.in_dll(react_lib, "pb_global")

# pointer
arm_pointer = ct.POINTER(arm)
arm_pointers = arm_pointer * (pb_global_const.maxarm + 1)

polymer_pointer = ct.POINTER(polymer)
polymer_pointers = polymer_pointer * (pb_global_const.maxpol + 1)

reactresults_pointer = ct.POINTER(reactresults)
reactresults_pointers = reactresults_pointer * (pb_global_const.maxreact + 1)

# function
react_pool_init = react_lib.react_pool_init
react_pool_init.restype = None

request_dist = react_lib.request_dist
request_dist.restype = ct.c_bool

return_dist_polys = react_lib.return_dist_polys
return_dist_polys.restype = None

return_dist = react_lib.return_dist
return_dist.restype = None

request_poly = react_lib.request_poly
request_poly.restype = ct.c_bool

# return_arm_pool = react_lib.return_arm_pool
# return_arm_pool.restype = arm_pointer

# return_br_poly = react_lib.return_br_poly
# return_br_poly.restype = polymer_pointer

return_react_dist = react_lib.return_react_dist
return_react_dist.restype = reactresults_pointer

set_br_poly_nextpoly = react_lib.set_br_poly_nextpoly
set_br_poly_nextpoly.restype = None

increase_arm_records_in_arm_pool = react_lib.increase_arm_records_in_arm_pool
increase_arm_records_in_arm_pool.restype = ct.c_bool

increase_polymer_records_in_br_poly = react_lib.increase_polymer_records_in_br_poly
increase_polymer_records_in_br_poly.restype = ct.c_bool

increase_dist_records_in_react_dist = react_lib.increase_dist_records_in_react_dist
increase_dist_records_in_react_dist.restype = ct.c_bool

set_do_prio_senio = react_lib.set_do_prio_senio
set_do_prio_senio.restype = None

set_flag_stop_all = react_lib.set_flag_stop_all
set_flag_stop_all.restype = None

init_bin_prio_vs_senio = react_lib.init_bin_prio_vs_senio
init_bin_prio_vs_senio.restype = None
init_bin_prio_vs_senio.argtypes = [ct.c_int]

return_avarmlen_v_senio = react_lib.return_avarmlen_v_senio
return_avarmlen_v_senio.restype = ct.c_double

return_avarmlen_v_prio = react_lib.return_avarmlen_v_prio
return_avarmlen_v_prio.restype = ct.c_double

return_avprio_v_senio = react_lib.return_avprio_v_senio
return_avprio_v_senio.restype = ct.c_double

return_avsenio_v_prio = react_lib.return_avsenio_v_prio
return_avsenio_v_prio.restype = ct.c_double

return_proba_prio = react_lib.return_proba_prio
return_proba_prio.restype = ct.c_double

return_max_prio = react_lib.return_max_prio
return_max_prio.restype = ct.c_int

return_proba_senio = react_lib.return_proba_senio
return_proba_senio.restype = ct.c_double

return_max_senio = react_lib.return_max_senio
return_max_senio.restype = ct.c_int

# initialise lists
react_dist = None


def link_react_dist():
    """link the Python list react_dist with the C array react_dist"""
    global reactresults_pointers
    global react_dist
    reactresults_pointers = reactresults_pointer * (pb_global_const.maxreact + 1)
    react_dist = reactresults_pointers(
        *list(
            [
                return_react_dist(ct.c_int(i))
                for i in range(pb_global_const.maxreact + 1)
            ]
        )
    )


react_pool_init()
link_react_dist()

###############
# tobitabatch.c
###############


# struct
class tobitabatch_global(ct.Structure):
    """Global state for Tobita batch polymerization simulations.

    This structure tracks the state of batch polymerization simulations using
    the Tobita algorithm for free-radical polymerization.

    Attributes:
        tobbatchnumber (int): Simulation identifier for current batch polymerization.
        tobitabatcherrorflag (bool): Error flag indicating if simulation encountered errors.
    """
    _fields_ = [("tobbatchnumber", ct.c_int), ("tobitabatcherrorflag", ct.c_bool)]


# global variable
tb_global = tobitabatch_global.in_dll(react_lib, "tb_global")

# function
tobbatchstart = react_lib.tobbatchstart
tobbatchstart.restype = None

tobbatch = react_lib.tobbatch
tobbatch.restype = ct.c_bool

###############
# binsandbob.c
###############


# struct
class binsandbob_global(ct.Structure):
    """Global results for multi-distribution binning and BoB analysis.

    This structure stores aggregate results when analyzing multiple polymer
    distributions together, combining their molecular weight distributions
    and branching statistics.

    Attributes:
        multi_m_w (float): Combined weight-average molecular weight (g/mol).
        multi_m_n (float): Combined number-average molecular weight (g/mol).
        multi_brav (float): Combined average branching per 1000 carbons.
        multi_nummwdbins (int): Number of bins used for combined distribution.
    """
    _fields_ = [
        ("multi_m_w", ct.c_double),
        ("multi_m_n", ct.c_double),
        ("multi_brav", ct.c_double),
        ("multi_nummwdbins", ct.c_int),
    ]


# global variable
bab_global = binsandbob_global.in_dll(react_lib, "bab_global")

# function
molbin = react_lib.molbin
molbin.restype = None

polyconfwrite = react_lib.polyconfwrite
polyconfwrite.restype = None

multipolyconfwrite = react_lib.multipolyconfwrite
multipolyconfwrite.restype = ct.c_ulonglong

multimolbin = react_lib.multimolbin
multimolbin.restype = None

return_binsandbob_multi_avbr = react_lib.return_binsandbob_multi_avbr
return_binsandbob_multi_avbr.restype = ct.c_double

return_binsandbob_multi_avg = react_lib.return_binsandbob_multi_avg
return_binsandbob_multi_avg.restype = ct.c_double

return_binsandbob_multi_lgmid = react_lib.return_binsandbob_multi_lgmid
return_binsandbob_multi_lgmid.restype = ct.c_double

return_binsandbob_multi_wmass = react_lib.return_binsandbob_multi_wmass
return_binsandbob_multi_wmass.restype = ct.c_double

return_binsandbob_multi_wt = react_lib.return_binsandbob_multi_wt
return_binsandbob_multi_wt.restype = ct.c_double

set_react_dist_monmass = react_lib.set_react_dist_monmass
set_react_dist_monmass.restype = ct.c_double

set_react_dist_M_e = react_lib.set_react_dist_M_e
set_react_dist_M_e.restype = ct.c_double

###############
# tobitaCSTR.c
###############


# struct
class tobitaCSTR_global(ct.Structure):
    """Global state for Tobita CSTR polymerization simulations.

    This structure tracks the state of continuous stirred-tank reactor (CSTR)
    polymerization simulations using the Tobita algorithm.

    Attributes:
        tobCSTRnumber (int): Simulation identifier for current CSTR polymerization.
        tobitaCSTRerrorflag (bool): Error flag indicating if simulation encountered errors.
    """
    _fields_ = [("tobCSTRnumber", ct.c_int), ("tobitaCSTRerrorflag", ct.c_bool)]


# global variable
tCSTR_global = tobitaCSTR_global.in_dll(react_lib, "tCSTR_global")

# function
tobCSTRstart = react_lib.tobCSTRstart
tobCSTRstart.restype = None

tobCSTR = react_lib.tobCSTR
tobCSTR.restype = ct.c_bool

###############
# dieneCSTR.c
###############


# struct
class dieneCSTR_global(ct.Structure):
    """Global state for diene CSTR polymerization simulations.

    This structure tracks the state of continuous stirred-tank reactor (CSTR)
    polymerization simulations involving diene monomers.

    Attributes:
        dieneCSTRnumber (int): Simulation identifier for current diene CSTR polymerization.
        dieneCSTRerrorflag (bool): Error flag indicating if simulation encountered errors.
    """
    _fields_ = [("dieneCSTRnumber", ct.c_int), ("dieneCSTRerrorflag", ct.c_bool)]


# global variable
dCSTR_global = dieneCSTR_global.in_dll(react_lib, "dCSTR_global")

# function
dieneCSTRstart = react_lib.dieneCSTRstart
dieneCSTRstart.restype = None

dieneCSTR = react_lib.dieneCSTR
dieneCSTR.restype = ct.c_bool

################
# MultiMetCSTR.c
################


# struct
class mulmetCSTR_global(ct.Structure):
    """Global state for multi-metallocene CSTR polymerization simulations.

    This structure tracks the state of continuous stirred-tank reactor (CSTR)
    polymerization simulations using multiple metallocene catalysts.

    Attributes:
        mulmetCSTRnumber (int): Simulation identifier for current multi-metallocene CSTR.
        mulmetCSTRerrorflag (bool): Error flag indicating if simulation encountered errors.
    """
    _fields_ = [("mulmetCSTRnumber", ct.c_int), ("mulmetCSTRerrorflag", ct.c_bool)]


# global variable
MMCSTR_global = mulmetCSTR_global.in_dll(react_lib, "MMCSTR_global")

# function
mulmetCSTRstart = react_lib.mulmetCSTRstart
mulmetCSTRstart.restype = None

mulmetCSTR = react_lib.mulmetCSTR
mulmetCSTR.restype = ct.c_bool

#############
# Other
############


def end_print(parent_theory, ndist, do_architecture):
    """Print comprehensive simulation results after reaction simulation completes.

    Outputs a formatted table of molecular weight averages, branching statistics,
    and architecture distribution (if enabled) to the theory's message panel.

    Args:
        parent_theory: The parent theory object providing Qprint method for output.
        ndist (int): Index of the reaction distribution in the react_dist array.
        do_architecture (bool): Whether to include architecture analysis in output.
            When True, prints distribution of polymer topologies (linear, star, H, etc.)
            within the specified molecular weight range.
    """
    parent_theory.Qprint("<b>Simulation Results:</b>")

    table = []
    table.append(["", ""])  # no header
    table.append(["Polymer made", "%d" % react_dist[ndist].contents.npoly])
    table.append(["Polymer saved", "%d" % react_dist[ndist].contents.nsaved])
    table.append(["Arm left in memory", "%d" % pb_global.arms_left])
    table.append(["Mn (g/mol)", "%.3g" % react_dist[ndist].contents.m_n])
    table.append(["Mw (g/mol)", "%.3g" % react_dist[ndist].contents.m_w])
    table.append(["Mz (g/mol)", "%.3g" % react_dist[ndist].contents.m_z])
    table.append(["Mz+1 (g/mol)", "%.3g" % react_dist[ndist].contents.m_zp1])
    table.append(["Mz+2 (g/mol)", "%.3g" % react_dist[ndist].contents.m_zp2])
    table.append(["Br/1000C", "%.3g" % react_dist[ndist].contents.brav])
    table.append(["Max br", "%d" % react_dist[ndist].contents.max_num_br])
    parent_theory.Qprint(table)

    if do_architecture:
        nlin = react_dist[ndist].contents.nlin
        nstar = react_dist[ndist].contents.nstar
        nH = react_dist[ndist].contents.nH
        n7arm = react_dist[ndist].contents.n7arm
        ncomb = react_dist[ndist].contents.ncomb
        nother = react_dist[ndist].contents.nother
        #
        wlin = react_dist[ndist].contents.wlin
        wstar = react_dist[ndist].contents.wstar
        wH = react_dist[ndist].contents.wH
        w7arm = react_dist[ndist].contents.w7arm
        wcomb = react_dist[ndist].contents.wcomb
        wother = react_dist[ndist].contents.wother

        name_list = ["Linear", "Star", "H", "7-arm", "Comb", "Other"]
        nlist = [nlin, nstar, nH, n7arm, ncomb, nother]
        wlist = [wlin, wstar, wH, w7arm, wcomb, wother]
        for i, n in enumerate(nlist):
            if n != 0:
                wlist[i] = wlist[i] / n

        norm = react_dist[ndist].contents.nsaved_arch / 100
        if norm != 0:
            parent_theory.Qprint(
                "<b>Architecture of %d Polymers: %.3g &lt; M &lt; %.3g g/mol:</b>"
                % (
                    react_dist[ndist].contents.nsaved_arch,
                    parent_theory.xmin,
                    parent_theory.xmax,
                )
            )
            table = """<table border="1" width="100%">"""
            table += (
                """<tr><th>Type</th><th>Prop.</th><th>&lt;Mw&gt; (g/mol)</th></tr>"""
            )
            for i in range(len(nlist)):
                table += """<tr><td>%s</td><td>%.3g%%</td><td>%.3g</td></tr>""" % (
                    name_list[i],
                    nlist[i] / norm,
                    wlist[i],
                )
            table += """</table><br>"""
            parent_theory.Qprint(table)


def prio_and_senio(parent_theory, f, ndist, do_architecture):
    """Extract and store priority/seniority analysis from C library to DataTable.

    Retrieves arm length probability distributions, branch point statistics, and
    priority-seniority relationships from the C simulation results and stores them
    in the theory's DataTable extra_tables dictionary for visualization.

    Args:
        parent_theory: The parent theory object containing tables dictionary.
        f: File object with file_name_short attribute used as table key.
        ndist (int): Index of the reaction distribution in the react_dist array.
        do_architecture (bool): Whether to include detailed priority/seniority analysis.
            When True, extracts and stores: avarmlen_v_senio, avarmlen_v_prio,
            avprio_v_senio, avsenio_v_prio, proba_senio, and proba_prio distributions.
    """
    tt = parent_theory.tables[f.file_name_short]
    # arm length
    lgmax = np.log10(react_dist[ndist].contents.arch_maxwt * 1.01)
    lgmin = np.log10(react_dist[ndist].contents.monmass / 1.01)
    num_armwt_bin = react_dist[ndist].contents.num_armwt_bin
    lgstep = (lgmax - lgmin) / (1.0 * num_armwt_bin)
    tmp_x = np.power(
        10,
        [lgmin + ibin * lgstep - 0.5 * lgstep for ibin in range(1, num_armwt_bin + 1)],
    )
    tmp_y = [
        react_dist[ndist].contents.numin_armwt_bin[ibin]
        for ibin in range(1, num_armwt_bin + 1)
    ]
    # trim right zeros
    tmp_y = np.trim_zeros(tmp_y, "b")
    new_len = len(tmp_y)
    tmp_x = tmp_x[:new_len]
    # trim left zeros
    tmp_y = np.trim_zeros(tmp_y, "f")
    new_len = len(tmp_y)
    tmp_x = tmp_x[-new_len:]

    tt.extra_tables["proba_arm_wt"] = np.zeros((new_len, 2))
    tt.extra_tables["proba_arm_wt"][:, 0] = tmp_x
    tt.extra_tables["proba_arm_wt"][:, 1] = tmp_y
    # normalize
    try:
        tt.extra_tables["proba_arm_wt"][:, 1] /= tt.extra_tables["proba_arm_wt"][
            :, 1
        ].sum()
    except ZeroDivisionError:
        pass

    # number of branch points branch point
    max_num_br = react_dist[ndist].contents.max_num_br

    # if max_num_br < 100:
    rmax = min(max_num_br + 1, pb_global_const.MAX_NBR)
    tt.extra_tables["proba_br_pt"] = np.zeros((max_num_br + 1, 2))
    tt.extra_tables["proba_br_pt"][:, 0] = np.arange(max_num_br + 1)
    tt.extra_tables["proba_br_pt"][:, 1] = [
        react_dist[ndist].contents.numin_num_br_bin[i] for i in range(max_num_br + 1)
    ]
    try:
        tt.extra_tables["proba_br_pt"][:, 1] /= tt.extra_tables["proba_br_pt"][
            :, 1
        ].sum()
    except ZeroDivisionError:
        pass
    # else:
    #     # bin the data
    #     tmp_x = list(np.arange(max_num_br + 1))
    #     tmp_y = [react_dist[ndist].contents.numin_num_br_bin[i] for i in range(max_num_br + 1)]
    #     hist, bin_edge = np.histogram(tmp_x, bins=20, weights=tmp_y, density=True)
    #     tt.extra_tables['proba_br_pt'] = np.zeros((len(hist), 2))
    #     tt.extra_tables['proba_br_pt'][:, 0] = np.diff(bin_edge) / 2 + bin_edge[:-1]
    #     tt.extra_tables['proba_br_pt'][:, 1] = hist

    if not do_architecture:
        return
    # P&S
    max_prio = return_max_prio()
    max_senio = return_max_senio()

    avarmlen_v_senio = [
        return_avarmlen_v_senio(ct.c_int(s), ct.c_int(ndist))
        for s in range(1, max_senio + 1)
    ]
    avarmlen_v_prio = [
        return_avarmlen_v_prio(ct.c_int(p), ct.c_int(ndist))
        for p in range(1, max_prio + 1)
    ]

    avprio_v_senio = [
        return_avprio_v_senio(ct.c_int(s)) for s in range(1, max_senio + 1)
    ]
    avsenio_v_prio = [
        return_avsenio_v_prio(ct.c_int(p)) for p in range(1, max_prio + 1)
    ]

    proba_senio = [return_proba_senio(ct.c_int(s)) for s in range(1, max_senio + 1)]
    proba_prio = [return_proba_prio(ct.c_int(p)) for p in range(1, max_prio + 1)]

    tt.extra_tables["avarmlen_v_senio"] = np.zeros((max_senio, 2))
    tt.extra_tables["avarmlen_v_senio"][:, 0] = np.arange(1, max_senio + 1)
    tt.extra_tables["avarmlen_v_senio"][:, 1] = avarmlen_v_senio[:]

    tt.extra_tables["avarmlen_v_prio"] = np.zeros((max_prio, 2))
    tt.extra_tables["avarmlen_v_prio"][:, 0] = np.arange(1, max_prio + 1)
    tt.extra_tables["avarmlen_v_prio"][:, 1] = avarmlen_v_prio[:]

    tt.extra_tables["avprio_v_senio"] = np.zeros((max_senio, 2))
    tt.extra_tables["avprio_v_senio"][:, 0] = np.arange(1, max_senio + 1)
    tt.extra_tables["avprio_v_senio"][:, 1] = avprio_v_senio[:]

    tt.extra_tables["avsenio_v_prio"] = np.zeros((max_prio, 2))
    tt.extra_tables["avsenio_v_prio"][:, 0] = np.arange(1, max_prio + 1)
    tt.extra_tables["avsenio_v_prio"][:, 1] = avsenio_v_prio[:]

    tt.extra_tables["proba_senio"] = np.zeros((max_senio, 2))
    tt.extra_tables["proba_senio"][:, 0] = np.arange(1, max_senio + 1)
    tt.extra_tables["proba_senio"][:, 1] = proba_senio[:]

    tt.extra_tables["proba_prio"] = np.zeros((max_prio, 2))
    tt.extra_tables["proba_prio"][:, 0] = np.arange(1, max_prio + 1)
    tt.extra_tables["proba_prio"][:, 1] = proba_prio[:]
