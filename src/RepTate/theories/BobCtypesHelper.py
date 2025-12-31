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
from ctypes import (
    CFUNCTYPE,
    c_double,
    c_int,
    c_char_p,
    byref,
    c_bool,
    c_void_p,
    POINTER,
    c_char,
)
import sys
import os

from RepTate.core.ctypes_loader import load_ctypes_library
if sys.platform == "darwin" or sys.platform == "linux":
    CHARCODE = "utf-8"
else:
    CHARCODE = "latin-1"

class BobError(Exception):
    """Class for BoB exceptions"""

    pass


class BobCtypesHelper:
    """Wrapper class to call BoB C++ functions"""

    CB_FTYPE_NONE_PCHAR = CFUNCTYPE(
        None, POINTER(c_char), c_int
    )  # callback [return type, [args types]]

    CB_FTYPE_NONE_CHAR = CFUNCTYPE(
        None, c_char_p
    )  # callback [return type, [args types]]

    CB_FTYPE_DOUBLE_NONE = CFUNCTYPE(
        c_double, c_void_p
    )  # callback [return type, [args types]]

    def __init__(self, parent_theory):

        self.parent_theory = parent_theory
        # get the directory path of current file
        dir_path = os.path.dirname(os.path.realpath(__file__))
        # load the sharedlibrary
        if sys.maxsize > 2 ** 32:
            # 64-bit system
            self.lib_path = os.path.join(dir_path, "bob2p5_lib_%s.so" % (sys.platform))
        else:
            # 32-bit system
            self.lib_path = os.path.join(dir_path, "bob2p5_lib_%s_i686.so" % (sys.platform))
        self.bob_lib = load_ctypes_library(self.lib_path, "BoB library")
        # link the C function to Python
        self.link_c_functions()

    def send_string(self, pointer_to_str, case):
        """BoB calls this function to send a string to C
        case 0: send filename containing polyconf input 
        case 1: send polymer name (max 9 caracters)
        """
        if case == 0:
            s = self.parent_theory.from_file_filename[
                self.parent_theory.from_file_filename_counter
            ]
            for i, c in enumerate(s):
                pointer_to_str[i] = c.encode(CHARCODE)
            self.parent_theory.from_file_filename_counter += 1
        elif case == 1:
            s = self.parent_theory.protoname[self.parent_theory.protoname_counter]
            for i, c in enumerate(s):
                pointer_to_str[i] = c.encode(CHARCODE)
            self.parent_theory.protoname_counter += 1

    def get_next_item_from_inp_file(self, *arg):
        """BoB calls this function to read the 'virtual' inp file.

        This callback function is invoked by the BoB C++ library to sequentially
        retrieve input parameters from a virtual input file stored in memory.

        Args:
            *arg: Variable arguments (unused, for compatibility with C callback signature).

        Returns:
            float or int: The next value from the virtual input file at the current counter position.
        """
        val = self.parent_theory.virtual_input_file[self.parent_theory.inp_counter]
        self.parent_theory.inp_counter += 1
        return val

    def get_next_item_from_proto_file(self, *arg):
        """BoB calls this function to read the 'virtual' proto file.

        This callback function is invoked by the BoB C++ library to sequentially
        retrieve protocol parameters from a virtual proto file stored in memory.

        Args:
            *arg: Variable arguments (unused, for compatibility with C callback signature).

        Returns:
            float or int: The next value from the virtual proto file at the current counter position.
        """
        val = self.parent_theory.virtual_proto_file[self.parent_theory.proto_counter]
        self.parent_theory.proto_counter += 1
        return val

    def get_freqmin(self, *arg):
        """BoB LVE calls this function to get the min frequency.

        This callback function is invoked by the BoB C++ library during LVE
        calculations to obtain the minimum frequency for the frequency sweep.

        Args:
            *arg: Variable arguments (unused, for compatibility with C callback signature).

        Returns:
            float: The minimum frequency value for the frequency sweep.
        """
        return self.parent_theory.freqmin

    def get_freqmax(self, *arg):
        """BoB LVE calls this function to get the max frequency.

        This callback function is invoked by the BoB C++ library during LVE
        calculations to obtain the maximum frequency for the frequency sweep.

        Args:
            *arg: Variable arguments (unused, for compatibility with C callback signature).

        Returns:
            float: The maximum frequency value for the frequency sweep.
        """
        return self.parent_theory.freqmax

    def get_freqint(self, *arg):
        """BoB LVE calls this function to get the frequency interval.

        This callback function is invoked by the BoB C++ library during LVE
        calculations to obtain the frequency interval spacing factor.

        Args:
            *arg: Variable arguments (unused, for compatibility with C callback signature).

        Returns:
            float: The frequency interval spacing factor (theory points spaced by log10(freqint)).
        """
        return self.parent_theory.freqint

    def print_err_from_c(self, char):
        """Function called by BoB from the C++ code when error occurs.

        This callback function is invoked by the BoB C++ library to report
        errors during execution. The error message is formatted and displayed
        to the user via the parent theory's Qprint method.

        Args:
            char (c_char_p): C-style character pointer containing the error message
                from the BoB library.
        """
        err_msg = "<b>ERROR encountered in BoB:</b><br>%s<hr>" % (char.decode())
        self.parent_theory.Qprint(err_msg)

    def print_from_c(self, char):
        """Function called by BoB from the C++ code during normal execution.

        This callback function is invoked by the BoB C++ library to report
        progress messages and information during normal execution. The message
        is decoded and displayed to the user via the parent theory's Qprint method.

        Args:
            char (c_char_p): C-style character pointer containing the message
                from the BoB library.
        """
        msg = "%s" % (char.decode())
        self.parent_theory.Qprint(msg)

    def link_c_functions(self):
        """Declare the Python functions equivalents to the C functions"""
        self.bob_save_polyconf_and_return_gpc = (
            self.bob_lib.reptate_save_polyconf_and_return_gpc
        )
        self.bob_save_polyconf_and_return_gpc.restype = c_bool

        self.run_bob_lve = self.bob_lib.run_bob_lve
        self.run_bob_lve.restype = c_bool

        self.get_bob_lve = self.bob_lib.get_bob_lve
        self.get_bob_lve.restype = c_bool

        self.run_bob_nlve = self.bob_lib.run_bob_nlve
        self.run_bob_nlve.restype = c_bool

        self.get_bob_nlve_results = self.bob_lib.get_bob_nlve_results
        self.get_bob_nlve_results.restype = c_bool

        # ask BoB to stop calculations
        self.set_flag_stop_bob = self.bob_lib.set_flag_stop_bob
        self.set_flag_stop_bob.restype = None

        # are priority and seniority calculated
        self.set_do_priority_seniority = self.bob_lib.set_do_priority_seniority
        self.set_do_priority_seniority.restype = None

        self.link_c_callback()

    def link_c_callback(self):
        """Callback C function from Python function.
        Must call it before each run to make sure the prints and calls are directed
        towards the correct theory.
        """
        self.cb_err_func = self.CB_FTYPE_NONE_CHAR(self.print_err_from_c)
        self.bob_lib.def_pyprint_err_func(self.cb_err_func)

        self.cb_func = self.CB_FTYPE_NONE_CHAR(self.print_from_c)
        self.bob_lib.def_pyprint_func(self.cb_func)

        self.cb_get_freqmin = self.CB_FTYPE_DOUBLE_NONE(self.get_freqmin)
        self.bob_lib.def_get_freqmin(self.cb_get_freqmin)

        self.cb_get_freqmax = self.CB_FTYPE_DOUBLE_NONE(self.get_freqmax)
        self.bob_lib.def_get_freqmax(self.cb_get_freqmax)

        self.cb_get_freqint = self.CB_FTYPE_DOUBLE_NONE(self.get_freqint)
        self.bob_lib.def_get_freqint(self.cb_get_freqint)

        self.cb_get_next_item_from_inp_file = self.CB_FTYPE_DOUBLE_NONE(
            self.get_next_item_from_inp_file
        )
        self.bob_lib.def_get_next_item_from_inp_file(
            self.cb_get_next_item_from_inp_file
        )

        self.cb_get_next_item_from_proto_file = self.CB_FTYPE_DOUBLE_NONE(
            self.get_next_item_from_proto_file
        )
        self.bob_lib.def_get_next_item_from_proto_file(
            self.cb_get_next_item_from_proto_file
        )

        self.cb_send_string = self.CB_FTYPE_NONE_PCHAR(self.send_string)
        self.bob_lib.def_get_string(self.cb_send_string)

    def save_polyconf_and_return_gpc(self, arg_list, npol_tot):
        """Run BoB asking for a polyconf file only and output polymer characteristics.

        Executes the BoB C++ library to generate a polymer configuration file
        without performing relaxation calculations. Returns gel permeation
        chromatography (GPC) data including molecular weight distribution,
        branching distribution, and g-factor distribution.

        Args:
            arg_list (list of str): Command-line arguments for the BoB executable,
                typically ["./bob", "-i", input_file, "-c", config_file].
            npol_tot (int): Total number of polymers in the configuration.

        Returns:
            list: A list containing [mn, mw, arrs] where:
                - mn (float): Number-average molecular weight.
                - mw (float): Weight-average molecular weight.
                - arrs (list of list): Four arrays containing [lgmid, wtbin, brbin, gbin]
                    representing log molecular weight bins, weight distribution,
                    branching distribution, and g-factor distribution.

        Raises:
            BobError: If an error occurs during BoB execution.
        """
        # prepare the arguments for bob_main function
        n_arg = len(arg_list)
        argv = (c_char_p * n_arg)()
        for i in range(n_arg):
            argv[i] = arg_list[i].encode(CHARCODE)

        # virtual inp and proto file
        self.parent_theory.inp_counter = 0
        self.parent_theory.proto_counter = 0
        self.parent_theory.from_file_filename_counter = 0
        self.parent_theory.protoname_counter = 0

        # prepare the arguments for GPC
        nbin = self.parent_theory.parameters[
            "nbin"
        ].value  # set the number of bins from param
        ncomp = c_int(-1)  # -1: all components
        ni = c_int(0)
        nf = c_int(npol_tot)  # all polymers
        lgmid_arr = (c_double * nbin)()
        wtbin_arr = (c_double * nbin)()
        brbin_arr = (c_double * nbin)()
        gbin_arr = (c_double * nbin)()
        mn = c_double()
        mw = c_double()

        # call C function, return False if error in BoB
        if self.bob_save_polyconf_and_return_gpc(
            c_int(n_arg),
            argv,
            c_int(nbin),
            ncomp,
            ni,
            nf,
            byref(mn),
            byref(mw),
            lgmid_arr,
            wtbin_arr,
            brbin_arr,
            gbin_arr,
        ):

            # return results
            arrs = [lgmid_arr[:], wtbin_arr[:], brbin_arr[:], gbin_arr[:]]
            return [mn.value, mw.value, arrs]

        # BoB encountered error
        raise BobError

    def return_bob_lve(self, arg_list):
        """Run BoB LVE and copy results to arrays.

        Executes the BoB C++ library to perform linear viscoelastic (LVE) calculations
        for a polymer configuration. Returns frequency-dependent storage and loss moduli.

        Args:
            arg_list (list of str): Command-line arguments for the BoB executable,
                typically ["./bob", "-i", input_file, "-c", config_file].

        Returns:
            list: A list containing [omega, g_p, g_pp] where:
                - omega (list of float): Angular frequency values (rad/s).
                - g_p (list of float): Storage modulus G' values (Pa).
                - g_pp (list of float): Loss modulus G'' values (Pa).

        Raises:
            BobError: If an error occurs during BoB execution.
        """
        # virtual inp file
        self.parent_theory.inp_counter = 0
        # prepare the arguments for bob_main function
        n_arg = len(arg_list)
        argv = (c_char_p * n_arg)()
        for i in range(n_arg):
            argv[i] = arg_list[i].encode(CHARCODE)

        # run bob LVE and get size of results
        # call C function, return False if error in BoB
        out_size = c_int()
        if self.run_bob_lve(c_int(n_arg), argv, byref(out_size)):
            # allocate Python memory for results and copy bob results
            omega = (c_double * out_size.value)()
            g_p = (c_double * out_size.value)()
            g_pp = (c_double * out_size.value)()

            if self.get_bob_lve(omega, g_p, g_pp):
                return [omega[:], g_p[:], g_pp[:]]

        # BoB encountered error
        raise BobError

    def return_bob_nlve(self, arg_list, flowrate, tmin, tmax, is_shear):
        """Run BoB NLVE and copy results to arrays.

        Executes the BoB C++ library to perform nonlinear viscoelastic (NLVE) calculations
        for a polymer configuration under shear or extensional flow. Returns time-dependent
        stress evolution.

        Args:
            arg_list (list of str): Command-line arguments for the BoB executable,
                typically ["./bob", "-i", input_file, "-c", config_file].
            flowrate (float): Flow rate (shear rate or extension rate) in s^-1.
            tmin (float): Minimum time for the calculation (s).
            tmax (float): Maximum time for the calculation (s).
            is_shear (bool): True for shear flow, False for extensional flow.

        Returns:
            list: A list containing [time_arr, stress_arr] where:
                - time_arr (list of float): Time values (s).
                - stress_arr (list of float): Stress values (Pa) - shear stress for shear flow,
                    extensional stress for extensional flow.

        Raises:
            BobError: If an error occurs during BoB execution.
        """
        # virtual inp file
        self.parent_theory.inp_counter = 0
        # prepare the arguments for bob_main function
        n_arg = len(arg_list)
        argv = (c_char_p * n_arg)()
        for i in range(n_arg):
            argv[i] = arg_list[i].encode(CHARCODE)

        # run bob and get size of results
        # call C function, return False if error in BoB
        out_size = c_int()
        if self.run_bob_nlve(
            c_int(n_arg),
            argv,
            c_double(flowrate),
            c_double(tmin),
            c_double(tmax),
            c_bool(is_shear),
            byref(out_size),
        ):
            # allocate Python memory for results and copy bob results
            time_arr = (c_double * out_size.value)()
            stress_arr = (c_double * out_size.value)()
            N1_arr = (c_double * out_size.value)()

            if self.get_bob_nlve_results(
                time_arr, stress_arr, N1_arr, c_bool(is_shear)
            ):
                return [time_arr[:], stress_arr[:]]

        # BoB encountered error
        raise BobError
