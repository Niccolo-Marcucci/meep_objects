#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
% Copyright 2020 Niccolò Marcucci <niccolo.marcucci@polito.it>
%
% Licensed under the Apache License, Version 2.0 (the "License")
% you may not use this file except in compliance with the License.
% You may obtain a copy of the License at
%
%     http://www.apache.org/licenses/LICENSE-2.0
%
% Unless required by applicable law or agreed to in writing, software
% distributed under the License is distributed on an "AS IS" BASIS,
% WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
% See the License for the specific language governing permissions and
% limitations under the License.
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
"""

import numpy as np
import math
from math import pi, cos, sin, tan, atan, atan2, sqrt, exp
from scipy import signal, io, interpolate as itp, optimize as opt
from tqdm import tqdm

def prepare_multilayer(d,n):
    """
    % This function is used for deleting all the layers that are set to
    % have zero thickness. Moreover, first (substrate) and last (air) are
    % set to have zero thickness, so that computations can be performed on
    % the stack only.
    % Dummy layers at end and beginning are removed (layers that have same
    % index than air or substrate)
    """

    N_l = np.size(d)
    d_start = d[0]
    d_end = d[-1]
    d[0] = 0
    d[-1] = 0

    # remove dummy layers at beginning/end, and set remaining first and
    # last layer with zero thickness
    i = 1
    while n[i] == n[0] or float(d[i]) == 0.:         # exit when encounter
        d[i] = 0                                    # the first layer with
        i += 1                                      # n~=n[1] that has
                                                    # thickness

    i = -2
    while np.real(n[i]) == np.real(n[-1]) or float(d[i]) == 0.:
        d[i] = 0
        i -= 1

    d_new = []
    n_new = []
    for i, l in enumerate(d):
        if float(l) != 0. or i == 0 or i == N_l-1:
            d_new.append(l)
            n_new.append(n[i])

    n = np.array(n_new)
    d = np.array(d_new)

    return d, n, d_start, d_end

def reflectivity (wavelengths, thetas, d,n,pol) : # [R,r,t,Tr] =
    """
    % This function computes the reflectivity and trasmissivity of a
    % dielectric multilayer stack. The multilayer vector has to include
    % the substrate (as first element) and the external medium (as lat
    % layer). The thickness of the latter will not matter, since the
    % computation will end on the right side of the last interface.
    %  The multilayer stack should not contain layers having zero
    %  thickness. In order to remove them see the function
    %  "prepare_multilayer.m"
    %
    % If the first layer thickness is set to zero the input field (and the
    % computed reflected field) will be located on the left side of the
    % first interface. Same for the last layer.
    """

    N_layers = np.size(d)
    N_w = np.size(wavelengths)
    N_t = np.size(thetas)

    # for n_l in n :
    #     if np.size(n_l) > 1 and np.size(n_l) == N_w:
    #         print(n_l)
    #     elif np.size(n_l) == 1:
    #         print(n_l)
    #     else:
    #         raise ValueError("if the index is dependent on wavength, it should have same size as wavelength vector.")

    # reshaping is necessary for allowing scalars as input
    wavelengths = np.array(wavelengths).reshape(N_w)
    thetas = np.array(thetas).reshape(N_t)

    r = np.zeros( (N_w, N_t), dtype=complex)
    t = np.zeros( (N_w, N_t), dtype=complex)
    R = np.zeros( (N_w, N_t), dtype=float )
    T = np.zeros( (N_w, N_t), dtype=float )

    for i in range( N_w ):
        for j in range( N_t):
            Lambda = wavelengths[i]

            theta_in = thetas[j] / 180*pi

            K = 2*pi/Lambda

            # if np.size(theta_in) > 1 or np.size(Lambda) > 1:
            #      raise ValueError("Lambda or theta should be scalar")

            # transverse wavevector.
            beta = n[0]*sin(theta_in)
            costheta_z = np.sqrt(n**2-beta**2)/n

            T11 = 1
            T12 = 0
            T21 = 0
            T22 = 1

            for k in range(N_layers-1) :
                n_i = n[k]
                n_j = n[k+1]
                costheta_i = costheta_z[k]
                costheta_j = costheta_z[k+1]
                kz = K*n_i*costheta_i
                Pr = np.exp(+1j*kz*d[k])
                Pl = np.exp(-1j*kz*d[k])

                # Fresnel coefficients
                if pol == 's' :
                    rij = (n_i*costheta_i-n_j*costheta_j)/(n_i*costheta_i+n_j*costheta_j)
                    rji = -rij
                    tji =  rji + 1
                elif pol == 'p' :
                    rij = (n_j*costheta_i-n_i*costheta_j)/(n_j*costheta_i+n_i*costheta_j)
                    rji = -rij
                    tji = (rji + 1)*n_j/n_i
                else:
                   raise ValueError("Invalid Polarization. Valid options are 's' or 'p'")

                # expicit matrix product for increasing speed.
                rtij = rji/tji
                T11t = Pr/tji*T11 + rtij*Pl*T21
                T12t = Pr/tji*T12 + rtij*Pl*T22
                T21t = rtij*Pr*T11 + Pl/tji*T21
                T22t = rtij*Pr*T12 + Pl/tji*T22

                T11 = T11t
                T12 = T12t
                T21 = T21t
                T22 = T22t

            rr = -T21 / T22

            tt = (T11 + rr*T12) #* np.exp(+1j*kz*d[-1]) # include propagation in the last layer

            r[i, j] = rr

            t[i, j] = tt

            T[i, j] = np.abs(tt)**2 * np.real(n[-1]/n[0] * costheta_z[-1]/costheta_z[0])


            R[i, j] = np.abs(rr)**2

    if N_w == 1 and N_t > 1:
        R = R.reshape( (N_t,) )
        T = T.reshape( (N_t,) )
        r = r.reshape( (N_t,) )
        t = t.reshape( (N_t,) )
    elif N_w > 1 and N_t == 1:
        R = R.reshape( (N_w,) )
        T = T.reshape( (N_w,) )
        r = r.reshape( (N_w,) )
        t = t.reshape( (N_w,) )
    elif N_w == 1 and N_t == 1:
        R = R[0]
        T = T[0]
        r = r[0]
        t = t[0]

    return R, r, t , T

def extract_bsw_dispersion(lambda_v, last_layer_info, design_file , pol, n_eff_range=(1, 1.15, 1000)):
    def lorenzian(x, p0, p1, p2):
        return p0 / ((x - p1)**2 + p2)

    data = io.loadmat(design_file)

    data["d_layers"][-2] = last_layer_info[0]
    data["idx_layers"][-2] = last_layer_info[1] + 1j*1e-4

    d, n, _, _ = prepare_multilayer(data["d_layers"], data["idx_layers"])

    n_eff_v = np.linspace(*n_eff_range)
    theta_v = np.real(np.arcsin(n_eff_v / n[0])) / pi*180 + 0.001

    n_eff_real = np.ones(len(lambda_v))
    n_eff_imag = np.zeros(len(lambda_v))

    for i, wavelength in tqdm(enumerate(lambda_v)):
        R, _, _, _ = reflectivity(wavelength, theta_v, d, n, pol)
        profile = 1 - R
        ix = np.argmax(profile)
        n_eff_real[i] = n_eff_v[ix]
        # try :
        #     popt, pcov = opt.curve_fit(lorenzian, n_eff_v, profile, p0=(profile.max()*0.01, n_eff_v[ix] , .001),
        #                                                             bounds = ([0,                      n_eff_v[ix]-0.01,   0 ],
        #                                                                       [profile.max()*0.01*1.1, n_eff_v[ix]+0.01, 0.01]))
        # except RuntimeError:
        #     print(f"didn't find opt param for wavelength {wavelength*1e9:.1f}")
        # else:
        #     n_eff_real[i] = popt[1]
        #     # n_eff_imag[i] = 3*sqrt(popt[2])

        #     import matplotlib.pyplot as plt
        #     plt.figure()
        #     plt.plot(n_eff_v, profile, n_eff_v, lorenzian(n_eff_v, *popt))

    return n_eff_real, n_eff_imag
