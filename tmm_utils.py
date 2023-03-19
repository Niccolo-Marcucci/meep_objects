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
            lambda_ = wavelengths[i]

            theta_in = thetas[j] / 180*pi

            K = 2*pi/lambda_

            # if np.size(theta_in) > 1 or np.size(lambda_) > 1:
            #      raise ValueError("lambda_ or theta should be scalar")

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

def field_distribution(lambda_, theta, d, n, pol, res=50):
    '''
    % Computes the field distribution inside a dielectric multilayer stack.
    %
    % - The output P is the normalized power density: |Hx|^2/|H0|^2 for
    %   p-polarization and |Ex|^2 for s-polarization
    % - The resolution parameter (in input) can be neglected
    % - The function can be used also to extract only nz. in this case use
    %   in the form:
    %   field_distribution(lambda_,theta,d,n,'',res)
    '''

    d, n, dsub, dair = prepare_multilayer(d,n);  # check fun description

    # determining an optimal resolution, 50 points on shortest wavelegth
    step = lambda_/max(np.real(n))/res
    sz = round(sum(d)/step)          # size of z vector
    z  = np.linspace(0, sum(d), sz)
    nz = np.ones(sz, dtype=complex)
    step = z[1] - z[0]

    # create nz vector: if a point is located exactly at an interface, it
    # will be cosidered to belong to the layer on the right hand side of
    # the interface
    j = 0
    zv = np.cumsum(d)
    za = zv[j]
    zb = zv[j+1]
    for i in range(1, sz):
        if abs(z[i] - zb) <= step/2:
            j = j+1
            z[i] = zb
            za = zv[j]
            zb = zv[j+1]
            nz[i] = n[j+1]
        elif (z[i] > za) and (z[i] < zb):
            nz[i] = n[j+1]

    # enforce first and last
    nz[0] = n[0]
    nz[-1] = n[-1]

    # substrate
    z_sub = np.arange(-dsub, 0, step)
    sz_sub = z_sub.size
    nz_sub = n[0] * np.ones(sz_sub, dtype=complex)

    # external medium
    z_air = np.arange(step, dair + step, step)
    sz_air = z_air.size
    nz_air = n[-1]*np.ones(sz_air, dtype=complex)

    z  = np.concatenate((z_sub,   z, z_air+z[-1]))
    nz = np.concatenate((nz_sub, nz, nz_air))


    # TMM applied from left to right, where
    #   Eout = T * Ein
    #   i.e. field on te right of the interface is equal to matrix T times
    #   field on the left (assuming input field comes from the left)
    # Check the appendix for more info.

    # first of all extract the reflected field at the first interface
    _, r, _, _ = reflectivity(lambda_, theta, d, n, pol)

    # determine wave direction in each layer
    K=2*pi/lambda_
    theta = theta/180*pi
    beta = n[0]*sin(theta)
    costheta = np.sqrt(n**2 - beta**2) / n
    # costheta(real(costheta) < 1e-15) = -costheta(real(costheta) < 1e-15)

    # and now propagate the fields
    E = np.zeros((2,sz), dtype=complex)
    E_air = np.zeros((2,sz_air), dtype=complex)
    E_sub = np.zeros((2,sz_sub), dtype=complex)

    j = 0
    zv = np.cumsum(d)
    za = zv[j]
    zb = zv[j+1]

    E[:,0] = [1, r]
    # if abs(np.imag(costheta)) > 1e-15:
    #     E[:,0] = [0, r]

    Mt = Tij(n[0], n[1], beta, pol)
    for i in range(1,sz):
        kz = K*n[j+1]*costheta[j+1]
        Pt = np.array([[exp(+1j*kz*(z[i]-za)), 0],
                       [0 , exp(-1j*kz*(z[i]-za)) ]], dtype=complex)
        if z[i] == zb:
            j=j+1
            T = Tij(n[j], n[j+1], beta, pol)
            Mt = np.matmul(T, np.matmul(Pt,Mt))
            za = zv[j]
            zb = zv[j+1]
            E[:,i]  = np.matmul(Mt, np.array([1,r], dtype=complex))
        else:
            E[:,i]  = np.matmul(Pt,np.matmul(Mt, np.array([1,r], dtype=complex)))

    E_air[0,:] = E[0,-1]*np.exp(+1j*K*nz[-1]* costheta[-1]*z_air)
    E_sub[0,:] = E[0,0] *np.exp(+1j*K*nz[0] * costheta[0]*z_sub)
    E_sub[1,:] = E[1,0] *np.exp(-1j*K*nz[0] * costheta[0]*z_sub)
    E = np.concatenate((E_sub, E, E_air), axis=1)

    # apply Maxwell equations in order to extract H and E
    costheta_z = np.sqrt(nz**2 - beta**2) / nz
    field = {}
    if pol == 'p':
        # In order to determine Ey and Ez it is necessary to take into
        # account the conventions of signs with which the Fresnel
        # coefficient have been analytically derived.
        # Check the appendix for more info.
        sintheta_z = beta/nz

        field["Ery"] = E[0,:]*costheta_z
        field["Ely"] = E[1,:]*costheta_z
        field["Erz"] =-E[0,:]*sintheta_z
        field["Elz"] = E[1,:]*sintheta_z
        field["Ey"]  = field["Ery"]  + field["Ely"]
        field["Ez"]  = field["Erz"]  + field["Elz"]

        field["Hrx"] = beta*field["Erz"]  - nz*costheta_z*field["Ery"]
        field["Hlx"] =-beta*field["Elz"]  - nz*costheta_z*field["Ely"]
        field["Hx"]  = field["Hrx"]  + field["Hlx"]

        P = abs(field["Hx"] )**2 #/(abs(field["Hrx"][z_sub.size+1])**2)
    else:
        # With s-polarisation the sign convention is much easier. We
        # simply apply the third Maxwell equation
        # Check the appendix for more info.

        field["Erx"] = E[0,:]
        field["Elx"] = E[1,:]
        field["Ex"]  = field["Erx"] + field["Elx"]

        field["Hry"] =  nz * costheta_z * field["Erx"]
        field["Hly"] = -nz * costheta_z * field["Elx"]
        field["Hrz"] = -beta*field["Erx"]
        field["Hlz"] = -beta*field["Elx"]

        field["Hy"]  = field["Hry"]  + field["Hly"]
        field["Hz"]  = field["Hrz"]  + field["Hlz"]

        P = abs(field["Ex"])**2

    return z, nz, P, field


def Tij(n_i, n_j, beta, pol):
# % TMM where E_out = T * E_in
# % r = -T21/T22, t = T11 + T12*r
    costheta_i = sqrt(n_i**2 - beta**2) / n_i
    costheta_j = sqrt(n_j**2 - beta**2) / n_j
    # Fresnel coefficients
    if pol == 's' :
        rij = (n_i*costheta_i - n_j*costheta_j) / (n_i*costheta_i + n_j*costheta_j)
        rji = -rij
        tji =  rji + 1
    elif pol == 'p' :
        rij = (n_j*costheta_i - n_i*costheta_j) / (n_j*costheta_i + n_i*costheta_j)
        rji = -rij
        tji = (rji + 1) * n_j/n_i
    else:
       raise ValueError("Invalid Polarization. Valid options are 's' or 'p'")

    T = 1/tji * np.array([[ 1 , rji ],
                          [rji,  1  ]], dtype=complex)
    return T