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
    d_start = d[1]
    d_end = d[-1]
    d[1] = 0
    d[-1] = 0
    
    # remove dummy layers at beginning/end, and set remaining first and
    # last layer with zero thickness
    i = 2
    while n[i] == n[1] or (d[i] == 0 or d[i] == 0.):# exit when encounter
        d[i] = 0                                    # the first layer with
        i = -1                                      # n~=n(1) that has
                                                    # thickness
    while (n[i]==n[-1]) or (d[i]==0):
        d[i] = 0
        i = i-1
    
    d_new = []
    n_new = []
    for i,l in enumerate(d):
        if l != 0 and l != 0.:
            d_new.append(l)
            n_new.append(n[i])

    n = n_new
    d = d_new
            
    return d, n, d_start, d_end

def reflectivity (Lambda,theta_in,d,n,pol) : # [R,r,t,Tr] = 
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
    % first interface.
    """
    N_layers = np.size(d)
    theta_in = theta_in/180*pi
    
    K = 2*pi/Lambda
    
    if np.size(theta_in) > 1 or np.size(Lambda) > 1:
        raise ValueError("Lambda or theta should be scalar")
    
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
        rtij = rji/tji;
        T11t = Pr/tji*T11 + rtij*Pl*T21;
        T12t = Pr/tji*T12 + rtij*Pl*T22;
        T21t = rtij*Pr*T11 + Pl/tji*T21;
        T22t = rtij*Pr*T12 + Pl/tji*T22;

        T11 = T11t;
        T12 = T12t;
        T21 = T21t;
        T22 = T22t;
     
    r = -T21/T22;
    t = T11+r*T12;
    # Tr(i) = abs( t(i)*n(end)/n(1)*real(costheta_z(end))...
    #                 /costheta_z(1) )^2;

    R = np.abs(r)**2
    
    return R, r, t
