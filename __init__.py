#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
% Copyright 2020 Niccolò Marcucci <niccolo.marcucci@polito.it>
%
% Licensed under the Apache License, Version 2.0 (the "License");
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


import meep as mp
import numpy as np
# import h5py as h5
from scipy.io import loadmat, savemat
from math import pi, cos, sin, tan, atan, atan2, sqrt
from matplotlib import pyplot as plt, cm, colors, widgets


from meep_objects.python_utils import *

def anysotropic_material (n, anisotropy1, anisotropy2=0, rot_angle_3=0,
                          rot_angle_1=0, rot_angle_2=0 ):
    """
    Creates and anysotropyc material with two extraordinary axis and one
    ordinary axis (the third one). The axis are assumed to be orthogonal.
    If all rotation angles are equal to 0 first axis would be x, second would
    be y and third z.

    Parameters
    ----------
    n : Real Positive
        Base refractive index.
    anisotropy1 : Real Positive
        Relative variation of the refractive index along the first axis with
        respect to the Base value, expressed as a percentage.
    anisotropy2 : Real Positive, optional
        Same as anisotropy1. The default is 0.
    rot_angle_3 : Real Positive, optional
        Rotation AROUND the third axis. The default is 0.
    rot_angle_1 : Real Positive, optional
        Rotation AROUND the third axis. The default is 0.
    rot_angle_2 : Real Positive, optional
        Rotation AROUND the third axis. The default is 0.

    Returns
    -------
    meep.Medium()
        DESCRIPTION.

    """
    eps = n**2

    eps_e1 = eps*(1+anisotropy1/100)
    eps_e2 = eps*(1+anisotropy2/100)
    eps_matrix = np.diag(np.array([eps_e1, eps_e2, eps]))

    # first rotation around z axis
    rot_matrix = np.array([[cos(rot_angle_3), -sin(rot_angle_3) , 0],
                           [sin(rot_angle_3),  cos(rot_angle_3) , 0],
                           [0                  ,  0             , 1]])
    eps_matrix = np.matmul(np.matmul(rot_matrix,eps_matrix),
                           np.linalg.inv(rot_matrix))

    # second rotation around x axis
    rot_matrix = np.array([[1 , 0                  , 0             ],
                           [0 , cos(rot_angle_1), -sin(rot_angle_1)],
                           [0 , sin(rot_angle_1),  cos(rot_angle_1)]])
    eps_matrix = np.matmul(np.matmul(rot_matrix,eps_matrix),
                           np.linalg.inv(rot_matrix))

    # second rotation around y axis
    rot_matrix = np.array([[ cos(rot_angle_2), 0 , sin(rot_angle_2)],
                           [0                , 1 , 0               ],
                           [-sin(rot_angle_2), 0 , cos(rot_angle_2)]])
    eps_matrix = np.matmul(np.matmul(rot_matrix,eps_matrix),
                           np.linalg.inv(rot_matrix))

    return mp.Medium(epsilon_diag = np.diag(eps_matrix),
                     epsilon_offdiag = np.array([eps_matrix[0,1],
                                                 eps_matrix[0,2],
                                                 eps_matrix[1,2]]))

def circular_DBR_cavity (medium_back=mp.Medium(epsilon=1),
                         medium_groove=mp.Medium(epsilon=2),
                         D=0.4, DBR_period=0.2, FF=0.5, N_rings=10,
                         thickness = 0, orientation = mp.Vector3(0,0,1)):
    """
    Circular DBR cavity created as a sequence of concentric cylinders

    Parameters
    ----------
    medium_back : mp.Medium(),
        Background medium. The default is Vacuum.
    medium_groove : mp.Medium(),
        Medium of the rings. The default is mp.Medium(epsilon=2).
    D : Real Positive
        Diameter of the centre of the cavity. The default is 0.4.
    DBR_period : Real Positive
        Radial period of the DBR. The default is 0.2.
    FF : Real, optional
        Fill factor defined as a number between 0 and 1. The default is 0.5.
    N_rings : Integer
        Number of DBR rings. The default is 10.

    Returns
    -------
    List
        Returns a list of (meep) geometric objects.

    """
    device = []
    rings = []
    for i in range(1,N_rings+1):
        c1 = mp.Cylinder(radius = D/2+(N_rings-i+FF)*DBR_period,
                         height = thickness,
                         axis = orientation,
                         material = medium_groove)
        c2 = mp.Cylinder(radius = D/2+(N_rings-i)*DBR_period,
                         height = thickness,
                         axis = orientation,
                         material = medium_back)
        rings.append(c1)
        rings.append(c2)
    device.extend(rings)
    return device

def dipole_source(f, df=0, source_pos=mp.Vector3(0,0,0),
                  theta=0, phi=0, amplitude = 1):
    """
    Supports only cylindrical coordinates for now

    Parameters
    ----------
    f : Real
        Centre frequency of the source Gaussian spectrum.
    df : Real
        Width of the source Gaussian spectrum. If set to 0 the source will not
        be Gaussian but a continous sinusoidal source.
        The default is 0.
    source_pos : meep.Vector3().
        Source posistion. The default is meep.Vector3(0,0,0).
    theta : Real, optional
        Inclination of the dipole moment with respect to z axis, in radiants.
        The default is 0.
    phi : Real, optional
        Azimutal orientation of the dipole moment, in radiants.
        e.g. theta = pi/2, phi = 0 implies the dipole moment oriented along y.
        The default is 0.
    amplitude: Complex, optional
        Complex amplitude multiplying the current source. Can impart a phase
        shift to the real part of the overall current.
        The default is 1.

    Returns
    -------
    list
        A list containing the three sources (meep objects) required for
        orienting a source arbitrarily in space.

    """

    if df == 0 :
        source_type = mp.ContinuousSource(f,width=0.1 )
    else :
        source_type = mp.GaussianSource(f,df)

    source_x = mp.Source(source_type,
                         component=mp.Ex,
                         amplitude=cos(phi + pi/2)*sin(theta)*amplitude,
                         center=source_pos,)

    source_y = mp.Source(source_type,
                         component=mp.Ey,
                         amplitude=sin(phi + pi/2)*sin(theta)*amplitude,
                         center=source_pos)

    source_z = mp.Source(source_type,
                         component=mp.Ez,
                         amplitude = cos(theta)*amplitude,
                         center=source_pos,)
    return [source_x,source_y, source_z]

def plane_wave_source(f, df, k, center, size, inc_plane_norm, amplitude=1):
    """
    Plane wave source.
    """
    uk = k.unit()
    inc_plane_norm = inc_plane_norm.unit()

    if uk.dot(inc_plane_norm) != 0.:
        # The dot product might never return 0 if inc_plane_norm i not one of
        # the axis due to numerical error.
        raise ValueError("the wavevector has to be orthogonal to the incidence plane")

    uE = uk.cross(inc_plane_norm)
    E = uE * amplitude

    def pw_amp(E, k, x0, component):
        def _pw_amp(x):
            if component == mp.Ex :
                ampl = E.x
            elif component == mp.Ey :
                ampl = E.y
            elif component == mp.Ez :
                ampl = E.z

            return ampl * np.exp( 1j*k.dot(x-x0) )

        return _pw_amp

    plane_wave = [mp.Source(mp.ContinuousSource(f,fwidth=0.1,is_integrated=True) if df==0 else mp.GaussianSource(f,fwidth=df,is_integrated=True),
                            component = mp.Ex,
                            center = center,
                            size = size,
                            amp_func = pw_amp(E, k, center, mp.Ex)),
                  mp.Source(mp.ContinuousSource(f,fwidth=0.1,is_integrated=True) if df==0 else mp.GaussianSource(f,fwidth=df,is_integrated=True),
                            component = mp.Ey,
                            center = center,
                            size = size,
                            amp_func = pw_amp(E, k, center, mp.Ey)),
                  mp.Source(mp.ContinuousSource(f,fwidth=0.1,is_integrated=True) if df==0 else mp.GaussianSource(f,fwidth=df,is_integrated=True),
                            component = mp.Ez,
                            center = center,
                            size = size,
                            amp_func = pw_amp(E, k, center, mp.Ez))]
    return plane_wave

def dielectric_multilayer(design_file, substrate_thickness, x_width,
                          y_width=1, unit = 'um', used_layer_info={}, exclude_last_layer=False,
                          buried=False, axis = mp.Z) :
    """
    Dielectric multilayer stack
    """

    data = loadmat(design_file)

    N = np.size(data['idx_layers'])
    idx_layers = data['idx_layers'].reshape(N)
    d_layers   = data['d_layers'].reshape(N)

    d_layers[ used_layer_info['used_layer'] ] = used_layer_info['thickness']*1e-6
    idx_layers[ used_layer_info['used_layer'] ] = used_layer_info['refractive index']

    if unit == 'nm' :
        d_layers *= 1e9
    elif unit == 'um' :
        d_layers *= 1e6
    else :
        raise ValueError("Unit not supported")

    d_layers[ 0] = substrate_thickness
    d_layers[-1] = 0

    if buried:
        z_shift = d_layers[-2]/2 + d_layers[-3]
    else:
        z_shift = d_layers[-2]/2

        # d_layers[-2] = 0

    multilayer = []

    for i, d in enumerate(d_layers) :
        if d == 0 :
            continue

        z = -d_layers[-2]/2 - my_sum(d_layers,i+1,-2) - d/2 + z_shift

        if not (i == N-2 and exclude_last_layer) :
            if axis == mp.Z :
                size = [x_width, y_width, d]
                centre = [0, 0, z]
            elif axis == mp.Y:
                size = [x_width, d, y_width]
                centre = [0, z, 0]
            elif axis == mp.Y:
                size = [d, y_width, x_width]
                centre = [z, 0, 0]

            multilayer.append(mp.Block(material = mp.Medium(index = np.real(idx_layers[i])),
                                       size     = mp.Vector3(*size),
                                       center   = mp.Vector3(*centre)))
            multilayer[i].name  = f'Layer_{i}'
            multilayer[i].group = 'Multilayer'

    thickness = np.sum(d_layers[1:])

    design_specs = {"d_layers"  :d_layers,
                    "idx_layers":idx_layers }
    return multilayer, thickness, design_specs

def my_sum(array, i, j):
    # equivalent for sum(array[i:j]), valid also for j<i
    # j is excluded from the sum, as per stadard python
    i, j = [array.size + k if k < 0 else k for k in [i, j] ]
    if i <= j:
        return +np.sum(array[i:j])
    else:
        return -np.sum(array[j:i])

def elliptic_DBR_cavity(medium_groove=mp.Medium(epsilon=2),
                        D=0.4, d=0.3, DBR_period=0.2, FF=0.5, N_rings=10,
                        thickness=0,
                        center=mp.Vector3(), axial_rotation=0):
    """
    Elliptic DBR cavity created as a sequence of concentric cylinders.

    Parameters
    ----------
    medium_groove : mp.Medium(),
        Medium of the rings. The default is mp.Medium(epsilon=2).
    D : Real Positive
        Major axis diametre of the centre of the cavity. The default is 0.4.
    d : Real Positive
        Miinor axis diameter of the centre of the cavity. The default is 0.3.
    DBR_period : Real Positive
        Radial period of the DBR. The default is 0.2.
    FF : Real, optional
        Fill factor defined as a number between 0 and 1. The default is 0.5.
    N_rings : Integer
        Number of DBR rings. The default is 10.
    axial_rotation : radians
        In plane rotation of the ellipse.

    Returns
    -------
    List
        Returns a list of (meep) geometric objects.

    """
    rings = []
    polygons = grating_veritices(DBR_period, D/2, d/2, N_rings, n_arms = 0,  FF = FF)
    for ring in polygons:
        vertices = [mp.Vector3(v[0]*cos(axial_rotation)-v[1]*sin(axial_rotation),
                               v[0]*sin(axial_rotation)+v[1]*cos(axial_rotation),
                               0)   for v in np.transpose(ring)]
        centroid =  sum(vertices, mp.Vector3(0)) * (1.0 / len(vertices))

        c1 = mp.Prism(vertices = vertices,
                      height = thickness,
                      axis = mp.Vector3(z=1),
                      center = center + centroid,
                      material = medium_groove)
        rings.append(c1)
    return rings


def linear_DBR_cavity(medium_groove=mp.Medium(epsilon=2),
                      D=0.4, DBR_period=0.2, FF=0.5, N_periods=10,
                      width=1, thickness=0, axis=mp.Z,
                      axial_rotation=0, center=mp.Vector3()):
    """
    Linear DBR cavity created as a sequence of rectangles.

    Parameters
    ----------
    medium_groove : mp.Medium(),
        Medium of the rings. The default is mp.Medium(epsilon=2).
    D : Real Positive
        spacer at the centre of the cavity. The default is 0.4.
    DBR_period : Real Positive
        period of the DBR. The default is 0.2.
    FF : Real, optional
        Fill factor defined as a number between 0 and 1. The default is 0.5.
    N_periods : Integer
        Number of DBR lines on each side of the spacer. The default is 10.
    width : Real, optional
        Length of each groove. Default is 10um
    axial_rotation : radians
        In plane rotation of the DBR.

    Returns
    -------
    List
        Returns a list of (meep) geometric objects.

    """
    device = []
    for i in range(1,N_periods+1):
        groove_size = mp.Vector3(FF*DBR_period, width, thickness)
        groove_centre = mp.Vector3(D/2 + FF*DBR_period/2 + (N_periods-i)*DBR_period, 0, 0)


        phi = axial_rotation
        e1 = mp.Vector3(cos(phi), sin(phi), 0)
        e2 = mp.Vector3(cos(phi+pi/2), sin(phi+pi/2), 0)
        e3 = mp.Vector3(z=1)
        groove_centre_right = groove_centre.rotate(mp.Vector3(z=1), phi)
        groove_centre_left  = groove_centre.rotate(mp.Vector3(z=1), phi + pi)

        if axis == mp.Y :
            e1 = e1.rotate(mp.Vector3(x=1), -pi/2)
            e2 = e2.rotate(mp.Vector3(x=1), -pi/2)
            e3 = e3.rotate(mp.Vector3(x=1), -pi/2)
            groove_centre_right = groove_centre_right.rotate(mp.Vector3(x=1), -pi/2) + center
            groove_centre_left = groove_centre_left.rotate(mp.Vector3(x=1), -pi/2) + center
        elif axis == mp.X :
            e1 = e1.rotate(mp.Vector3(y=1), -pi/2)
            e2 = e2.rotate(mp.Vector3(y=1), -pi/2)
            e3 = e3.rotate(mp.Vector3(y=1), -pi/2)
            groove_centre_right = groove_centre_right.rotate(mp.Vector3(y=1), -pi/2) + center
            groove_centre_left = groove_centre_left.rotate(mp.Vector3(y=1), -pi/2) + center
        elif axis == mp.Z :
            groove_centre_right += center
            groove_centre_left += center
        else:
            raise ValueError("Wrong axis value.")

        b1 = mp.Block(e1 = e1,
                      e2 = e2,
                      e3 = e3,
                      size = groove_size,
                      center = groove_centre_right,
                      material = medium_groove)
        b2 = mp.Block(e1 = e1,
                      e2 = e2,
                      e3 = e3,
                      size = groove_size,
                      center = groove_centre_left,
                      material = medium_groove)

        device.append(b1)
        device.append(b2)
    return device

def metasurface_radial_grating(medium_groove=mp.Medium(epsilon=2),
                               D=2, metasurface_period=0.2, scatter_length = 0.4,
                               scatter_width=0.5, scatter_tilt=pi/4,
                               scatter_type = 'radial', N_rings=10, N_arms=0,
                               thickness=0, orientation=mp.Vector3(0, 0, 1)):
    """
    Complex metasurface-like grating.

    Parameters
    ----------
    medium_groove : mp.Medium(),
        Medium of the rings. The default is mp.Medium(epsilon=2).
    D : Real Positive
        Diameter of the centre of the metasurface. The default is 2.
    metasurface_period : Real Positive
        Radial period of the DBR. The default is 0.2.
    scatter_width : TYPE, optional
        DESCRIPTION. The default is 0.5.
    scatter_tilt : TYPE, optional
        DESCRIPTION. The default is pi/4.
    N_rings : Integer
        Number of DBR rings. The default is 10.
    N_arms : TYPE, optional
        DESCRIPTION. The default is 0.
    thickness : TYPE, optional
        DESCRIPTION. The default is 0.
    orientation : TYPE, optional
        DESCRIPTION. The default is mp.Vector3(0, 0, 1).

    Raises
    ------
    ValueError
        DESCRIPTION.

    Returns
    -------
    None.

    """
    metasurface = []
    start_radius = D / 2
    if N_arms != 0:
        print('To be Implemented!')
        # for j in range(np.abs(N_arms)):
        #     b1 = mp.Block(e1=mp.Vector3(1,0,0).rotate(mp.Vector3(0,0,1),rotation),
        #      e2=mp.Vector3(0,1,0).rotate(mp.Vector3(0,0,1),rotation),
        #      size=groove_size,
        #      center=groove_centre.rotate(mp.Vector3(0,0,1),rotation),
        #      material = medium_groove)

    else:
        if N_rings != 0:
            for n in range(N_rings):
                radius = D/2 + n * metasurface_period

                if scatter_type == "filled":
                    N_scatters = round(2*pi * radius / scatter_length)
                elif scatter_type == "radial":
                    N_scatters = round(2*pi * start_radius / scatter_length)
                elif scatter_type == "radial_filled":
                    N_scatters = round(2*pi * start_radius / scatter_length)
                else:
                    raise ValueError()

                for k in range(N_scatters):
                    location_tilt = 2*pi / N_scatters * k

                    tilt = (pi/2 + scatter_tilt + location_tilt + n*pi/6)

                    metasurface.append(mp.Block(
                            e1=mp.Vector3(cos(tilt), sin(tilt), 0),
                            e2=mp.Vector3(cos(tilt+pi/2), sin(tilt+pi/2), 0),
                            size=mp.Vector3(scatter_length*0.75, scatter_width, thickness),
                            center=mp.Vector3(
                                radius * cos(location_tilt),
                                radius * sin(location_tilt),
                                0),
                            material=medium_groove))

                    metasurface[-1].name  = f'Scatter_{n}_{k}'
                    metasurface[-1].group = 'Metasurface'

    return metasurface

def pol_splitting_grating(  medium_groove=mp.Medium(epsilon=2),
                            D=2, metasurface_period=0.4, scatter_length = 0.4,
                            scatter_width=0.1, scatter_tilt=pi/3,
                            scatter_shape = '', scatter_disposition='radial',
                            topology='circular',
                            n_rings=9, n_arms=0, lambda_bsw=0.5,
                            thickness=1, center=mp.Vector3(0, 0, 0)):
    """
    Circular polarization sensitive grating. Similar tu the metasurface-like
    grating, but with specific purpose.
    """
    sc_width = scatter_width
    sc_length = scatter_length

    # following lists are for v-shaped scatters
    rel_width = sc_width / sc_length
    scatter_vertices_x = [0, .5, .5-rel_width/0.866, 0, -.5+rel_width*0.866, -.5]
    scatter_vertices_y = [-.5, .5, .5, -.5+rel_width/0.5, .5, .5]
    scatter_vertices = np.array([scatter_vertices_x, scatter_vertices_y]).transpose()

    metasurface = []

    if n_rings != 0:
        for n in range(n_rings):
            r0 = D/2 + n * metasurface_period

            # the following if statement is just for setting the actual
            # beginning of the scatter at D/2
            if scatter_shape == 'V' or scatter_shape == 'v':
                r0 += metasurface_period/2
            else:
                r0 += sc_width/2

            if scatter_disposition == 'filled':
                L = 2*pi*r0 + pi*metasurface_period*n_arms      # length of on turn in archimede's spiral
                N_scatters = round(L / metasurface_period)
            if scatter_disposition == 'radial':
                r1 = r0 - n * metasurface_period
                L = 2*pi*r1 + pi*metasurface_period*n_arms
                N_scatters = round(L / metasurface_period)


            if n_arms != 0:
                N_scatters -= np.mod(N_scatters,n_arms)     # make N_scatters divisible by n_arms
                theta = np.linspace(0, 2*pi/n_arms, int(N_scatters/n_arms))
                theta = np.tile(theta, n_arms)
            else:
                theta = np.zeros((N_scatters,1))

            for k in range(N_scatters):
                location_tilt = 2*pi / N_scatters * k

                if topology == 'spiral':
                    radius = r0 + (lambda_bsw * n_arms * theta[k]/2/pi)
                    tilt = pi/2 + location_tilt + n*scatter_tilt
                elif topology == 'circular':
                    radius = r0
                    tilt = pi/2 + location_tilt + n*scatter_tilt + theta[k]
                else:
                    raise ValueError('Topology can be either "spiral" or "circular"')

                if scatter_shape == 'V' or scatter_shape == 'v' :
                    tilt = tilt - pi
                    vertices = [mp.Vector3(v[0],v[1],0).rotate(mp.Z,tilt)*sc_length for v in scatter_vertices]
                    centroid =  sum(vertices, mp.Vector3(0)) * (1.0 / len(vertices))

                    scatter =  mp.Prism(vertices = vertices,
                                        height = thickness,
                                        center=mp.Vector3(radius * cos(location_tilt),
                                                          radius * sin(location_tilt),
                                                          0) + center + centroid,
                                        axis = mp.Vector3(z=1),
                                        material = medium_groove)
                else:
                    scatter = mp.Block(e1=mp.Vector3(cos(tilt), sin(tilt), 0),
                                       e2=mp.Vector3(cos(tilt+pi/2), sin(tilt+pi/2), 0),
                                       size=mp.Vector3(scatter_length*0.75, scatter_width, thickness),
                                       center=mp.Vector3(radius * cos(location_tilt),
                                                         radius * sin(location_tilt),
                                                         0) + center,
                                       material=medium_groove)

                metasurface.append(scatter)
                metasurface[-1].name  = f'Scatter_{n}_{k}'
                metasurface[-1].group = 'Metasurface'

    return metasurface


def spiral_grating(medium_groove=mp.Medium(epsilon=2),
                   D=0.4, d=None, DBR_period=0.2, FF=0.5, N_rings=10,
                   N_arms=2, thickness=0, center=mp.Vector3(0, 0, 0)):
    """
    Elliptic DBR cavity created as a sequence of concentric cylinders.

    Parameters
    ----------
    medium_groove : mp.Medium(),
        Medium of the rings. The default is mp.Medium(epsilon=2).
    D : Real Positive
        Major axis diametre of the centre of the cavity.
        The default is 0.4.
    d : Real Positive
        Minor axis diameter of the centre of the cavity.
        The default is 0.3.
    DBR_period : Real Positive
        Radial period of the DBR.
        The default is 0.2.
    FF : Real, optional
        Fill factor defined as a number between 0 and 1.
        The default is 0.5.
    N_rings : Integer
        Number of DBR rings.
        The default is 10.
    N_arms : Integer
        Number of arms of the spiaral. a.k.a. topological charge.
        Default is 2
    orientation : meeep Vector3()
        Orientation of the prism. Useful for 1D symulations.
        Default is along z-axis

    Returns
    -------
    List
        Returns a list of (meep) geometric objects.

    """
    device = []

    if d == None:
        d = D

    polygons = grating_veritices(DBR_period, D/2, d/2, N_rings, n_arms = N_arms,  FF = FF)


    for polygon in polygons:
        vertices = [mp.Vector3(v[0],v[1],0) for v in np.transpose(polygon)]
        centroid =  sum(vertices, mp.Vector3(0)) * (1.0 / len(vertices))

        c1 = mp.Prism(vertices = vertices,
                      height = thickness,
                      axis = mp.Vector3(0,0,1),
                      center = center + centroid,
                      material = medium_groove)
        device.append(c1)

    return device

def grating_veritices(period, start_radius1,
                      start_radius2=0, N_periods=10, n_arms=0,  FF=0.5,
                      spacer='empty') :
    """
    Function for generating the list of vertices por the circular, spiral
    and elliptic gratings.
    """

    if start_radius2 == 0:
        start_radius2 = start_radius1
    a = start_radius1
    b = start_radius2

    vert_list = []

    if n_arms != 0 :
        half_res = np.sum( [ (max([a,b]) + period*i) * 2*pi for i in range(N_periods)]) / period
        half_res = max(int(half_res), 32*int(N_periods/n_arms))
        half_res -= np.mod(half_res, int(N_periods/n_arms)) + 1

        res = 2 * half_res

        theta= np.linspace(0, 2*pi*N_periods/n_arms, half_res)

        # Each arm of the spiral is defined by a polygon
        for j in range(np.abs(n_arms)) :

            # from where to start the spiral, might even be elliptic
            if a == b == 0 :
                start_radius = np.zeros((half_res,))
            else:
                start_radius = a * b / np.sqrt(
                                          (b * np.cos(theta + 2*pi*j/n_arms) )**2 +
                                          (a * np.sin(theta + 2*pi*j/n_arms) )**2 )
            # parametrize the radius
            radius = start_radius+period*theta/(2*pi/n_arms)

            vertices = np.zeros((2, res))
            vertices[0, 0:half_res] = radius*np.cos(theta+2*pi*j/n_arms)
            vertices[1, 0:half_res] = radius*np.sin(theta+2*pi*j/n_arms)
            vertices[0, half_res:res] = np.flip( (radius+period*FF) * np.cos(theta+2*pi*j/n_arms) )
            vertices[1, half_res:res] = np.flip( (radius+period*FF) * np.sin(theta+2*pi*j/n_arms) )

            vert_list.append(vertices)

            # close the spiral on the origin if the initial radius is null
            if min(a,b) == 0 :
                centre_res = 15

                r_centre = np.linspace(period*FF, 0, centre_res)

                theta_centre = np.linspace(0, -pi/n_arms, centre_res)

                vert = np.zeros( (2,centre_res) )
                vert[0, :] = r_centre * np.cos(theta_centre + 2*pi*j/n_arms)
                vert[1, :] = r_centre * np.sin(theta_centre + 2*pi*j/n_arms)

                vert_list.append(vert)

    else :
        if spacer == 'full':
            # make a circle at the centre
            extra_r = period * (1 - FF)

            theta = np.linspace(0, 2*pi*(1-1/32), 32)
            radius = a * b / np.sqrt( (b*np.cos(theta))**2 +
                                      (a*np.sin(theta))**2 )

            vertices = np.zeros((2,32))
            vertices[0,:] = radius * np.cos(theta)
            vertices[1,:] = radius * np.sin(theta)

            vert_list.append(vertices)
        elif spacer == 'empty':
            extra_r = 0
            print('spacer is empty')
        else:
            raise ValueError("Invalid spacer value. Either 'empty' or 'full'")

        for j in range(N_periods):
            half_res = int( (max([a,b]) + period*j) * 2*pi / period)
            half_res = max(int(half_res), 32)
            half_res -= np.mod(half_res,4) +1

            res = 2 * half_res

            theta = np.linspace(0, 2*pi, half_res)

            # first circle radius, can be elliptic
            start_radius = a * b / np.sqrt( (b*np.cos(theta))**2 +
                                            (a*np.sin(theta))**2 )
            radius = start_radius + period*j + extra_r
            vertices = np.zeros((2,res))
            vertices[0,0:half_res] = radius * np.cos(theta)
            vertices[1,0:half_res] = radius * np.sin(theta)
            vertices[0,half_res:res] = np.flip( (radius+period*FF) * np.cos(theta))
            vertices[1,half_res:res] = np.flip( (radius+period*FF) * np.sin(theta))
            vertices[1,half_res] -= .001   # this is necessary for solving a bug
                                           # see https://github.com/NanoComp/libctl/issues/61
            vert_list.append(vertices)

    return vert_list

def create_openscad(sim, scale_factor=1):
    try:
        import openpyscad as ops
    except ModuleNotFoundError:
        print("WARNING openpyscad is not installed in this environment")
    else:
        try:
            scad_name = f"{sim.name}.scad"
        except:
            print("WARNING Simulation doesn't have a name")
            scad_name = "meep_sim.scad"

        with open(scad_name,"w") as file:
            print("written")
            file.write(f"//Simulation {scad_name}\n")

        for obj in sim.geometry:
            if obj.__class__ == mp.Block:
                cube = ops.Cube([obj.size.x*scale_factor,
                                 obj.size.y*scale_factor,
                                 obj.size.z*scale_factor], center = True)
                tilt = np.arctan2(obj.e3.y, obj.e3.x)
                print(obj.e1)
                if tilt != 0:
                    cube = cube.rotate([0, 0, tilt/np.pi*180])
                tilt = np.arctan2(obj.e1.z, obj.e1.x)
                if tilt != 0:
                    cube = cube.rotate([0, -tilt/np.pi*180, 0])
                tilt = np.arctan2(obj.e1.z, obj.e1.y)
                if tilt != 0:
                    cube = cube.rotate([tilt/np.pi*180, 0, 0])

                index = np.round(np.sqrt(obj.material.epsilon_diag.x),2)
                if index == 2.53:
                    color = [0, 0, 1]
                elif index == 1.65:
                    color = [0, 1, 0]
                elif index == 1.46:
                    color = [1, 0, 0]
                elif index == 1.48:
                    color = [1, 1, 0]
                elif index == 2.08:
                    color = [0, 1, 1]
                else:
                    color = [1, 0, 0]
                cube = cube.color(color)

                scad = cube

            elif obj.__class__ == mp.Cylinder:
                cyl = ops.Cylinder(h=obj.height*scale_factor,
                                    r=obj.radius*scale_factor, center = True)

                scad = cyl

            elif obj.__class__ == mp.Prism:
                print("to be defined")
                #prism = ops.Polyhedron(points)

            scad = scad.translate([obj.center.x*scale_factor,
                                   obj.center.y*scale_factor,
                                   obj.center.z*scale_factor])
            scad.write(scad_name, mode='a')

        # sim_domain = ops.Cube([(sim.cell_size.x - sim.PML_width) * scale_factor,
        #                         (sim.cell_size.y - sim.PML_width) * scale_factor,
        #                         (sim.cell_size.z - sim.PML_width) * scale_factor], center = True)
        # sim_domain = sim_domain.color([.5, .5, .5, .5])
        # sim_domain = sim_domain.translate([sim.geometry_center.x*scale_factor,
        #                                     sim.geometry_center.y*scale_factor,
        #                                     sim.geometry_center.z*scale_factor])