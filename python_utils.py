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

import numpy as np
from scipy.io import loadmat, savemat
from math import pi, cos, sin, tan, atan, atan2, sqrt
from matplotlib import pyplot as plt, cm, colors, widgets

def convert_seconds (elapsed):
    minutes = np.floor(elapsed/60)
    secs = elapsed-minutes*60
    secs = np.round(secs*100)/100

    hours = np.int_(np.floor(minutes/60))
    minutes = np.int_(minutes-hours*60)

    return f'{hours}h-{minutes}min-{secs}s'

def plot_data_section(data, x=None, y=None, z=None):

    # Create the figure and the line that we will manipulate
    fig = plt.figure()
    ax = plt.axes()
    plt.imshow(data[:,:,0], vmax=9.0, vmin=1.0)


    # adjust the main plot to make room for the sliders
    plt.subplots_adjust(left=0.2)
    axcolor = 'lightgoldenrodyellow'

    # Make a vertically oriented slider to control the amplitude
    z = plt.axes([0.1, 0.25, 0.0225, 0.63], facecolor=axcolor)
    z_slider = widgets.Slider(
        ax=z,
        label="Z",
        valmin=0,
        valmax=np.shape(data)[2],
        valinit=0,
        valstep=np.arange(0, np.shape(data)[2], dtype=int),
        orientation="vertical"
    )

    # Create a `matplotlib.widgets.Button` to reset the sliders to initial values.
    resetax = plt.axes([0.025, 0.025, 0.1, 0.04])
    # button = widgets.Button(resetax, 'Reset', color=axcolor, hovercolor='0.975')

    layer = widgets.TextBox(resetax,'','0', color=axcolor, hovercolor='0.975')

    # The function to be called anytime a slider's value changes
    def update(val):
        ax.imshow(data[:, :, z_slider.val], vmax=3.0, vmin=1.0)
        fig.canvas.draw_idle()
        layer.set_val(f'{val}')

    # register the update function with each slider
    z_slider.on_changed(update)

    def set_slider(event):
        z_slider.val = int(layer.text)
    # button.on_clicked(reset)
    layer.on_submit(set_slider)

    plt.show()

def plot_image(x, y, image,  n_grid = 11, **kwargs):
    fig, ax = plt.subplots(1,1)

    plt.imshow(np.transpose(image), origin='lower', **kwargs)

    x_label_list = np.round(np.linspace(x.min(), x.max(), n_grid), 2)
    y_label_list = np.round(np.linspace(y.min(), y.max(), n_grid), 2)

    x_positions = np.linspace(0, x.size, n_grid)
    y_positions = np.linspace(0, y.size, n_grid)

    plt.xticks(x_positions, x_label_list)
    plt.yticks(y_positions, y_label_list)

    return fig