#!/usr/bin/env python3
import sys
import os
os.environ['NUMBA_THREADING_LAYER'] = 'workqueue'
import numpy as np
from matplotlib import pyplot
from PyMPDATA import ScalarField, Solver, Stepper, VectorField, Options, boundary_conditions
from open_atmos_jupyter_utils import show_plot, show_anim

def plot_preview():
    pyplot.figure(figsize=(14.5,3))
    pyplot.axis('off')

    x = np.linspace(-np.pi, np.pi)
    z = np.cos(x+2)/2+.25
    b = np.sin(x)/3-2

    arrow_kwargs = {'length_includes_head': True, 'head_width': .05}

    ix = 5; color='brown'; label='$b$'
    pyplot.fill_between(x, b, -2.75, color=color, label=f'$z=-${label}', alpha=.5)
    pyplot.arrow(x[ix], 0, 0, b[ix], color=color, **arrow_kwargs)
    pyplot.annotate(f" {label}", xy=(x[ix], b[ix] + 1), color=color)

    ix = 10; color='blue'; label=r'$\zeta$'
    pyplot.fill_between(x, z, b, color=color, label=f'$z=${label}', alpha=.1)
    pyplot.arrow(x[ix], 0, 0, z[ix], color=color, **arrow_kwargs)
    pyplot.annotate(f" {label}", xy=(x[ix], z[ix] - .5), color=color)

    ix = 15; color='black'; label='$h$'
    pyplot.plot(x, np.full_like(x, 0), linestyle='--', color=color, label='$z=0$')
    pyplot.arrow(x[ix], 0, 0, z[ix], color=color, **arrow_kwargs)
    pyplot.arrow(x[ix], 0, 0, b[ix], color=color, **arrow_kwargs)
    pyplot.annotate(f" {label}", xy=(x[ix], -1), color=color)

    pyplot.legend(loc=10)
    show_plot('symbols')


class ShallowWaterEquationsIntegratorWithBathymetry:
    def __init__(self, *, h_initial: np.ndarray, bathymetry: np.ndarray, options: Options = None):
        options = options or Options(nonoscillatory=True, infinite_gauge=True)
        self.bathymetry = bathymetry
        X, Y, grid = 0, 1, h_initial.shape
        stepper = Stepper(options=options, grid=grid)
        kwargs = {
            'boundary_conditions': [boundary_conditions.Periodic()] * len(grid),
            'halo': options.n_halo,
        }
        advectees = {
            "h": ScalarField(h_initial, **kwargs),
            "uh": ScalarField(np.zeros(grid), **kwargs),
            "vh": ScalarField(np.zeros(grid), **kwargs),
        }
        self.advector = VectorField((
                np.zeros((grid[X] + 1, grid[Y])),
                np.zeros((grid[X], grid[Y] + 1))
            ), **kwargs
        )
        self.solvers = { k: Solver(stepper, v, self.advector) for k, v in advectees.items() }

    def __getitem__(self, key):
        return self.solvers[key].advectee.get()

    def _apply_half_rhs(self, *, key, axis, g_times_dt_over_dxy):
        self[key][:] -= g_times_dt_over_dxy*.5*self['h'] * np.gradient(self['h'] - self.bathymetry, axis=axis)

    def _update_courant_numbers(self, *, axis, key, mask, dt_over_dxy):
        velocity = np.where(mask, np.nan, 0)
        momentum = self[key]
        np.divide(momentum, self['h'], where=mask, out=velocity)

        all = slice(None, None)
        all_but_last = slice(None, -1)
        all_but_first_and_last = slice(1, -1)

        velocity_at_cell_boundaries = velocity[(
            (all_but_last, all),
            (all, all_but_last),
        )[axis]] + np.diff(velocity, axis=axis) / 2
        courant_number = self.advector.get_component(axis)[(
            (all_but_first_and_last, all),
            (all, all_but_first_and_last)
        )[axis]]
        courant_number[:] = velocity_at_cell_boundaries * dt_over_dxy[axis]
        assert np.amax(np.abs(courant_number)) <= 1

    def __call__(self, *, nt: int, g: float, dt_over_dxy: tuple, outfreq: int, eps: float=1e-7):
        output = {k: [] for k in self.solvers.keys()}
        for it in range(nt + 1):
            if it != 0:
                mask = self['h'] > eps
                for axis, key in enumerate(("uh", "vh")):
                    self._update_courant_numbers(axis=axis, key=key, mask=mask, dt_over_dxy=dt_over_dxy)
                self.solvers["h"].advance(n_steps=1)
                for axis, key in enumerate(("uh", "vh")):
                    self._apply_half_rhs(key=key, axis=axis, g_times_dt_over_dxy=g * dt_over_dxy[axis])
                    self.solvers[key].advance(n_steps=1)
                    self._apply_half_rhs(key=key, axis=axis, g_times_dt_over_dxy=g * dt_over_dxy[axis])
            if it % outfreq == 0:
                for key in self.solvers.keys():
                    output[key].append(self[key].copy())
        return output


bathymetry = np.ones((100,100))
xi, yi = np.indices(bathymetry.shape)

radius = 20
radius2 = 30
#bathymetry -= 1*xi/bathymetry.shape[1]
rows, cols = bathymetry.shape

y, x = np.ogrid[:rows, :cols]

x0 = cols / 2
y0 = rows / 2

distance = np.sqrt((x - x0)**2 + (y - y0)**2)

change = np.clip((radius2 - distance) / (radius2 - radius), 0, 1)

bathymetry -= change*.9

h_initial = bathymetry.copy() - np.where(bathymetry != 0, .05, 0)
h_initial[0:4, 0:4] += .025

def simulate():
    from scipy.constants import g

    return ShallowWaterEquationsIntegratorWithBathymetry(
        h_initial=h_initial, bathymetry=bathymetry
    )(
        g=g,
        nt=250,
        dt_over_dxy=(.25, .25),
        outfreq=3,
    )


def plot(i, *, zlim=(-.1, .1)):
    """ plots psi as a wireframe with contoured and coloured bathymetry below """
    xi, yi = np.indices(output['h'][i].shape)
    fig, ax = pyplot.subplots(subplot_kw={"projection": "3d"})
    pyplot.gca().plot_wireframe(
        xi+.5, yi+.5,
        output['h'][i] - bathymetry,
#        output['vh'][i],
        color='blue', linewidth=.5
    )
    ax.set_zlim(zlim)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.fill = False
        axis.pane.set_edgecolor('black')
        axis.pane.set_alpha(1)
    ax.grid(False)
    ax.set_zticks([])
    for axis in ('x', 'y'):
        getattr(ax, f'set_{axis}label')(f"{axis}/d{axis}")
    ax.set_proj_type('ortho')
    cnt = pyplot.contourf(xi+.5, yi + .5, bathymetry, zdir='z', offset=zlim[0])
    pyplot.colorbar(cnt, pad=.1, aspect=10, fraction=.04, label='batymetria')
    return fig

