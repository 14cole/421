from __future__ import annotations

import numpy as np
from matplotlib.colors import Normalize

C0 = 299_792_458.0
MAX_AZ_SAMPLES = 96
MAX_EL_SAMPLES = 64
MAX_FREQ_SAMPLES = 128
MIN_GRID_SIZE = 32
MAX_GRID_SIZE = 96
MAX_RENDER_POINTS = 30_000


def _thin_indices(indices: list[int], max_count: int) -> tuple[list[int], bool]:
    if len(indices) <= max_count:
        return indices, False
    picks = np.linspace(0, len(indices) - 1, max_count, dtype=int)
    picks = np.unique(picks)
    return [indices[i] for i in picks], True


def _unit_to_hz_scale(unit: str) -> float:
    unit = unit.strip().lower()
    if unit == "hz":
        return 1.0
    if unit == "khz":
        return 1e3
    if unit == "mhz":
        return 1e6
    if unit == "ghz":
        return 1e9
    return 1e9


def _bounded_grid_size(value: int) -> int:
    value = max(MIN_GRID_SIZE, int(value))
    pow2 = 1 << int(np.ceil(np.log2(value)))
    return int(min(MAX_GRID_SIZE, max(MIN_GRID_SIZE, pow2)))


def _colorbar_label(linear_scale: bool) -> str:
    if linear_scale:
        return "Reflectivity (Linear)"
    return "Reflectivity (dBsm)"


def _set_3d_axes_style(self) -> None:
    text_color = self._current_plot_text()
    self.plot_ax.set_facecolor(self._current_plot_bg())
    self.plot_ax.tick_params(colors=text_color)
    self.plot_ax.xaxis.label.set_color(text_color)
    self.plot_ax.yaxis.label.set_color(text_color)
    self.plot_ax.zaxis.label.set_color(text_color)
    self.plot_ax.grid(self._plot_grid_enabled(), color=self._current_plot_grid(), alpha=0.35)
    for axis in (self.plot_ax.xaxis, self.plot_ax.yaxis, self.plot_ax.zaxis):
        try:
            axis.pane.set_facecolor(self._current_plot_bg())
            axis.pane.set_edgecolor(self.palette["border"])
        except Exception:
            pass


def render(self) -> None:
    self.last_plot_mode = "isar_3d"
    if self.active_dataset is None:
        self.status.showMessage("Select a dataset before plotting.")
        return

    az_indices = sorted(self._selected_indices(self.list_az))
    elev_indices = sorted(self._selected_indices(self.list_elev))
    freq_indices = sorted(self._selected_indices(self.list_freq))
    if len(az_indices) < 2:
        self.status.showMessage("Select at least 2 azimuth samples for 3D ISAR.")
        return
    if len(elev_indices) < 2:
        self.status.showMessage("Select at least 2 elevation samples for 3D ISAR.")
        return
    if len(freq_indices) < 2:
        self.status.showMessage("Select at least 2 frequency samples for 3D ISAR.")
        return

    pol_idx = self._single_selection_index(self.list_pol, "polarization")
    if pol_idx is None:
        return

    auto_thin = True
    if hasattr(self, "chk_isar3d_auto_thin"):
        auto_thin = bool(self.chk_isar3d_auto_thin.isChecked())

    max_az_samples = MAX_AZ_SAMPLES
    if hasattr(self, "spin_isar3d_max_az"):
        max_az_samples = max(2, int(round(float(self.spin_isar3d_max_az.value()))))

    max_el_samples = MAX_EL_SAMPLES
    if hasattr(self, "spin_isar3d_max_el"):
        max_el_samples = max(2, int(round(float(self.spin_isar3d_max_el.value()))))

    max_freq_samples = MAX_FREQ_SAMPLES
    if hasattr(self, "spin_isar3d_max_freq"):
        max_freq_samples = max(2, int(round(float(self.spin_isar3d_max_freq.value()))))

    max_render_points = MAX_RENDER_POINTS
    if hasattr(self, "spin_isar3d_max_voxels"):
        max_render_points = max(1, int(round(float(self.spin_isar3d_max_voxels.value()))))

    voxel_quantile = 0.995
    if hasattr(self, "spin_isar3d_quantile"):
        voxel_quantile = float(self.spin_isar3d_quantile.value())
        voxel_quantile = min(1.0, max(0.0, voxel_quantile))

    point_size = 10.0
    if hasattr(self, "spin_isar3d_point_size"):
        point_size = max(1.0, float(self.spin_isar3d_point_size.value()))

    if auto_thin:
        az_indices, az_thinned = _thin_indices(az_indices, max_az_samples)
        elev_indices, elev_thinned = _thin_indices(elev_indices, max_el_samples)
        freq_indices, freq_thinned = _thin_indices(freq_indices, max_freq_samples)
    else:
        az_thinned = False
        elev_thinned = False
        freq_thinned = False

    az_values = self.active_dataset.azimuths[az_indices].astype(float)
    elev_values = self.active_dataset.elevations[elev_indices].astype(float)
    freq_values = self.active_dataset.frequencies[freq_indices].astype(float)

    az_order = np.argsort(az_values)
    elev_order = np.argsort(elev_values)
    freq_order = np.argsort(freq_values)
    az_indices = [az_indices[i] for i in az_order]
    elev_indices = [elev_indices[i] for i in elev_order]
    freq_indices = [freq_indices[i] for i in freq_order]
    az_values = az_values[az_order]
    elev_values = elev_values[elev_order]
    freq_values = freq_values[freq_order]

    if not np.all(np.isfinite(az_values)) or not np.all(np.isfinite(elev_values)):
        self.status.showMessage("3D ISAR requires finite azimuth/elevation samples.")
        return
    if not np.all(np.isfinite(freq_values)):
        self.status.showMessage("3D ISAR requires finite frequency samples.")
        return
    if np.any(np.diff(freq_values) <= 0.0):
        self.status.showMessage("Frequency samples must be strictly increasing for 3D ISAR.")
        return

    freq_unit = str(self.active_dataset.units.get("frequency", "ghz"))
    freq_scale = _unit_to_hz_scale(freq_unit)
    freq_hz = freq_values * freq_scale

    rcs_slice = self.active_dataset.rcs[np.ix_(az_indices, elev_indices, freq_indices, [pol_idx])]
    rcs_slice = rcs_slice[:, :, :, 0]
    phase_slice = self.active_dataset.rcs_phase[np.ix_(az_indices, elev_indices, freq_indices, [pol_idx])]
    phase_slice = phase_slice[:, :, :, 0]
    if not np.any(np.isfinite(phase_slice)):
        self.status.showMessage(
            "3D ISAR requires phase-aware samples; selected data has no finite rcs_phase."
        )
        return
    rcs_slice = np.where(np.isfinite(rcs_slice), rcs_slice, 0.0)

    win_az = self._isar_window(az_values.size)
    win_el = self._isar_window(elev_values.size)
    win_freq = self._isar_window(freq_hz.size)
    window = win_az[:, None, None] * win_el[None, :, None] * win_freq[None, None, :]
    rcs_windowed = rcs_slice * window

    az_rad = np.deg2rad(az_values)
    elev_rad = np.deg2rad(elev_values)
    k_mag = (4.0 * np.pi / C0) * freq_hz

    cos_az = np.cos(az_rad)[:, None, None]
    sin_az = np.sin(az_rad)[:, None, None]
    cos_el = np.cos(elev_rad)[None, :, None]
    sin_el = np.sin(elev_rad)[None, :, None]
    kx = k_mag[None, None, :] * cos_el * cos_az
    ky = k_mag[None, None, :] * cos_el * sin_az
    kz = np.ones((az_values.size, 1, 1), dtype=float) * k_mag[None, None, :] * sin_el

    sample_values = rcs_windowed.ravel()
    kx_flat = kx.ravel()
    ky_flat = ky.ravel()
    kz_flat = kz.ravel()
    valid = (
        np.isfinite(sample_values)
        & np.isfinite(kx_flat)
        & np.isfinite(ky_flat)
        & np.isfinite(kz_flat)
    )
    if not np.any(valid):
        self.status.showMessage("3D ISAR reconstruction failed: no finite samples.")
        return

    sample_values = sample_values[valid]
    kx_flat = kx_flat[valid]
    ky_flat = ky_flat[valid]
    kz_flat = kz_flat[valid]

    kx_min, kx_max = float(np.min(kx_flat)), float(np.max(kx_flat))
    ky_min, ky_max = float(np.min(ky_flat)), float(np.max(ky_flat))
    kz_min, kz_max = float(np.min(kz_flat)), float(np.max(kz_flat))
    if (kx_max - kx_min) <= 1e-9 or (ky_max - ky_min) <= 1e-9 or (kz_max - kz_min) <= 1e-9:
        self.status.showMessage(
            "3D ISAR needs wider azimuth/elevation/frequency aperture to resolve all dimensions."
        )
        return

    # Use per-dimension grid sizes: freq drives range (x), az drives cross-range (y),
    # el drives height (z). Keeping each axis proportional to its sample count produces
    # better-balanced resolution than a single cubic grid.
    nx = _bounded_grid_size(2 * freq_hz.size)
    ny = _bounded_grid_size(2 * az_values.size)
    nz = _bounded_grid_size(2 * elev_values.size)

    ix = np.rint((kx_flat - kx_min) / (kx_max - kx_min) * (nx - 1)).astype(np.int64)
    iy = np.rint((ky_flat - ky_min) / (ky_max - ky_min) * (ny - 1)).astype(np.int64)
    iz = np.rint((kz_flat - kz_min) / (kz_max - kz_min) * (nz - 1)).astype(np.int64)
    ix = np.clip(ix, 0, nx - 1)
    iy = np.clip(iy, 0, ny - 1)
    iz = np.clip(iz, 0, nz - 1)

    linear_index = (ix * ny + iy) * nz + iz
    k_grid = np.zeros((nx, ny, nz), dtype=np.complex128)
    weight_grid = np.zeros((nx, ny, nz), dtype=np.float64)
    np.add.at(k_grid.ravel(), linear_index, sample_values)
    np.add.at(weight_grid.ravel(), linear_index, 1.0)
    populated = weight_grid > 0.0
    if not np.any(populated):
        self.status.showMessage("3D ISAR reconstruction failed during k-space gridding.")
        return
    k_grid[populated] = k_grid[populated] / weight_grid[populated]

    image_volume = np.fft.ifftn(np.fft.ifftshift(k_grid, axes=(0, 1, 2)))
    image_volume = np.fft.fftshift(image_volume, axes=(0, 1, 2))
    magnitude = np.abs(image_volume)
    peak = float(np.max(magnitude))
    if peak <= 0.0:
        self.status.showMessage("3D ISAR reconstruction failed: zero image energy.")
        return

    linear_scale = self._plot_scale_is_linear()
    if linear_scale:
        volume_display = magnitude
    else:
        volume_display = self.active_dataset.rcs_to_dbsm(magnitude)

    dkx = (kx_max - kx_min) / float(max(nx - 1, 1))
    dky = (ky_max - ky_min) / float(max(ny - 1, 1))
    dkz = (kz_max - kz_min) / float(max(nz - 1, 1))
    x_axis = np.fft.fftshift(np.fft.fftfreq(nx, d=dkx / (2.0 * np.pi)))
    y_axis = np.fft.fftshift(np.fft.fftfreq(ny, d=dky / (2.0 * np.pi)))
    z_axis = np.fft.fftshift(np.fft.fftfreq(nz, d=dkz / (2.0 * np.pi)))

    finite_mask = np.isfinite(volume_display)
    if not np.any(finite_mask):
        self.status.showMessage("3D ISAR reconstruction failed: no finite voxels.")
        return
    finite_values = volume_display[finite_mask]
    threshold = float(np.quantile(finite_values, voxel_quantile))
    voxel_mask = finite_mask & (volume_display >= threshold)
    voxel_indices = np.flatnonzero(voxel_mask)
    if voxel_indices.size == 0:
        voxel_indices = np.flatnonzero(finite_mask)
    if voxel_indices.size > max_render_points:
        flat_vals = volume_display.ravel()
        top_idx = np.argpartition(flat_vals, -max_render_points)[-max_render_points:]
        top_idx = top_idx[np.isfinite(flat_vals[top_idx])]
        voxel_indices = top_idx

    voxel_values = volume_display.ravel()[voxel_indices]
    ii, jj, kk = np.unravel_index(voxel_indices, volume_display.shape)
    x_pts = x_axis[ii]
    y_pts = y_axis[jj]
    z_pts = z_axis[kk]

    self._remove_colorbar()
    self.plot_figure.clear()
    self.plot_ax = self.plot_figure.add_subplot(111, projection="3d")
    self.plot_axes = None
    self.plot_figure.set_facecolor(self._current_plot_bg())
    _set_3d_axes_style(self)

    zmin = self.spin_plot_zmin.value()
    zmax = self.spin_plot_zmax.value()
    use_clamp = zmin < zmax
    if use_clamp:
        norm = Normalize(vmin=zmin, vmax=zmax)
    else:
        vmin = float(np.min(voxel_values))
        vmax = float(np.max(voxel_values))
        if np.isclose(vmax, vmin):
            vmax = vmin + 1e-9
        norm = Normalize(vmin=vmin, vmax=vmax)

    scatter = self.plot_ax.scatter(
        x_pts,
        y_pts,
        z_pts,
        c=voxel_values,
        cmap=self._effective_colormap(),
        norm=norm,
        s=point_size,
        linewidths=0.0,
        depthshade=True,
    )

    pol_value = self.active_dataset.polarizations[pol_idx]
    self.plot_ax.set_title(
        (
            f"3D ISAR Volume | Pol {pol_value} | "
            f"Az {az_values.min():.2f}..{az_values.max():.2f} deg, "
            f"El {elev_values.min():.2f}..{elev_values.max():.2f} deg"
        ),
        color=self._current_plot_text(),
    )
    self.plot_ax.set_xlabel("X Range (m)")
    self.plot_ax.set_ylabel("Y Range (m)")
    self.plot_ax.set_zlabel("Z Range (m)")

    if self.chk_colorbar.isChecked():
        colorbar = self.plot_figure.colorbar(scatter, ax=self.plot_ax, pad=0.12, shrink=0.75)
        self.plot_colorbars = [colorbar]
        self._apply_colorbar_ticks(colorbar)
        colorbar.set_label(_colorbar_label(linear_scale), color=self._current_plot_text())
        colorbar.ax.tick_params(colors=self._current_plot_text())
        for label in colorbar.ax.get_yticklabels():
            label.set_color(self._current_plot_text())
    else:
        self.plot_colorbars = []

    self.spin_plot_xmin.blockSignals(True)
    self.spin_plot_xmax.blockSignals(True)
    self.spin_plot_ymin.blockSignals(True)
    self.spin_plot_ymax.blockSignals(True)
    self.spin_plot_xmin.setValue(float(np.min(x_axis)))
    self.spin_plot_xmax.setValue(float(np.max(x_axis)))
    self.spin_plot_ymin.setValue(float(np.min(y_axis)))
    self.spin_plot_ymax.setValue(float(np.max(y_axis)))
    self.spin_plot_xmin.blockSignals(False)
    self.spin_plot_xmax.blockSignals(False)
    self.spin_plot_ymin.blockSignals(False)
    self.spin_plot_ymax.blockSignals(False)

    self._apply_plot_limits()

    thinning_note = []
    if az_thinned:
        thinning_note.append(f"Az->{az_values.size}")
    if elev_thinned:
        thinning_note.append(f"El->{elev_values.size}")
    if freq_thinned:
        thinning_note.append(f"Freq->{freq_values.size}")
    thinning_text = ""
    if thinning_note:
        thinning_text = " Downsampled: " + ", ".join(thinning_note) + "."
    if not auto_thin:
        thinning_text += " Auto-thinning off."

    self.status.showMessage(
        (
            f"3D ISAR volume updated. Grid {nx}x{ny}x{nz}, "
            f"displaying {voxel_indices.size} voxels (q={voxel_quantile:.4f}, max={max_render_points})."
            f"{thinning_text}"
        )
    )
