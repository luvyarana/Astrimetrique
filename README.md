# Astrimetrique 🌌

**Astrimetrique** is a professional, macOS-native scientific tool for high-precision astrometric data reduction. Designed as a modern spiritual successor to *Astrometrica*, it translates CCD pixel coordinates into sub-arcsecond Right Ascension (RA) and Declination (Dec) measurements and outputs Minor Planet Center (MPC) 80-column reports.

---

## Key Features

- 🔭 **Astronomical FITS Engine**:
  - Load single and multi-extension `.fits`, `.fit`, `.fts` images via `astropy.io.fits`.
  - Automatic `ZScale` contrast stretching, Logarithmic, Square Root, Linear, Min-Max, and Histogram Equalization.
  - Inverted / Negative color mode for spotting faint minor planets and stars.
  - Interactive FITS Header Inspector.

- 🛸 **Sub-Pixel Image Registration & Warping**:
  - Computes 6-parameter affine coordinate transformations from matched star pairs or LSPC plate solutions.
  - High-precision sub-pixel image warping via bilinear (`order=1`) and bicubic spline (`order=3`) interpolation with `scipy.ndimage.map_coordinates`.
  - Recovers fractional-pixel shifts and rotations with $<0.1$ px alignment residual.

- ⚡ **Multi-Epoch Blinking Engine**:
  - High-speed visual frame toggling (0.1s to 2.0s / 100ms to 2000ms variable intervals) between aligned observation epochs.
  - **Locked Uniform ZScale**: Dynamic range is locked uniformly across frames to eliminate brightness flicker and eye strain during rapid blinking.

- 🔍 **Photometric Diffing & Transient Detection**:
  - Robust background normalization and source-signal linear regression gain matching.
  - Signed difference map ($\Delta I = I_A - s \cdot I_{B,\text{aligned}}$) and absolute difference map.
  - Automatic $\sigma$-clipped noise thresholding and automated moving asteroid candidate extraction with sub-pixel centroids and $S/N$ metrics.

- 🎯 **Precision Interaction & Sub-Pixel Centroiding**:
  - Non-linear 2D Gaussian Levenberg-Marquardt fitting (`scipy.optimize.curve_fit`).
  - Estimates sub-pixel centroid coordinates $(x_0, y_0)$, FWHM, SNR, Peak Flux, and local background.
  - Automatic Center-of-Mass and marginal Gaussian fallback for faint or crowded sources.

- 📐 **Rigorous Least-Squares Plate Solver (LSPC)**:
  - Standard tangent-plane (gnomonic) celestial projection $(\alpha, \delta) \leftrightarrow (\xi, \eta)$.
  - 6-constant affine transformation matrix solved using Singular Value Decomposition (SVD).
  - Accurate physical derivation: Plate Scale ($''/\text{px}$), Field Rotation Angle ($\theta$), and Center Offset.
  - Calculates sub-arcsecond residuals ($\Delta\alpha\cos\delta, \Delta\delta, \text{Total RMS}$).

- ✨ **Gaia DR3 Catalog API Integration**:
  - Direct online query to ESA Gaia DR3 catalog via `astroquery.gaia` and VizieR fallback.
  - 1-click automatic Gaia coordinate and magnitude fetching for selected stars.
  - Field-wide Gaia DR3 cone search.

- 📡 **Minor Planet Center (MPC) 80-Column Exporter**:
  - Formats solved minor bodies and transient objects into standard MPC 80-column lines.
  - Supports numbered & provisional designations, discovery flags, fractional-day UTC, sexagesimal formatting, magnitude/band, and observatory codes.
  - One-click copy-to-clipboard and `.mpc` / `.txt` report export.

- 🎨 **Obsidian Space UI**:
  - Native macOS dark theme styled with deep charcoal backgrounds (`#0B0E14`), neon-cyan highlights (`#00F2FE`), and monospace coordinate readouts.
  - Hardware-accelerated viewport with high-precision crosshair cursor and real-time ADU / sky coordinate inspector.

---

## Installation & Quick Start

### 1. Prerequisites
- macOS (Apple Silicon / ARM64 or Intel)
- Python 3.11+ (Python 3.12+ recommended)

### 2. Environment Setup
```bash
# Clone the repository
git clone https://github.com/your-org/Astrimetrique.git
cd Astrimetrique

# Create and activate virtual environment
uv venv .venv
source .venv/bin/activate

# Install dependencies
uv pip install -e .
```

### 3. Launch Astrimetrique
```bash
python3 main.py
```

### 4. Run Automated Test Suite
```bash
pytest tests/ -v
```

---

## Theoretical Overview

### Tangent Plane Projection (Gnomonic Coordinates)
Given celestial coordinates $(\alpha, \delta)$ and tangent center $(\alpha_0, \delta_0)$:
$$\xi = \frac{\cos\delta \sin(\alpha - \alpha_0)}{\sin\delta \sin\delta_0 + \cos\delta \cos\delta_0 \cos(\alpha - \alpha_0)}$$
$$\eta = \frac{\sin\delta \cos\delta_0 - \cos\delta \sin\delta_0 \cos(\alpha - \alpha_0)}{\sin\delta \sin\delta_0 + \cos\delta \cos\delta_0 \cos(\alpha - \alpha_0)}$$

### Least-Squares Plate Constants (6-Constant Model)
The linear transformation connecting CCD coordinates $(x, y)$ to standard coordinates $(\xi, \eta)$:
$$\xi = ax + by + c$$
$$\eta = dx + ey + f$$

Solved via normal equations:
$$\mathbf{A}^T \mathbf{A} \mathbf{p} = \mathbf{A}^T \mathbf{b}$$
Where $\mathbf{A} = \begin{bmatrix} x_i & y_i & 1 \end{bmatrix}$.

---

## License
Open-source under the MIT License.
