# Atmosphere, wind and turbulence

**Code:** `src/plume/physics/atmosphere.py`, `src/plume/physics/wind.py`, `src/plume/physics/turbulence.py`
**Selection:** `WorldSpec.atmosphere_model.model` (`us76` | `nrlmsise00` | `sounding`), plus `WorldSpec.wind`
**Verification status:** verified against published tables and reference outputs (`tests/vv/test_environment.py`). Validation against measured soundings is pending.

## U.S. Standard Atmosphere 1976 (`us76`, default)

**Below 86 km:** seven-layer hydrostatic model in geopotential altitude, with r₀ = 6 356.766 km, g₀ = 9.80665 m/s² and R = 287.053 J/(kg·K). Matches the published table (NOAA-S/T 76-1562, Table I) at 0, 10, 20, 50 and 80 km:
- temperature within 0.02 K
- pressure and density within 0.03 %

**86–1000 km:** the air is no longer well mixed, and US76 is defined there by species diffusion equations. Plume uses the published table values (20 levels):
- monotone cubic (Fritsch–Carlson) interpolation in log p and log ρ
- linear interpolation in temperature
- the speed of sound uses the local specific gas constant p/(ρT)

At 250 km, which is not a table node, density is within 3 % of the published value.

**Above 1000 km:** exponential extrapolation.

**Off-standard day:** `temperature_offset` shifts T below 86 km at standard pressure, so density changes as 1/T.

## NRLMSISE-00 (`nrlmsise00`)

Picone et al. (2002) empirical model via the `nrlmsise00` package (extra `hifi`).

- **Inputs:** epoch (UTC), F10.7 (previous day), F10.7a (81-day mean), daily Ap, and the vehicle's geodetic position.
- **Pressure:** p = n·k·T, summing the number densities of He, O, N₂, O₂, Ar, H and N.
- **Profiles:** a vertical profile (1 km steps to 120 km, 5 km to 1000 km) is cached at the current position and refreshed after 50 km of horizontal travel. Interpolated values agree with direct evaluation to within 2 %.
- **Check:** the official distribution test case 1 reproduces ρ = 4.074714e-15 g/cm³ and T∞ = 1250.54 K.

## Soundings (`sounding`)

`atmosphere_model.sounding: <csv>` (a path, or a name in `configs/winds/`).

- **Columns** are matched case-insensitively. Both plain SI names and University of Wyoming names work:
  - altitude: `HGHT`, `altitude_m`
  - temperature: `TEMP` °C, `temp_k`
  - pressure: `PRES` hPa, `pressure_pa`
  - wind: `DRCT` with `SKNT`, `speed_mps` with `from_deg`, or `east_mps` with `north_mps`
- **Pressure:** taken from the file where measured; elsewhere integrated hydrostatically with the layer-mean temperature.
- **Above the sounding top:** the temperature offset from US76 fades out over 10 km, and the profile then joins US76, scaled for continuity.
- **Check:** a sounding built from US76 temperatures reproduces US76 pressure to 0.2 % at 5, 15 and 29 km.

## Mean wind

- **Parametric** (`WindModel`):
  - power-law shear, exponent 0.143 up to `shear_top`
  - constant above that, fading to zero at `fade_top`
  - optional discrete 1−cos gusts
  - optional first-order turbulence (`turbulence`, σ in m/s), intended for fast mode and RL
- **Measured or forecast profile:** `wind.profile: <csv>` uses the same column conventions as soundings. It is linearly interpolated and held constant beyond the ends.

## Continuous turbulence: MIL-F-8785C / MIL-HDBK-1797

Enabled with `wind.turbulence_severity: light | moderate | severe`. The model is set by `wind.turbulence_model: dryden | von_karman`, where `auto` means von Kármán.

**Field:** a frozen spatial random field (Taylor's hypothesis), sampled along the path the vehicle travels relative to the mean wind. A hovering vehicle in a breeze therefore still sees turbulence carried past it. Each component is a sum of 160 log-spaced cosine modes with random phases. Mode amplitudes follow the exact spectrum and are normalised to the specified variance:

| | longitudinal Φᵤ(Ω) | lateral / vertical Φᵥ(Ω) |
|---|---|---|
| Dryden | σ² (2L/π) / (1 + (LΩ)²) | σ² (L/π)(1 + 3(LΩ)²) / (1 + (LΩ)²)² |
| von Kármán | σ² (2L/π) / (1 + (1.339LΩ)²)^(5/6) | σ² (L/π)(1 + 8/3 (1.339LΩ)²) / (1 + (1.339LΩ)²)^(11/6) |

**Intensities and scale lengths** (h is height above ground, in feet):
- **Below 1000 ft:**
  - L_w = h, L_u = L_v = h / (0.177 + 0.000823h)^1.2
  - σ_w = 0.1·W₂₀, σ_u = σ_v = σ_w / (0.177 + 0.000823h)^0.4
  - W₂₀ = 15, 30 or 45 kt for light, moderate or severe
- **Above 2000 ft:** isotropic. L = 1750 ft (Dryden) or 2500 ft (von Kármán). σ comes from the MIL-F-8785C probability-of-exceedance table: 10⁻² for light, 10⁻³ for moderate, 10⁻⁵ for severe.
- **1000–2000 ft:** linear blend between the two.

**Axes at low altitude:** u along the mean wind (or along the horizontal air path when calm), v horizontal, w vertical.

**Checks:**
- component RMS within 12 % of σ, for both spectra
- Dryden longitudinal autocorrelation at one scale length is e⁻¹ ± 0.08
- low-altitude formulas match exactly

## Known limitations

- Humidity is not modelled. Its effect on density is below 1 % for typical surface conditions.
- Wind is horizontally uniform. It varies with altitude and time, plus turbulence along the path.
- There is no terrain-induced flow, such as mountain waves or rotors.
- Turbulence amplitudes rescale with altitude on a fixed phase field. This is continuous, but not a strict non-stationary process.
