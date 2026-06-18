# Data

Input series for the book, grouped by the system they belong to. Each `<system>/`
folder holds the raw series used by the Chapter 1–3 case studies; `benchmarks/`
holds the four series used by the Chapter 9 forecasting benchmark (some are the
same records in a different form).

## Policy on redistribution

Every series the book analyzes is **bundled here** so the code runs out of the box.
Recompute any checksum with `sha256sum <file>`. Two notes:

- The **S&P 500** daily close is factual market data sourced from Stooq (`^SPX`). It
  is bundled for research and educational use; `python data/fetch_sp500.py` refreshes
  it to the latest date if you want a longer record.
- The **GOY turbulence** series ships downsampled (every 4th step) to keep the file
  small; the full-resolution run is reproducible with
  `python case_studies/turbulence_goy/dynamics_goy_shell.py`.

The code license (root `LICENSE`) covers the code only; each dataset keeps the
terms of its original provider, listed below.

## Provenance

| series | file | source | terms |
|---|---|---|---|
| Santa Fe laser (full) | `benchmarks/santafe_laser_full.txt` | 1991–92 Santa Fe competition, set A; mirrored by the CHARC repo | research/educational use |
| Santa Fe laser (case study) | `santafe_laser/santafe.txt` | standardised first 4000 samples of the full record | research/educational use |
| ENSO Niño 3.4 | `enso_nino34/enso_nino34.txt`, `benchmarks/enso_nino34.txt` | NOAA PSL `nina34.anom.data` | U.S. Government work, public domain |
| Mackey–Glass | `benchmarks/mackeyglass_tau17.txt` | generated from the DDE (β=0.2, γ=0.1, n=10, τ=17), RK4 | generated here, public domain |
| Sunspot number | `sunspot/sunspot_monthly.txt` | SILSO / WDC-SILSO, Royal Obs. of Belgium | free for non-commercial use, cite SILSO |
| Central England Temp. | `cet/cet_monthly_temps.txt` | UK Met Office Hadley Centre (HadCET) | Open Government Licence |
| England & Wales precip. | `ewp/ewp_monthly_precip.txt` | UK Met Office (HadEWP) | Open Government Licence |
| AMO index | `amo/amo_monthly.txt` | NOAA PSL | U.S. Government work, public domain |
| Nile annual minima | `nile/nile_annual.txt` | classical historical record | public domain |
| Fremantle sea level | `fremantle/fremantle_monthly_*.txt` | PSMSL | free for research, cite PSMSL |
| GOY shell turbulence | `turbulence_goy/goy_shell_data.npz` **(downsampled)** | generated from the GOY shell model | generated here, public domain |
| Lorenz-63 / Lorenz-96 | (generated in-script) | `case_studies/lorenz_*/dynamics_*.py` | generated here, public domain |
| S&P 500 daily close | `sp500/sp500_daily_close.txt`, `benchmarks/sp500_daily_close.txt` | Stooq ^SPX daily, refresh via `fetch_sp500.py` | factual price data; research/educational use |

Original sources:
- Santa Fe laser: `https://raw.githubusercontent.com/MaterialMan/CHARC/master/Support%20files/other/Datasets/laser.txt`
  (Weigend & Gershenfeld 1993; Hübner, Abraham & Weiss, *Phys. Rev. A* 40, 6354, 1989).
- ENSO Niño 3.4: `https://psl.noaa.gov/data/correlation/nina34.anom.data`.
- SILSO sunspots: `https://www.sidc.be/SILSO/datafiles`.
- HadCET / HadEWP: `https://www.metoffice.gov.uk/hadobs/`.

## Checksums (bundled series)

```
santafe_laser/santafe.txt         651f07fac52eaeb874a0ba3962a6b2355f7ea3421ac736810a83370161246114
benchmarks/santafe_laser_full.txt 2445f3df2b91cfb41c3f4f1143e8882e8329b9449ec7ffc739c6d4bd5c6650a0
benchmarks/enso_nino34.txt         e4cf102f0bac8ce9069376d7d4919e35c1ac40bf1e11dc8a9e5b337f86c6cb8a
benchmarks/mackeyglass_tau17.txt   59df4c84210823c8d8a666421453a3665dae6ba677e429ca3b5e3bbdb8da682d
benchmarks/sp500_daily_close.txt   7af3f592bdd96d3937791592c54ed313f7895920a0e5640d1dc2b13150ef0167
turbulence_goy/goy_shell_data.npz  fa4e1f1a655a81a855da9248521535daa3134dd91f85365497908eef639aadf1
```
(The S&P file refreshes to the latest date if you re-run `fetch_sp500.py`, which
changes its checksum.)
