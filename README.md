# El Niño: from Pacific buoys to Los Angeles rain

BANA 212 (UC Irvine MSBA), Team 6. An interactive companion to our project notebook: what the equatorial Pacific buoy array
says about El Niño and La Niña, how early it can call a change, and what an El Niño fall has meant for winter rain in Los Angeles.

**Live app:** https://bana212-elnino-la-rain.streamlit.app

## The questions and what we found

Results were frozen on 22 Nov 2026; every number below comes from `data/summary.json`.

| Question | Answer |
|---|---|
| **Q1** How do ocean and air conditions along the equator differ between El Niño, La Niña and neutral months? | El Niño months are warmer with weaker trade winds. El Niño − La Niña gap in daily anomalies: sea surface temperature 1.42 SD, air temperature 1.33 SD, zonal wind 0.67 SD (95% intervals from a bootstrap over whole years exclude 0); humidity and meridional wind barely differ. The gaps hold on the 1998–2026 buoy data (1.40, 1.31, 0.63 SD). |
| **Q2** Which buoy readings signal the phase best, and can they call it earlier than NOAA's index? | Three months ahead, logistic regression on buoy readings plus the latest ONI reaches a balanced accuracy of 0.71 against 0.58 for persistence over 28 walk-forward test years (1999–2026) and catches 73 of 141 phase changes; on the 1995–98 test it scores 0.79 against 0.52. At 0–1 months the latest ONI is hard to beat. |
| **Q3** After an El Niño fall, how much more likely is a wet Los Angeles winter? | 65% of winters after an El Niño fall (13 of 20) were wetter than the 1950–2025 median, against 45% after other falls (25 of 56), with 5.7 vs 3.75 heavy-rain days (≥ 1 inch) per season. |
| **Winter 2026–27** | The Aug–Oct 2026 ONI is +2.46, the highest since 1950. In October 2026 the Niño 3.4 buoys read +3.14 °C and the model calls El Niño through April 2027. The odds lean towards a wet, stormy winter: a seasonal risk outlook, not a forecast. |

## The app

| Tab | Contents |
|---|---|
| Executive Summary | The questions, the data, the three findings and this winter's outlook. |
| Descriptive Analytics | Buoys reporting per month, ONI since 1950 by phase, El Niño vs La Niña along the equator, Q1 on the new years, LA season rain after El Niño and other falls. |
| Model Performance | Balanced accuracy by lead time from the frozen results (logistic regression, random forest, persistence; main split and walk-forward), what the final models rely on (logistic coefficients, random-forest importances), and a hyperparameter-tuning section: best settings per model and horizon, the cross-validation table, and tuned vs fixed settings on the held-out tests. |
| Interactive Prediction | Sliders for the nine model inputs (historical ranges, starting at the latest observed month, October 2026) and the phase probabilities 0, 1, 3 and 6 months ahead from the saved models, with the matching LA wet-winter odds. |

The app only loads saved files; it never fits a model.

## Repository layout

```
app.py                            Streamlit app
requirements.txt                  pinned packages (pandas, numpy, scikit-learn match the notebook run)
.streamlit/config.toml            theme
notebooks/
  elnino_analysis_v3.ipynb        the frozen analysis (sections 1–10), executed 22 Nov 2026
  elnino_analysis_v4.ipynb        v3 + section 11 (hyperparameter tuning) + section 12 (export for the app); only the title cell differs
models/
  logistic_k{0,1,3,6}.joblib      logistic regression per horizon, fixed settings of the report (the app's main model)
  random_forest_k{0,1,3,6}.joblib random forest per horizon, fixed settings of the report
  metadata.json                   feature order, class order, horizons, training window, best settings, CV tables, test scores, versions
data/
  summary.json                    every number quoted in the project, frozen 22 Nov 2026
  model_inputs_monthly.csv        the nine model inputs for every month with ≥ 5 buoys, 1985–2026
  buoys_monthly.csv               buoys reporting SST per month and the Niño 3.4-box buoy anomaly, 1980–2026
  oni_monthly.csv                 ONI and RONI by month since 1950
  la_rain_seasons.csv             Los Angeles Downtown rain seasons 1950–2025 with the fall ONI and RONI
  q2_test_scores.csv              held-out test scores for every model, setting and horizon
```

## Models

* **Target:** the ONI phase of month *t + k* (El Niño if ONI ≥ +0.5, La Niña if ≤ −0.5, else Neutral), forecast at the start of
  month *t + 1* from the buoy anomalies of month *t* and the latest official ONI (month *t − 1*).
* **Inputs (9):** Niño 3.4-box SST, all-buoy SST, air temperature, all-buoy zonal wind, west-Pacific zonal wind and meridional wind
  anomalies, the one-month changes in Niño 3.4 SST and west-Pacific wind, and the latest ONI.
* **Models:** multinomial logistic regression (median imputation, standardisation, balanced class weights) and a random forest,
  against a persistence baseline (the latest ONI phase continues). Score: balanced accuracy (0.33 = guessing).
* **Final models (notebook section 12):** the fixed settings of the report (logistic C = 1 with balanced class weights; random
  forest with 500 trees, at least 3 months per leaf, random_state 212), one pipeline per horizon, fitted exactly like the models
  behind the winter 2026–27 outlook: issue months 1996–2025 with anomalies against the same 30 years. The saved logistic models
  reproduce that outlook.
* **Hyperparameter tuning (notebook section 11):** time-based cross-validation on training-period data only, with expanding-window
  folds by year and balanced accuracy pooled over 7 validation years. Grids: C ∈ {0.01, 0.03, 0.1, 0.3, 1, 3, 10}; random forest
  n_estimators ∈ {200, 500} × max_depth ∈ {None, 4, 8} × min_samples_leaf ∈ {1, 3, 5, 10}. The cross-validated choice varied
  across windows and did not consistently beat the fixed settings on held-out years, so the final models keep the fixed settings
  used in the report. The results are in `models/metadata.json` and the app's Model Performance tab.

## Run locally

Python 3.12:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

To rebuild the models, run `notebooks/elnino_analysis_v4.ipynb` top to bottom (Google Colab or Jupyter; the first run downloads
about 150 MB of raw data into `data/raw/`), then copy `app_export/models/` and `app_export/data/` over `models/` and `data/`.

## Data sources

* **UCI El Niño dataset** (TAO buoy array, 1980–1998). UCI Machine Learning Repository, dataset 122.
  <https://archive.ics.uci.edu/dataset/122/el+nino>
* **NOAA PMEL Global Tropical Moored Buoy Array (GTMBA)**, TAO/TRITON daily sea surface temperature, air temperature, winds and
  relative humidity (`pmelTaoDySst`, `pmelTaoDyAirt`, `pmelTaoDyW`, `pmelTaoDyRh`), 1997–2026, via ERDDAP:
  <https://data.pmel.noaa.gov/pmel/erddap/>. Data courtesy of the GTMBA Project Office of NOAA/PMEL. McPhaden, M. J., et al. (1998).
  The Tropical Ocean-Global Atmosphere observing system: A decade of progress. *Journal of Geophysical Research*, 103(C7), 14169–14240.
* **NOAA Climate Prediction Center, Oceanic Niño Index (ONI)**, 1950–2026:
  <https://www.cpc.ncep.noaa.gov/data/indices/oni.ascii.txt>, and the **Relative Oceanic Niño Index (RONI)**:
  <https://www.cpc.ncep.noaa.gov/data/indices/RONI.ascii.txt>.
* **NOAA NCEI GHCN-Daily, station USW00093134** (Los Angeles Downtown, USC), daily precipitation:
  <https://www.ncei.noaa.gov/data/global-historical-climatology-network-daily/access/USW00093134.csv>. Menne, M. J., Durre, I.,
  Vose, R. S., Gleason, B. E., & Houston, T. G. (2012). An overview of the Global Historical Climatology Network-Daily database.
  *Journal of Atmospheric and Oceanic Technology*, 29(7), 897–910.

ONI, RONI and the station file were downloaded on 22 Nov 2026 and the buoy files run to 1 Nov 2026; NOAA revises recent values,
so a rerun on later downloads can move the newest numbers.
