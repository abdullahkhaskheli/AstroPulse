# ASTROPULSE // CHeCS-DS

**Crew Health Care System — Deep Space** is a Streamlit demonstration console for the 2026 NASA Space Apps Challenge, “Create Health Monitoring Software for Astronauts on Space Missions.”

The app demonstrates synthetic crew and cabin telemetry, threshold-based local triage, measurement procedure cards, CSV analysis, and compact delay-tolerant-network (DTN) packet previews. It is not a certified medical device, operational flight software, NASA system, or source of medical advice.

## Features

- Flight-director strip with mission elapsed time, readiness score, edge latency, and simulated Earth one-way light time.
- Four crew profiles, two display modes, five scenario injections, and a 0–22 minute communication-delay control.
- Six live KPI channels with individual crew reference comparisons and an autonomous triage checklist.
- Synthetic ECG/heart-rate, HRV recovery, and multi-modal correlation charts.
- Six non-physician measurement guides, manual reading entry, and readiness recalculation.
- CSV validation and combined configured-z-score/threshold anomaly analysis with a downloadable `.txt` report.
- Optional local display of `data/LSDS-8_Multiplex_serum_cardiovascular_EvePanel_TRANSFORMED.csv` in Tabs 3 and 4, with a sample table and selectable, within-crew biomarker comparison normalized to each crew's L-3 sample.
- Compact JSON and zlib DTN payload previews. Telemetry remains local; the app does not transmit data to Earth.
- On first run, generates `data/nasa_osdr_sample_telemetry.csv` when it does not already exist. The generated data is synthetic and includes one simulated CO2/SpO2 event.

## Run locally

Python 3.10 or newer is recommended.

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
streamlit run app.py
```

If PowerShell script execution is restricted, activate the environment using the appropriate command for your shell or invoke `.venv\Scripts\python.exe -m streamlit run app.py`.

## Create the automated demo video

Start the app in one terminal with `streamlit run app.py`, then install the recording tools and Chromium in another:

```powershell
pip install -r requirements-demo.txt
playwright install chromium
python make_final_video.py
```

The script uses Microsoft Edge TTS over the internet and records the four URL-driven demo views. It writes `raw_video.webm`, the 1.0x `voiceover.mp3`, and `AstroPulse_Final_Demo.mp4` in the project root; the finished video is rendered at 1920×1080 and capped below 29.5 seconds. The supplied narration takes about 35.6 seconds at the requested normal speech rate, so FFmpeg's `-shortest` mux cuts off its final portion in the MP4; `voiceover.mp3` retains the full narration.

## Deploy on Streamlit Community Cloud (3 steps)

1. Push this repository to a GitHub repository. Ensure `app.py`, `requirements.txt`, and `data/nasa_std_3001_osdr.json` are in the repository root / expected paths.
2. In [Streamlit Community Cloud](https://share.streamlit.io/), select **Create app**, choose the repository and branch, and set the main file path to `app.py`.
3. Select **Deploy**. Streamlit installs `requirements.txt`; the app creates the synthetic sample CSV in its runtime `data/` directory if absent.

## NASA Data & Resources Used

Copy/paste submission text:

> ASTROPULSE cites NASA-STD-3001 Volume 2, Revision E (NASA Technical Reports Server record 20250004555) as a human-system design resource; NASA’s Open Science Data Repository (OSDR) as the open repository for space-life-science datasets; NASA’s Human Research Program (HRP) for human-health and performance research context; and a historical NASA Technical Reports Server record for Crew Health Care System (CHeCS) engineering context. When present, `data/LSDS-8_Multiplex_serum_cardiovascular_EvePanel_TRANSFORMED.csv` is displayed as the local OSD-575 Inspiration4 clinical panel and is kept separate from synthetic telemetry. See the app’s Data Citations tab and `data/nasa_std_3001_osdr.json` for source links and provenance notes. The application does not claim NASA endorsement. All generated telemetry is synthetic. Numeric operating bands in this prototype are challenge-brief demonstration settings, not verified NASA-STD-3001 requirements, approved mission flight rules, or clinical limits. Validate all limits and procedures against controlled current mission documentation before operational use.

### Source links

- [NASA-STD-3001 Volume 2, Revision E (NTRS 20250004555)](https://ntrs.nasa.gov/citations/20250004555) — NASA Spaceflight Human-System Standard, Human Factors, Habitability, and Environmental Health.
- [NASA Open Science Data Repository](https://science.nasa.gov/biological-physical/data/osdr/) — repository overview and search links.
- [NASA OSDR repository search](https://osdr.nasa.gov/bio/repo/) — explore public study records and datasets.
- [NASA Human Research Program](https://www.nasa.gov/hrp/) — research program overview.
- [NASA CHeCS design-research record (NTRS 20110011351)](https://ntrs.nasa.gov/citations/20110011351) — historical engineering and operations context only, not a current flight procedure.

## Data provenance and safety

`data/nasa_std_3001_osdr.json` holds the demo's operational bands, crew baselines, SOP text, and source metadata. The challenge-provided numbers are retained as **configured demonstration references**; they are not presented as direct quotations from NASA standards. If present, the transformed OSD-575 panel CSV is loaded locally for table/chart viewing; it is not combined with generated telemetry and is not transmitted by the app. The separate generated mission-telemetry CSV and live values are synthetic.

The SOP panels are educational measurement reminders, not controlled CHeCS procedures. Always use mission-approved devices, training, flight rules, and onboard medical checklists. The scenario simulator and readiness score must not be used to determine fitness for duty or guide clinical treatment.

## CSV format

Upload a comma-separated file with these exact required columns:

```text
timestamp,hr_bpm,rr_ms,spo2_pct,temp_c,co2_mmhg,rad_msv_hr
```

Timestamps must be parseable as UTC timestamps. The analyzer flags configured absolute z-score excursions and the demo's clinical-threshold crossings. This is a visualization demonstration, not validated anomaly-detection software.
