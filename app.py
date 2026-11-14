from __future__ import annotations

import glob
import json
import time
import zlib
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots
from scipy.signal import find_peaks


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
STANDARDS_PATH = DATA_DIR / "nasa_std_3001_osdr.json"
SAMPLE_PATH = DATA_DIR / "nasa_osdr_sample_telemetry.csv"
OSDR_DATASET_LABEL = "Real NASA OSDR Inspiration4 Clinical Dataset (OSD-575)"
OSDR_TIMEPOINT_ORDER = ["L-92", "L-44", "L-3", "R+1", "R+45", "R+82", "R+194"]
MISSION_START = datetime(2026, 4, 1, 0, 0, tzinfo=timezone.utc)
REQUIRED_COLUMNS = [
    "timestamp",
    "hr_bpm",
    "rr_ms",
    "spo2_pct",
    "temp_c",
    "co2_mmhg",
    "rad_msv_hr",
]

COLORS = {
    "canvas": "#06090F",
    "panel": "#0D131F",
    "border": "#1E293B",
    "text": "#F1F5F9",
    "muted": "#CBD5E1",
    "cyan": "#38BDF8",
    "amber": "#F59E0B",
    "red": "#EF4444",
    "green": "#34D399",
}

SCENARIOS = [
    "NOMINAL ORBIT",
    "EVA SUIT LCVG COOLING FAULT",
    "SOLAR PARTICLE EVENT (SPE SURGE)",
    "CDRA CO2 SCRUBBER DEGRADATION",
    "ORTHOSTATIC / ARRHYTHMIA EPISODE",
]

DEMO_STEPS = {
    "1": {"scenario": "NOMINAL ORBIT", "light_time": 0.0, "tab": "live"},
    "2": {
        "scenario": "CDRA CO2 SCRUBBER DEGRADATION",
        "light_time": 14.0,
        "tab": "live",
    },
    "3": {"scenario": "NOMINAL ORBIT", "light_time": 0.0, "tab": "sop"},
    "4": {"scenario": "NOMINAL ORBIT", "light_time": 14.0, "tab": "arch"},
}

BASE_TELEMETRY = {
    "hr_bpm": 68.0,
    "hrv_rmssd_ms": 48.0,
    "hrv_sdnn_ms": 61.0,
    "spo2_pct": 98.0,
    "temp_c": 36.9,
    "co2_mmhg": 1.2,
    "rad_msv_hr": 0.008,
    "glucose_mg_dl": 96.0,
    "rr_ms": 882.0,
}

SCENARIO_OVERRIDES = {
    "NOMINAL ORBIT": {},
    "EVA SUIT LCVG COOLING FAULT": {
        "hr_bpm": 142.0,
        "hrv_rmssd_ms": 18.0,
        "hrv_sdnn_ms": 28.0,
        "spo2_pct": 95.0,
        "temp_c": 38.7,
        "co2_mmhg": 1.3,
        "rad_msv_hr": 0.011,
        "rr_ms": 423.0,
    },
    "SOLAR PARTICLE EVENT (SPE SURGE)": {
        "hr_bpm": 112.0,
        "hrv_rmssd_ms": 29.0,
        "hrv_sdnn_ms": 39.0,
        "spo2_pct": 97.0,
        "temp_c": 37.4,
        "co2_mmhg": 1.4,
        "rad_msv_hr": 4.8,
        "rr_ms": 536.0,
    },
    "CDRA CO2 SCRUBBER DEGRADATION": {
        "hr_bpm": 126.0,
        "hrv_rmssd_ms": 22.0,
        "hrv_sdnn_ms": 31.0,
        "spo2_pct": 90.5,
        "temp_c": 37.2,
        "co2_mmhg": 4.7,
        "rad_msv_hr": 0.009,
        "rr_ms": 476.0,
    },
    "ORTHOSTATIC / ARRHYTHMIA EPISODE": {
        "hr_bpm": 136.0,
        "hrv_rmssd_ms": 16.0,
        "hrv_sdnn_ms": 78.0,
        "spo2_pct": 96.0,
        "temp_c": 37.1,
        "co2_mmhg": 1.2,
        "rad_msv_hr": 0.008,
        "rr_ms": 441.0,
    },
}


st.set_page_config(
    page_title="ASTROPULSE // CHeCS-DS",
    page_icon="A",
    layout="wide",
    initial_sidebar_state="expanded",
)


def find_osdr_dataset_paths() -> list[Path]:
    return sorted(
        Path(path) for path in glob.glob(str(DATA_DIR / "LSDS-8*TRANSFORMED.csv"))
    )


@st.cache_data
def load_standards() -> dict[str, Any]:
    with STANDARDS_PATH.open("r", encoding="utf-8") as file:
        return json.load(file)


@st.cache_data
def load_csv_bytes(content: bytes) -> pd.DataFrame:
    from io import BytesIO

    return pd.read_csv(BytesIO(content))


@st.cache_data
def load_osdr_dataset(path: str) -> pd.DataFrame:
    frame = pd.read_csv(path)
    if "Sample Name" not in frame.columns or frame.empty:
        raise ValueError("The OSDR panel CSV must contain sample rows and a 'Sample Name' column.")
    biomarker_columns = [
        column
        for column in frame.columns
        if "_concentration_" in column or column.endswith("_percent")
    ]
    if not biomarker_columns:
        raise ValueError("The OSDR panel CSV has no numeric biomarker columns.")
    for column in biomarker_columns:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    if frame[biomarker_columns].isna().any().any():
        raise ValueError("The OSDR panel contains missing or non-numeric biomarker values.")
    return frame


def ensure_sample_csv() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if SAMPLE_PATH.exists():
        return
    rng = np.random.default_rng(20261114)
    periods = 180
    timestamps = pd.date_range(
        end=pd.Timestamp.now(tz="UTC").floor("min"),
        periods=periods,
        freq="min",
    )
    hr = 67 + 2.1 * np.sin(np.linspace(0, 8 * np.pi, periods)) + rng.normal(0, 1.6, periods)
    rr = 60000 / hr + rng.normal(0, 19, periods)
    spo2 = 98.0 + rng.normal(0, 0.45, periods)
    temp = 36.8 + rng.normal(0, 0.12, periods)
    co2 = 1.22 + 0.08 * np.sin(np.linspace(0, 3 * np.pi, periods)) + rng.normal(0, 0.035, periods)
    radiation = np.clip(0.008 + rng.normal(0, 0.0015, periods), 0.002, None)

    # A short, synthetic scrubber excursion gives the demo analyzer a visible event.
    event_start = 130
    co2[event_start : event_start + 9] = np.linspace(2.4, 4.65, 9)
    spo2[event_start : event_start + 9] = np.linspace(95.0, 90.7, 9)
    hr[event_start : event_start + 9] = np.linspace(78, 124, 9)
    rr[event_start : event_start + 9] = 60000 / hr[event_start : event_start + 9]
    pd.DataFrame(
        {
            "timestamp": timestamps.astype(str),
            "hr_bpm": np.round(hr, 2),
            "rr_ms": np.round(rr, 2),
            "spo2_pct": np.round(spo2, 2),
            "temp_c": np.round(temp, 3),
            "co2_mmhg": np.round(co2, 3),
            "rad_msv_hr": np.round(radiation, 5),
        }
    ).to_csv(SAMPLE_PATH, index=False)


def apply_styles() -> None:
    st.markdown(
        f"""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600&family=Inter:wght@400;500;600;700&display=swap');
        :root {{
            --canvas: {COLORS["canvas"]};
            --panel: {COLORS["panel"]};
            --border: {COLORS["border"]};
            --text: {COLORS["text"]};
            --muted: {COLORS["muted"]};
            --cyan: {COLORS["cyan"]};
            --amber: {COLORS["amber"]};
            --red: {COLORS["red"]};
        }}
        html, body, [class*="css"] {{
            font-family: Inter, ui-sans-serif, system-ui, sans-serif;
            color: var(--text);
        }}
        .stApp {{ background: var(--canvas); }}
        [data-testid="stHeader"] {{ background: rgba(6,9,15,.96); }}
        [data-testid="stSidebar"] {{ background: #080D16; border-right: 1px solid var(--border); }}
        [data-testid="stSidebar"] > div:first-child {{ padding-top: 1.2rem; }}
        [data-testid="stSidebar"] label,
        [data-testid="stSidebar"] p,
        [data-testid="stSidebar"] [data-testid="stMarkdownContainer"],
        [data-testid="stSidebar"] [data-testid="stCaptionContainer"],
        [data-testid="stSidebar"] [data-testid="stRadio"] div[role="radiogroup"] label div {{
            color: #CBD5E1 !important;
        }}
        [data-testid="stCaptionContainer"],
        .stCaption,
        [data-testid="stTab"][aria-selected="false"],
        [data-testid="stTab"][aria-selected="false"] *,
        [data-testid="stTab"][aria-selected="false"] p {{
            color: #CBD5E1 !important;
        }}
        [data-testid="stTab"][aria-selected="true"],
        [data-testid="stTab"][aria-selected="true"] p {{
            color: var(--cyan) !important;
        }}
        [data-testid="stSelectbox"] [data-baseweb="select"] > div,
        [data-testid="stNumberInput"] input,
        [data-testid="stTextInput"] input,
        [data-testid="stFileUploader"] [data-testid="stFileUploaderDropzone"] {{
            background: #0F172A !important;
            border: 1px solid #334155 !important;
            color: #F1F5F9 !important;
        }}
        [data-testid="stSelectbox"] [data-baseweb="select"] *,
        [data-testid="stNumberInput"] input,
        [data-testid="stTextInput"] input,
        [data-testid="stFileUploader"] [data-testid="stFileUploaderDropzone"] * {{
            color: #F1F5F9 !important;
        }}
        [data-testid="stAlert"] {{
            background: #0F172A !important;
            border: 1px solid #334155 !important;
            color: #F1F5F9 !important;
            border-radius: 0 !important;
        }}
        [data-testid="stAlert"] *,
        [data-testid="stAlert"] p,
        [data-testid="stAlert"] svg {{
            color: #F1F5F9 !important;
            fill: #F1F5F9 !important;
        }}
        h1, h2, h3 {{ letter-spacing: -.025em; color: var(--text); }}
        .mono, .mono * {{ font-family: "IBM Plex Mono", "SFMono-Regular", Consolas, monospace !important; }}
        .eyebrow {{
            color: var(--muted); font-family: "IBM Plex Mono", monospace;
            font-size: .68rem; letter-spacing: .12em; text-transform: uppercase;
        }}
        .top-strip {{
            display: grid; grid-template-columns: 1.25fr 1fr 1fr 1fr 1.1fr;
            gap: 0; border: 1px solid var(--border); background: var(--panel);
            margin: .25rem 0 1.25rem 0;
        }}
        .strip-cell {{ padding: .8rem 1rem; border-right: 1px solid var(--border); min-width: 0; }}
        .strip-cell:last-child {{ border-right: 0; }}
        .strip-label {{ color: var(--muted); font: 500 .63rem "IBM Plex Mono", monospace; letter-spacing: .1em; }}
        .strip-value {{ color: var(--text); font: 600 .95rem "IBM Plex Mono", monospace; margin-top: .35rem; white-space: nowrap; }}
        .status-dot {{ color: var(--cyan); }}
        .kpi {{
            border: 1px solid var(--border); background: var(--panel);
            padding: .95rem 1rem .85rem; min-height: 120px;
        }}
        .kpi-label {{ color: var(--muted); font: 500 .66rem "IBM Plex Mono", monospace; letter-spacing: .08em; }}
        .kpi-value {{ color: var(--text); font: 600 1.8rem "IBM Plex Mono", monospace; margin: .4rem 0 .1rem; }}
        .kpi-unit {{ color: var(--muted); font: 400 .72rem "IBM Plex Mono", monospace; }}
        .kpi-footer {{ color: var(--muted); font: 400 .68rem "IBM Plex Mono", monospace; margin-top: .45rem; }}
        .badge {{
            display: inline-block; font: 600 .62rem "IBM Plex Mono", monospace;
            letter-spacing: .08em; padding: .16rem .4rem; border: 1px solid currentColor;
        }}
        .badge-nominal {{ color: {COLORS["cyan"]}; }}
        .badge-caution {{ color: {COLORS["amber"]}; }}
        .badge-critical {{ color: {COLORS["red"]}; }}
        .panel {{
            background: var(--panel); border: 1px solid var(--border);
            padding: 1rem 1.1rem; margin-bottom: .8rem;
        }}
        .subtle {{ color: var(--muted); font-size: .84rem; line-height: 1.55; }}
        .sop-step {{
            display: flex; gap: .7rem; margin: .55rem 0; line-height: 1.45;
            color: var(--text); font-size: .88rem;
        }}
        .sop-number {{ color: var(--cyan); font-family: "IBM Plex Mono", monospace; min-width: 1.4rem; }}
        div[data-testid="stMetric"] {{
            background: var(--panel); border: 1px solid var(--border);
            padding: .75rem; border-radius: 0;
        }}
        button[kind="primary"] {{ border-radius: 0; }}
        .stTabs [data-baseweb="tab-list"] {{ gap: .3rem; border-bottom: 1px solid var(--border); }}
        [data-testid="stTab"] {{
            color: #CBD5E1; font-family: "IBM Plex Mono", monospace;
            font-size: .78rem; padding: .65rem .85rem;
        }}
        @media (max-width: 900px) {{
            .top-strip {{ grid-template-columns: repeat(2, 1fr); }}
            .strip-cell {{ border-bottom: 1px solid var(--border); }}
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def elapsed_met(now: datetime) -> str:
    seconds = max(0, int((now - MISSION_START).total_seconds()))
    days, seconds = divmod(seconds, 86400)
    hours, seconds = divmod(seconds, 3600)
    minutes, seconds = divmod(seconds, 60)
    return f"{days:03d}:{hours:02d}:{minutes:02d}:{seconds:02d}"


def crew_by_id(crew_id: str, crew_profiles: list[dict[str, Any]]) -> dict[str, Any]:
    return next(profile for profile in crew_profiles if profile["id"] == crew_id)


def live_telemetry(scenario: str, crew: dict[str, Any]) -> dict[str, float]:
    values = BASE_TELEMETRY.copy()
    values["hr_bpm"] = float(crew["baseline_hr_bpm"] + 4)
    values["hrv_rmssd_ms"] = float(crew["baseline_hrv_rmssd_ms"] * 0.94)
    values["spo2_pct"] = float(crew["baseline_spo2_pct"])
    values["temp_c"] = float(crew["baseline_temp_c"])
    values.update(SCENARIO_OVERRIDES[scenario])
    return values


def metric_status(metric: str, value: float, crew: dict[str, Any]) -> str:
    if metric == "hr_bpm":
        if value < 45 or value > 120:
            return "CRITICAL"
        return "NOMINAL" if 55 <= value <= 85 else "CAUTION"
    if metric == "hrv_rmssd_ms":
        if value < crew["baseline_hrv_rmssd_ms"] * 0.7:
            return "CRITICAL"
        return "NOMINAL" if 28 <= value <= 65 else "CAUTION"
    if metric == "spo2_pct":
        if value < 92:
            return "CRITICAL"
        return "NOMINAL" if 96 <= value <= 100 else "CAUTION"
    if metric == "temp_c":
        if value > 38.5 or value < 35.0:
            return "CRITICAL"
        return "NOMINAL" if 36.5 <= value <= 37.5 else "CAUTION"
    if metric == "co2_mmhg":
        if value > 4:
            return "CRITICAL"
        return "NOMINAL" if value < 2 else "CAUTION"
    if metric == "rad_msv_hr":
        if value > 2:
            return "CRITICAL"
        return "NOMINAL" if value < 0.02 else "CAUTION"
    if metric == "glucose_mg_dl":
        if value < 54 or value > 250:
            return "CRITICAL"
        return "NOMINAL" if 70 <= value <= 140 else "CAUTION"
    return "NOMINAL"


def readiness_index(values: dict[str, float], crew: dict[str, Any]) -> int:
    penalty = 0
    for metric in ("hr_bpm", "hrv_rmssd_ms", "spo2_pct", "temp_c", "co2_mmhg", "rad_msv_hr"):
        status = metric_status(metric, float(values[metric]), crew)
        penalty += {"NOMINAL": 0, "CAUTION": 8, "CRITICAL": 20}[status]
    if "glucose_mg_dl" in values:
        glucose_status = metric_status("glucose_mg_dl", float(values["glucose_mg_dl"]), crew)
        penalty += {"NOMINAL": 0, "CAUTION": 8, "CRITICAL": 20}[glucose_status]
    if values.get("vision_change", False):
        penalty += 15
    return max(0, min(100, 100 - penalty))


def z_score(value: float, baseline: float, spread: float) -> float:
    return (value - baseline) / spread if spread > 0 else 0.0


def critical_actions(metric: str, status: str) -> list[str]:
    action_map = {
        "hr_bpm": [
            "Pause nonessential activity; secure the crewmember and assess symptoms using the onboard checklist.",
            "Repeat/confirm the measurement with the approved device and review the ECG signal quality.",
            "Notify the designated medical lead and flight surgeon per the mission communications plan; escalate immediately for chest pain, fainting, severe breathlessness, or persistent abnormal rhythm.",
        ],
        "hrv_rmssd_ms": [
            "Check measurement quality, recent activity, posture, sleep, and symptoms; do not interpret HRV in isolation.",
            "Repeat only as directed by the onboard checklist and compare with the personal baseline trend.",
            "Send the trend and context to the medical lead; escalate if accompanied by symptoms or abnormal ECG.",
        ],
        "spo2_pct": [
            "Stop exertion and follow the onboard oxygen/medical checklist; assess the crewmember and verify probe fit/signal quality.",
            "Recheck once using the approved procedure while observing symptoms and pulse waveform.",
            "Notify the medical lead and flight surgeon promptly for persistent low reading or any respiratory distress.",
        ],
        "temp_c": [
            "Stop heat-generating work, move to the approved thermal safe state, and follow the suit/cooling contingency checklist.",
            "Confirm temperature using the approved method and check thermal-sensor/device status.",
            "Notify the medical lead and flight surgeon; escalate immediately for confusion, collapse, or persistent severe heat symptoms.",
        ],
        "co2_mmhg": [
            "Follow the approved cabin-atmosphere alarm response and verify the sensor/status source; reduce exertion as directed.",
            "Initiate the mission's approved CDRA/cabin contingency actions; do not improvise hardware repair.",
            "Notify the flight director and flight surgeon through the communications plan; report persistent excursion and crew symptoms.",
        ],
        "rad_msv_hr": [
            "Follow the mission radiation alarm/SPE shelter procedure and move to the designated protected location when directed.",
            "Verify personal dosimeter status and preserve event time, dose-rate, location, and activity data.",
            "Notify the radiation authority/flight director and medical lead; remain under the approved exposure-control plan.",
        ],
    }
    actions = action_map.get(metric, [
        "Repeat the measurement only as permitted by the approved onboard medical checklist.",
        "Record symptoms and measurement context; do not self-diagnose or alter treatment.",
        "Notify the designated medical lead and follow the current flight rule for escalation.",
    ])
    if status == "CAUTION":
        return [actions[0], actions[1], "Document the trend and contact the medical lead if the condition persists or symptoms develop."]
    return actions


def figure_layout(fig: go.Figure, height: int = 310) -> go.Figure:
    fig.update_layout(
        height=height,
        margin=dict(l=12, r=12, t=34, b=14),
        paper_bgcolor=COLORS["panel"],
        plot_bgcolor=COLORS["panel"],
        font=dict(color=COLORS["muted"], family="IBM Plex Mono, monospace", size=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        xaxis=dict(gridcolor=COLORS["border"], zerolinecolor=COLORS["border"]),
        yaxis=dict(gridcolor=COLORS["border"], zerolinecolor=COLORS["border"]),
    )
    return fig


def make_ecg_figure(values: dict[str, float], scenario: str) -> go.Figure:
    rng = np.random.default_rng(22)
    time_s = np.linspace(0, 10, 2500)
    rate = max(35.0, values["hr_bpm"])
    beat_period = 60.0 / rate
    ecg = np.zeros_like(time_s)
    beats = []
    beat = 0.25
    while beat < 10:
        beats.append(beat)
        if scenario == "ORTHOSTATIC / ARRHYTHMIA EPISODE":
            beat += beat_period * rng.uniform(0.62, 1.38)
        else:
            beat += beat_period * rng.uniform(0.99, 1.01)
    for beat in beats:
        ecg += 0.10 * np.exp(-((time_s - (beat - 0.16)) / 0.045) ** 2)
        ecg += -0.12 * np.exp(-((time_s - (beat - 0.025)) / 0.012) ** 2)
        ecg += 1.0 * np.exp(-((time_s - beat) / 0.014) ** 2)
        ecg += -0.23 * np.exp(-((time_s - (beat + 0.035)) / 0.018) ** 2)
        ecg += 0.28 * np.exp(-((time_s - (beat + 0.24)) / 0.08) ** 2)
    ecg += rng.normal(0, 0.012, time_s.size)
    peaks, _ = find_peaks(ecg, height=0.55, distance=max(1, int(0.25 * len(time_s) / 10)))
    peak_times = time_s[peaks]
    inst_hr = 60 / np.diff(peak_times) if len(peak_times) > 1 else np.array([rate])
    hr_times = peak_times[1:] if len(peak_times) > 1 else np.array([5.0])
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=False,
        vertical_spacing=0.16,
        subplot_titles=("SYNTHETIC LEAD-II MORPHOLOGY · 10 S", "INSTANTANEOUS HEART RATE · SYNTHETIC RR SERIES"),
    )
    fig.add_trace(go.Scatter(x=time_s, y=ecg, name="ECG (simulated)", line=dict(color=COLORS["cyan"], width=1)), row=1, col=1)
    fig.add_trace(go.Scatter(x=hr_times, y=inst_hr, name="HR", mode="lines+markers", marker=dict(size=4), line=dict(color=COLORS["amber"], width=1.5)), row=2, col=1)
    fig.update_xaxes(title_text="Time (s)", row=1, col=1)
    fig.update_xaxes(title_text="Time (s)", row=2, col=1)
    fig.update_yaxes(title_text="mV · illustrative", row=1, col=1)
    fig.update_yaxes(title_text="bpm", row=2, col=1)
    return figure_layout(fig, 480)


def make_recovery_figure(values: dict[str, float], crew: dict[str, Any]) -> go.Figure:
    minutes = np.arange(0, 31)
    phase = np.linspace(0, 5 * np.pi, len(minutes))
    rmssd = values["hrv_rmssd_ms"] + 2.5 * np.sin(phase) + np.linspace(1.5, -1.5, len(minutes))
    sdnn = values["hrv_sdnn_ms"] + 4.0 * np.cos(phase / 2) + np.linspace(2, -2, len(minutes))
    baseline = crew["baseline_hrv_rmssd_ms"]
    fig = go.Figure()
    fig.add_hrect(y0=baseline * 0.9, y1=baseline * 1.1, fillcolor=COLORS["cyan"], opacity=0.08, line_width=0, annotation_text="RMSSD baseline band", annotation_position="top left")
    fig.add_trace(go.Scatter(x=minutes, y=rmssd, name="RMSSD · ms", line=dict(color=COLORS["cyan"], width=2)))
    fig.add_trace(go.Scatter(x=minutes, y=sdnn, name="SDNN · ms", line=dict(color=COLORS["amber"], width=1.6, dash="dot")))
    fig.update_layout(title="AUTONOMIC NERVOUS SYSTEM RECOVERY · SIMULATED", xaxis_title="Window (min)", yaxis_title="Variability (ms)")
    return figure_layout(fig)


def make_fusion_figure(scenario: str, values: dict[str, float]) -> go.Figure:
    rng = np.random.default_rng(53)
    n = 45
    co2 = np.linspace(1.05, values["co2_mmhg"], n) + rng.normal(0, 0.06, n)
    radiation = np.linspace(0.006, values["rad_msv_hr"], n) + rng.normal(0, max(0.001, values["rad_msv_hr"] * 0.015), n)
    hr = np.linspace(66, values["hr_bpm"], n) + rng.normal(0, 2, n)
    spo2 = np.linspace(98, values["spo2_pct"], n) + rng.normal(0, 0.35, n)
    df = pd.DataFrame({"co2": co2, "radiation": radiation, "hr": hr, "spo2": spo2})
    df["event"] = "Nominal window"
    if scenario != "NOMINAL ORBIT":
        df.loc[n - 10 :, "event"] = "Injected event"
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df["co2"],
        y=df["spo2"],
        mode="markers",
        name="ppCO2 vs SpO2",
        marker=dict(size=8, color=df["hr"], colorscale=[[0, COLORS["cyan"]], [1, COLORS["red"]]], colorbar=dict(title="HR bpm"), line=dict(color=COLORS["border"], width=1)),
        text=[f"HR {h:.0f} bpm · rad {r:.3f} mSv/hr · {e}" for h, r, e in zip(df["hr"], df["radiation"], df["event"])],
        hovertemplate="ppCO2 %{x:.2f} mmHg<br>SpO2 %{y:.1f}%<br>%{text}<extra></extra>",
    ))
    fig.update_layout(title="MULTI-MODAL FUSION · COLOR=HR, HOVER=RADIATION", xaxis_title="Cabin ppCO2 (mmHg)", yaxis_title="SpO2 (%)")
    return figure_layout(fig)


def validate_log(frame: pd.DataFrame) -> pd.DataFrame:
    missing = [column for column in REQUIRED_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}")
    result = frame[REQUIRED_COLUMNS].copy()
    result["timestamp"] = pd.to_datetime(result["timestamp"], utc=True, errors="coerce")
    if result["timestamp"].isna().any():
        raise ValueError("The timestamp column contains values that cannot be parsed as UTC timestamps.")
    numeric = REQUIRED_COLUMNS[1:]
    for column in numeric:
        result[column] = pd.to_numeric(result[column], errors="coerce")
    if result[numeric].isna().any().any():
        raise ValueError("Telemetry values must be numeric and cannot be blank.")
    if result.empty:
        raise ValueError("The uploaded telemetry log contains no data rows.")
    result = result.sort_values("timestamp").reset_index(drop=True)
    return result


def analyze_log(frame: pd.DataFrame) -> pd.DataFrame:
    result = validate_log(frame)
    z_specs = {
        "hr_bpm": (67.0, 8.0),
        "spo2_pct": (98.0, 2.0),
        "temp_c": (36.8, 0.5),
        "co2_mmhg": (1.2, 0.6),
        "rad_msv_hr": (0.008, 0.01),
        "rr_ms": (900.0, 100.0),
    }
    z_flags = []
    for column, (center, spread) in z_specs.items():
        z_flags.append(((result[column] - center) / spread).abs() >= 3)
    result["z_anomaly"] = np.logical_or.reduce([flag.to_numpy() for flag in z_flags])
    result["clinical_anomaly"] = (
        (result["hr_bpm"] > 120)
        | (result["hr_bpm"] < 45)
        | (result["spo2_pct"] < 92)
        | (result["temp_c"] > 38.5)
        | (result["co2_mmhg"] > 4.0)
        | (result["rad_msv_hr"] > 2.0)
    )
    result["anomaly"] = result["z_anomaly"] | result["clinical_anomaly"]
    return result


def make_analyzer_figure(frame: pd.DataFrame) -> go.Figure:
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Scatter(x=frame["timestamp"], y=frame["co2_mmhg"], name="ppCO2 · mmHg", line=dict(color=COLORS["amber"], width=1.7)), secondary_y=False)
    fig.add_trace(go.Scatter(x=frame["timestamp"], y=frame["spo2_pct"], name="SpO2 · %", line=dict(color=COLORS["cyan"], width=1.7)), secondary_y=True)
    anomalies = frame[frame["anomaly"]]
    if not anomalies.empty:
        fig.add_trace(go.Scatter(x=anomalies["timestamp"], y=anomalies["co2_mmhg"], name="Flagged interval", mode="markers", marker=dict(color=COLORS["red"], size=9, symbol="x")), secondary_y=False)
        for timestamp in anomalies["timestamp"]:
            fig.add_vline(x=timestamp, line_width=1, line_dash="dot", line_color=COLORS["red"], opacity=0.45)
    fig.update_layout(title="ANOMALY TIMELINE · CLINICAL THRESHOLD + ROBUST CONFIGURED Z-SCORE")
    fig.update_xaxes(title_text="UTC")
    fig.update_yaxes(title_text="ppCO2 (mmHg)", secondary_y=False, gridcolor=COLORS["border"])
    fig.update_yaxes(title_text="SpO2 (%)", secondary_y=True, gridcolor=COLORS["border"])
    return figure_layout(fig, 390)


def osdr_biomarker_options(frame: pd.DataFrame) -> dict[str, str]:
    options = {}
    for column in frame.columns:
        if "_concentration_" in column:
            biomarker, unit_name = column.split("_concentration_", maxsplit=1)
            unit = {
                "nanogram_per_milliliter": "ng/mL",
                "picogram_per_milliliter": "pg/mL",
            }.get(unit_name, unit_name.replace("_", " "))
        elif column.endswith("_percent"):
            biomarker = column.removesuffix("_percent")
            unit = "source percent"
        else:
            continue
        name = biomarker.replace("_", " ").title()
        options[f"{name} · {unit}"] = column
    return options


def make_osdr_biomarker_figure(
    frame: pd.DataFrame,
    selected_column: str,
    label: str,
) -> go.Figure:
    sample_parts = frame["Sample Name"].str.extract(
        r"^(?P<crew>C\d+)_serum_(?P<phase>L|R)(?P<offset>[+-]?\d+)$"
    )
    if sample_parts.isna().any().any():
        raise ValueError("One or more sample names do not match the expected crew/timepoint format.")
    samples = frame.copy()
    samples["crew_id"] = sample_parts["crew"]
    samples["timepoint"] = sample_parts["phase"] + sample_parts["offset"]
    samples["days_from_launch"] = pd.to_numeric(sample_parts["offset"])
    samples.loc[sample_parts["phase"] == "L", "days_from_launch"] *= -1
    samples["timepoint_order"] = samples["timepoint"].map(
        {timepoint: index for index, timepoint in enumerate(OSDR_TIMEPOINT_ORDER)}
    )
    samples = samples.sort_values("timepoint_order")
    unit = label.split(" · ", maxsplit=1)[-1]
    fig = make_subplots(
        rows=2,
        cols=1,
        vertical_spacing=0.2,
        subplot_titles=(
            "INDIVIDUAL CREW TIME COURSE · MEASURED CONCENTRATION",
            "GROUPED MEAN · PRE-FLIGHT VS POST-FLIGHT",
        ),
    )
    crew_ids = sorted(samples["crew_id"].unique())
    for crew_id in crew_ids:
        crew_samples = samples[samples["crew_id"] == crew_id]
        fig.add_trace(
            go.Scatter(
                x=crew_samples["timepoint"],
                y=crew_samples[selected_column],
                name=crew_id,
                mode="lines+markers",
                line=dict(width=2),
                marker=dict(size=7),
                hovertemplate=f"{crew_id} · %{{x}}<br>%{{y:.4g}} {unit}<extra></extra>",
                legendgroup=crew_id,
            ),
            row=1,
            col=1,
        )

    preflight_means = [
        float(samples[(samples["crew_id"] == crew_id) & (samples["timepoint"].str.startswith("L"))][selected_column].mean())
        for crew_id in crew_ids
    ]
    postflight_means = [
        float(samples[(samples["crew_id"] == crew_id) & (samples["timepoint"].str.startswith("R"))][selected_column].mean())
        for crew_id in crew_ids
    ]
    fig.add_trace(
        go.Bar(x=crew_ids, y=preflight_means, name="Pre-flight mean", marker_color=COLORS["cyan"]),
        row=2,
        col=1,
    )
    fig.add_trace(
        go.Bar(x=crew_ids, y=postflight_means, name="Post-flight mean", marker_color=COLORS["amber"]),
        row=2,
        col=1,
    )
    fig.update_layout(
        title=f"OSD-575 · {label} · C001–C004",
        barmode="group",
        legend=dict(orientation="h", yanchor="bottom", y=1.05, x=0),
    )
    fig.update_xaxes(
        title_text="Mission-relative sample timepoint",
        categoryorder="array",
        categoryarray=OSDR_TIMEPOINT_ORDER,
        row=1,
        col=1,
    )
    fig.update_yaxes(title_text=f"Concentration ({unit})", row=1, col=1)
    fig.update_xaxes(title_text="Crew ID", row=2, col=1)
    fig.update_yaxes(title_text=f"Mean concentration ({unit})", row=2, col=1)
    return figure_layout(fig, 620)


def render_osdr_dataset_view(key: str) -> None:
    dataset_paths = find_osdr_dataset_paths()
    if not dataset_paths:
        st.info("Optional OSD-575 dataset not found: data/LSDS-8*TRANSFORMED.csv")
        return
    try:
        dataset_path = dataset_paths[0]
        frame = load_osdr_dataset(str(dataset_path))
        options = osdr_biomarker_options(frame)
        if not options:
            st.error("The local panel contains no supported concentration biomarkers.")
            return
        sample_parts = frame["Sample Name"].str.extract(
            r"^(?P<crew>C\d+)_serum_(?P<phase>L|R)(?P<offset>[+-]?\d+)$"
        )
        if sample_parts.isna().any().any():
            raise ValueError("One or more sample names do not match the expected crew/timepoint format.")
        st.caption(
            f"Active embedded dataset: OSD-575 (LSDS-8 SpaceX Inspiration4 Cardiovascular Panel) · {dataset_path.name}. "
            "Values are displayed as supplied; crew-level pre/post means are descriptive only and are not clinical interpretations."
        )
        selected_label = st.selectbox(
            "CARDIOVASCULAR PROTEIN BIOMARKER",
            list(options),
            key=f"{key}_biomarker",
        )
        selected_column = options[selected_label]
        st.plotly_chart(
            make_osdr_biomarker_figure(frame, selected_column, selected_label),
            width="stretch",
            theme=None,
            key=f"{key}_biomarker_chart",
        )
        st.dataframe(frame, width="stretch", hide_index=True, key=f"{key}_table")
    except (OSError, pd.errors.ParserError, UnicodeDecodeError, ValueError) as error:
        st.error(f"Could not load or visualize the local OSDR panel: {error}")


def report_text(frame: pd.DataFrame, crew_name: str) -> str:
    flagged = frame[frame["anomaly"]]
    lines = [
        "ASTROPULSE // CHeCS-DS — FLIGHT SURGEON TRIAGE SUMMARY",
        "EDUCATIONAL SYNTHETIC/UPLOADED DATA REVIEW — NOT A MEDICAL DIAGNOSIS",
        f"Generated UTC: {datetime.now(timezone.utc).isoformat(timespec='seconds')}",
        f"Crew context: {crew_name}",
        f"Samples: {len(frame)}",
        f"Interval: {frame['timestamp'].min().isoformat()} to {frame['timestamp'].max().isoformat()}",
        f"Flagged intervals: {len(flagged)}",
        "",
        "Configured demo fusion rules: |z| >= 3 using fixed reference centers/spreads, or HR <45/>120 bpm,",
        "SpO2 <92%, temperature >38.5 C, ppCO2 >4.0 mmHg, or radiation >2.0 mSv/hr.",
        "These settings are illustrative and are not official flight rules or diagnostic criteria.",
        "",
    ]
    if flagged.empty:
        lines.append("No rows crossed the configured anomaly rules.")
    else:
        for row in flagged.itertuples(index=False):
            codes = []
            if row.hr_bpm > 120 or row.hr_bpm < 45:
                codes.append(f"HR={row.hr_bpm:.1f} bpm")
            if row.spo2_pct < 92:
                codes.append(f"SpO2={row.spo2_pct:.1f}%")
            if row.temp_c > 38.5:
                codes.append(f"Tcore={row.temp_c:.2f} C")
            if row.co2_mmhg > 4:
                codes.append(f"ppCO2={row.co2_mmhg:.2f} mmHg")
            if row.rad_msv_hr > 2:
                codes.append(f"radiation={row.rad_msv_hr:.2f} mSv/hr")
            if row.z_anomaly:
                codes.append("configured z-score excursion")
            lines.append(f"{row.timestamp.isoformat()} — " + "; ".join(codes))
    lines.extend(["", "Operational response: verify data quality and follow current approved onboard procedures."])
    return "\n".join(lines)


def build_dtn_packet(crew: dict[str, Any], values: dict[str, float], readiness: int, scenario: str, light_time: float, met: str) -> tuple[str, bytes]:
    status_items = {
        metric: metric_status(metric, float(values[metric]), crew)
        for metric in ("hr_bpm", "hrv_rmssd_ms", "spo2_pct", "temp_c", "co2_mmhg", "rad_msv_hr")
    }
    packet = {
        "v": 1,
        "mission": "ARTEMIS-IV / MARS TRANSIT",
        "met": met,
        "utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "crew": crew["id"],
        "scenario": scenario,
        "readiness": readiness,
        "vitals": {key: round(float(values[key]), 2) for key in ("hr_bpm", "hrv_rmssd_ms", "spo2_pct", "temp_c", "co2_mmhg", "rad_msv_hr")},
        "status": status_items,
        "earth_one_way_min": round(light_time, 1),
        "edge_latency_ms": 12.4,
        "queue": "DTN_EARTH",
        "synthetic": True,
    }
    text = json.dumps(packet, separators=(",", ":"), ensure_ascii=True)
    return text, zlib.compress(text.encode("utf-8"), level=9)


def render_kpi(label: str, value: float, unit: str, status: str, baseline_delta: str) -> None:
    badge_class = {"NOMINAL": "badge-nominal", "CAUTION": "badge-caution", "CRITICAL": "badge-critical"}[status]
    st.markdown(
        f"""
        <div class="kpi">
            <div class="kpi-label">{label}</div>
            <div class="kpi-value">{value:.1f}<span class="kpi-unit"> {unit}</span></div>
            <span class="badge {badge_class}">{status}</span>
            <div class="kpi-footer">{baseline_delta}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


@st.fragment(run_every="1s")
def render_top_strip(readiness: int, light_time: float) -> None:
    now = datetime.now(timezone.utc)
    met = elapsed_met(now)
    link_state = "DSN LINK · NOMINAL" if light_time < 1 else f"DTN QUEUED · {light_time:.1f} MIN"
    link_color = COLORS["cyan"] if light_time < 1 else COLORS["amber"]
    st.markdown(
        f"""
        <div class="top-strip">
            <div class="strip-cell"><div class="strip-label">MISSION</div><div class="strip-value">ARTEMIS-IV / MARS TRANSIT</div></div>
            <div class="strip-cell"><div class="strip-label">MET · DD:HH:MM:SS</div><div class="strip-value">{met}</div></div>
            <div class="strip-cell"><div class="strip-label">CREW READINESS INDEX</div><div class="strip-value">{readiness:03d}<span style="color:{COLORS["muted"]}"> / 100%</span></div></div>
            <div class="strip-cell"><div class="strip-label">EDGE INFERENCE LATENCY</div><div class="strip-value">12.4 ms <span class="status-dot">●</span></div></div>
            <div class="strip-cell"><div class="strip-label">EARTH DSN LINK STATUS</div><div class="strip-value" style="color:{link_color}">{link_state}</div></div>
        </div>
        """,
        unsafe_allow_html=True,
    )


@st.fragment(run_every="1s")
def render_exam_countdown(deadline: float, role: str) -> None:
    remaining = max(0, int(deadline - time.monotonic()))
    if remaining:
        minutes, seconds = divmod(remaining, 60)
        countdown = f"{minutes:03d}:{seconds:02d} MIN"
    else:
        countdown = "DUE · CHECK SCHEDULE"
    st.markdown(
        f'<div class="eyebrow">{role} · NEXT EXAM IN <span class="mono">{countdown}</span></div>',
        unsafe_allow_html=True,
    )


def main() -> None:
    ensure_sample_csv()
    apply_styles()
    standards = load_standards()
    crew_profiles = standards["crew_profiles"]
    demo_step = st.query_params.get("demo_step")
    demo_config = DEMO_STEPS.get(str(demo_step)) if demo_step is not None else None

    st.sidebar.markdown("## ASTROPULSE")
    st.sidebar.markdown('<div class="eyebrow">CHeCS-DS · FLIGHT MEDICAL SYSTEM</div>', unsafe_allow_html=True)
    st.sidebar.markdown("---")
    crew_names = [profile["name"] for profile in crew_profiles]
    selected_name = st.sidebar.selectbox("CREW ROSTER", crew_names, index=0)
    crew = next(profile for profile in crew_profiles if profile["name"] == selected_name)
    deadline_key = f"exam_deadline_{crew['id']}"
    if deadline_key not in st.session_state:
        st.session_state[deadline_key] = time.monotonic() + crew["next_exam_minutes"] * 60
    with st.sidebar:
        render_exam_countdown(float(st.session_state[deadline_key]), crew["role"])
    mode = st.sidebar.radio("OPERATIONAL VIEW", ["CREW AUTONOMOUS MODE", "FLIGHT SURGEON CONSOLE (MCC-H)"], index=0)
    if demo_config is not None:
        st.session_state["scenario_select"] = demo_config["scenario"]
        st.session_state["light_time_slider"] = demo_config["light_time"]
    scenario = st.sidebar.selectbox(
        "FAULT-INJECTION SIMULATOR",
        SCENARIOS,
        index=0,
        key="scenario_select",
    )
    light_time = st.sidebar.slider(
        "EARTH ONE-WAY LIGHT-TIME · MIN",
        0.0,
        22.0,
        0.0,
        0.5,
        key="light_time_slider",
    )
    st.sidebar.caption("Edge triage executes locally; Earth link is asynchronous at nonzero light-time.")
    st.sidebar.markdown("---")
    st.sidebar.markdown(
        f'<div class="eyebrow">PREFLIGHT HRV BASELINE</div><div class="mono">{crew["baseline_hrv_rmssd_ms"]} ms RMSSD</div>',
        unsafe_allow_html=True,
    )

    now = datetime.now(timezone.utc)
    met = elapsed_met(now)
    values = live_telemetry(scenario, crew)
    for metric, session_key in [
        ("hr_bpm", "manual_hr_bpm"),
        ("hrv_rmssd_ms", "manual_hrv_rmssd_ms"),
        ("spo2_pct", "manual_spo2_pct"),
        ("temp_c", "manual_temp_c"),
        ("co2_mmhg", "manual_co2_mmhg"),
        ("rad_msv_hr", "manual_rad_msv_hr"),
        ("glucose_mg_dl", "manual_glucose_mg_dl"),
    ]:
        if session_key in st.session_state:
            values[metric] = float(st.session_state[session_key])
    values["vision_change"] = bool(st.session_state.get("manual_sans_change", False))
    readiness = readiness_index(values, crew)

    render_top_strip(readiness, light_time)
    st.markdown(
        f'<div class="eyebrow">LIVE HEALTH & HABITABILITY TELEMETRY <span style="color:{COLORS["muted"]}">/ {now.strftime("%Y-%m-%d %H:%M:%S UTC")} / {scenario}</span></div>',
        unsafe_allow_html=True,
    )
    if any(item.get("configured_for_demo") for item in standards["thresholds"].values()):
        st.caption("SIMULATION / TRAINING DISPLAY — threshold bands are challenge-demo configuration, not controlled NASA flight rules or clinical advice.")

    tabs = {
        "live": "01 // LIVE BIOMETRICS & TRIAGE",
        "sop": "02 // MEDICAL SOP & SELF-MEASUREMENT",
        "analyzer": "03 // OSDR BATCH TELEMETRY ANALYZER",
        "arch": "04 // DATA CITATIONS & DTN",
    }
    tab_order = list(tabs)
    if demo_config is not None:
        selected_tab = demo_config["tab"]
        tab_order = [selected_tab] + [name for name in tab_order if name != selected_tab]
    tab_containers = st.tabs([tabs[name] for name in tab_order])
    tab_by_name = dict(zip(tab_order, tab_containers))
    tab_live = tab_by_name["live"]
    tab_sop = tab_by_name["sop"]
    tab_analyzer = tab_by_name["analyzer"]
    tab_arch = tab_by_name["arch"]

    with tab_live:
        metrics = [
            ("HEART RATE", "hr_bpm", "bpm", crew["baseline_hr_bpm"], 8.0),
            ("HRV · RMSSD", "hrv_rmssd_ms", "ms", crew["baseline_hrv_rmssd_ms"], 10.0),
            ("SpO2", "spo2_pct", "%", crew["baseline_spo2_pct"], 1.5),
            ("CORE TEMP", "temp_c", "°C", crew["baseline_temp_c"], 0.3),
            ("CABIN ppCO2", "co2_mmhg", "mmHg", 1.2, 0.6),
            ("RADIATION RATE", "rad_msv_hr", "mSv/hr", 0.008, 0.01),
        ]
        for row_index in range(2):
            columns = st.columns(3)
            row_metrics = metrics[row_index * 3 : (row_index + 1) * 3]
            for column, (label, metric, unit, baseline, spread) in zip(columns, row_metrics):
                status = metric_status(metric, values[metric], crew)
                delta = values[metric] - baseline
                z = z_score(delta, 0, spread)
                with column:
                    render_kpi(label, values[metric], unit, status, f"Δ {delta:+.1f} · z {z:+.2f} vs ref")

        flagged = [
            (metric, label, metric_status(metric, values[metric], crew))
            for label, metric, _, _, _ in metrics
            if metric_status(metric, values[metric], crew) != "NOMINAL"
        ]
        if metric_status("glucose_mg_dl", values["glucose_mg_dl"], crew) != "NOMINAL":
            flagged.append(("glucose_mg_dl", "SPOT BLOOD GLUCOSE", metric_status("glucose_mg_dl", values["glucose_mg_dl"], crew)))
        if values.get("vision_change", False):
            flagged.append(("vision_change", "AMSLER GRID · NEW VISUAL CHANGE", "CAUTION"))
        if flagged:
            severities = {"CAUTION": 1, "CRITICAL": 2}
            flagged.sort(key=lambda item: severities[item[2]], reverse=True)
            first = flagged[0]
            color = COLORS["red"] if first[2] == "CRITICAL" else COLORS["amber"]
            st.markdown(
                f'<div class="panel" style="border-left:3px solid {color}"><div class="eyebrow" style="color:{color}">AUTONOMOUS CDS · {first[2]} · {len(flagged)} FLAGGED CHANNEL(S)</div><h3 style="margin:.45rem 0">Action checklist · {first[1]}</h3><div class="subtle">Local decision support is active during Earth light-time delay. Use current approved flight rules and onboard medical procedures; this demonstrator does not issue diagnosis or treatment.</div></div>',
                unsafe_allow_html=True,
            )
            for index, (metric, label, status) in enumerate(flagged, start=1):
                with st.expander(f"{index}. {status} — {label}: {values[metric]:.2f}", expanded=index == 1):
                    for step_no, action in enumerate(critical_actions(metric, status), start=1):
                        st.markdown(f'<div class="sop-step"><span class="sop-number">{step_no:02d}</span><span>{action}</span></div>', unsafe_allow_html=True)
                    st.caption("Crew checklist language is a simulation aid; follow the current controlled onboard procedure.")
        else:
            st.markdown(
                f'<div class="panel"><div class="eyebrow" style="color:{COLORS["cyan"]}">AUTONOMOUS CDS · NOMINAL</div><div class="subtle">No configured alert threshold crossed. Continue scheduled monitoring; this status does not establish medical fitness for duty.</div></div>',
                unsafe_allow_html=True,
            )

        if mode == "FLIGHT SURGEON CONSOLE (MCC-H)":
            st.markdown("#### FLIGHT SURGEON ANALYTICS")
            d1, d2, d3, d4 = st.columns(4)
            d1.metric("RMSSD", f"{values['hrv_rmssd_ms']:.1f} ms", f"{(values['hrv_rmssd_ms'] / crew['baseline_hrv_rmssd_ms'] - 1) * 100:+.1f}% vs preflight")
            d2.metric("SDNN", f"{values['hrv_sdnn_ms']:.1f} ms")
            d3.metric("pNN50", f"{max(0, min(100, values['hrv_rmssd_ms'] * 0.72)):.1f}%")
            stress_index = 1000 / max(values["hrv_rmssd_ms"], 1)
            d4.metric("STRESS INDEX · RELATIVE", f"{stress_index:.1f}")
            crew_rows = []
            for profile in crew_profiles:
                member_values = live_telemetry(scenario, profile)
                crew_rows.append({
                    "CREW": profile["name"],
                    "HR bpm": round(member_values["hr_bpm"], 1),
                    "RMSSD ms": round(member_values["hrv_rmssd_ms"], 1),
                    "SpO2 %": round(member_values["spo2_pct"], 1),
                    "Tcore °C": round(member_values["temp_c"], 2),
                    "READINESS": readiness_index(member_values, profile),
                    "NEXT EXAM min": profile["next_exam_minutes"],
                })
            st.dataframe(pd.DataFrame(crew_rows), width="stretch", hide_index=True)
            with st.expander("RAW TELEMETRY PACKET · CURRENT CREW"):
                packet_text, compressed = build_dtn_packet(crew, values, readiness, scenario, light_time, met)
                st.code(packet_text, language="json")
                st.caption(f"Compact JSON {len(packet_text.encode('utf-8'))} B · zlib payload {len(compressed)} B")

        left, right = st.columns(2)
        with left:
            st.plotly_chart(make_ecg_figure(values, scenario), width="stretch", theme=None)
        with right:
            st.plotly_chart(make_recovery_figure(values, crew), width="stretch", theme=None)
        st.plotly_chart(make_fusion_figure(scenario, values), width="stretch", theme=None)

    with tab_sop:
        sop_items = standards["biomarker_sops"]
        selected_sop_name = st.selectbox("BIOMARKER / PROCEDURE", [item["name"] for item in sop_items])
        sop = next(item for item in sop_items if item["name"] == selected_sop_name)
        limits = sop["limits"]
        c1, c2 = st.columns(2)
        with c1:
            st.markdown('<div class="panel"><div class="eyebrow">01 · SENSOR HARDWARE</div><p class="subtle">' + sop["sensor_hardware"] + "</p></div>", unsafe_allow_html=True)
            st.markdown('<div class="panel"><div class="eyebrow">02 · NON-DOCTOR MEASUREMENT PROTOCOL</div>', unsafe_allow_html=True)
            for index, step in enumerate(sop["protocol"], start=1):
                st.markdown(f'<div class="sop-step"><span class="sop-number">{index:02d}</span><span>{step}</span></div>', unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)
        with c2:
            st.markdown('<div class="panel"><div class="eyebrow">03 · WHY IT MATTERS IN DEEP SPACE</div><p class="subtle">' + sop["pathophysiology"] + "</p></div>", unsafe_allow_html=True)
            st.markdown('<div class="panel"><div class="eyebrow">NASA-STD-3001 / DEMO REFERENCE BAND</div><p class="mono">' + limits + '</p><p class="subtle">These challenge-demo bands are not confirmed requirement text from NASA-STD-3001 and are not clinical directives.</p></div>', unsafe_allow_html=True)
            st.markdown('<div class="panel"><div class="eyebrow">04 · SCHEDULED CHECK CADENCE</div><p class="subtle">' + sop["cadence"] + "</p></div>", unsafe_allow_html=True)

        st.markdown("#### LOG MANUAL CREW READING")
        input_options = {
            "Heart rate (bpm)": ("manual_hr_bpm", "hr_bpm", 30.0, 220.0, values["hr_bpm"]),
            "HRV RMSSD (ms)": ("manual_hrv_rmssd_ms", "hrv_rmssd_ms", 1.0, 200.0, values["hrv_rmssd_ms"]),
            "SpO2 (%)": ("manual_spo2_pct", "spo2_pct", 50.0, 100.0, values["spo2_pct"]),
            "Core temperature (°C)": ("manual_temp_c", "temp_c", 30.0, 43.0, values["temp_c"]),
            "Cabin ppCO2 (mmHg)": ("manual_co2_mmhg", "co2_mmhg", 0.0, 20.0, values["co2_mmhg"]),
            "Radiation dose rate (mSv/hr)": ("manual_rad_msv_hr", "rad_msv_hr", 0.0, 100.0, values["rad_msv_hr"]),
            "Spot blood glucose (mg/dL)": ("manual_glucose_mg_dl", "glucose_mg_dl", 0.0, 600.0, values["glucose_mg_dl"]),
        }
        with st.form("manual_reading_form"):
            chosen_reading = st.selectbox("READING CHANNEL", [*input_options, "SANS Amsler grid check"])
            if chosen_reading == "SANS Amsler grid check":
                vision_result = st.selectbox("SCREENING OBSERVATION", ["No new change", "New visual change reported"])
                submitted_value = None
            else:
                session_key, metric_key, minimum, maximum, current_value = input_options[chosen_reading]
                submitted_value = st.number_input("MEASURED VALUE", min_value=minimum, max_value=maximum, value=float(current_value), step=0.1, format="%.2f")
            submitted_note = st.text_input("CONTEXT / DEVICE STATUS (OPTIONAL)", max_chars=120)
            submitted = st.form_submit_button("LOG READING & RECOMPUTE READINESS", type="primary")
        if submitted:
            if chosen_reading == "SANS Amsler grid check":
                st.session_state["manual_sans_change"] = vision_result == "New visual change reported"
            else:
                st.session_state[session_key] = float(submitted_value)
            st.session_state["manual_reading_note"] = submitted_note
            st.rerun()
        if "manual_reading_note" in st.session_state and st.session_state["manual_reading_note"]:
            st.caption("Last note: " + st.session_state["manual_reading_note"])
        st.info(
            "Cadence: " + sop["cadence"],
            icon=None,
        )

    with tab_analyzer:
        st.markdown("#### NASA OSDR BATCH TELEMETRY ANALYZER")
        st.caption("This tool demonstrates an analysis pipeline; the bundled calibrated log is synthetic, not a NASA OSDR download or participant dataset.")
        osdr_paths = find_osdr_dataset_paths()
        if osdr_paths and st.toggle(
            "Real NASA OSDR Inspiration4 Clinical Biomarkers (OSD-575)",
            value=True,
            key="tab3_show_osdr",
        ):
            st.markdown("##### REAL NASA OSDR INSPIRATION4 CLINICAL DATASET · OSD-575")
            render_osdr_dataset_view("tab3_osdr")
            st.markdown("---")
            st.markdown("##### SYNTHETIC / UPLOADED MISSION TELEMETRY")
        load_col, upload_col = st.columns([1, 2])
        with load_col:
            load_sample = st.button("LOAD SYNTHETIC CALIBRATED SPACEFLIGHT LOG", type="primary", width="stretch")
        with upload_col:
            uploaded_file = st.file_uploader("Upload CSV · timestamp, hr_bpm, rr_ms, spo2_pct, temp_c, co2_mmhg, rad_msv_hr", type=["csv"])
        source_frame = None
        if load_sample:
            source_frame = pd.read_csv(SAMPLE_PATH)
            st.session_state["analyzer_source_name"] = "Bundled synthetic demo telemetry"
        elif uploaded_file is not None:
            try:
                source_frame = load_csv_bytes(uploaded_file.getvalue())
                st.session_state["analyzer_source_name"] = uploaded_file.name
            except Exception as error:
                st.error(f"Could not read uploaded CSV: {error}")
        elif "analyzer_result" in st.session_state:
            source_frame = st.session_state["analyzer_result"]

        if source_frame is not None:
            try:
                analyzed = analyze_log(source_frame)
                st.session_state["analyzer_result"] = analyzed
                flagged_count = int(analyzed["anomaly"].sum())
                a1, a2, a3, a4 = st.columns(4)
                a1.metric("SAMPLES", f"{len(analyzed):,}")
                a2.metric("FLAGGED INTERVALS", f"{flagged_count:,}", delta="Review" if flagged_count else "No flags")
                a3.metric("PEAK HR", f"{analyzed['hr_bpm'].max():.1f} bpm")
                a4.metric("PEAK ppCO2", f"{analyzed['co2_mmhg'].max():.2f} mmHg")
                st.plotly_chart(make_analyzer_figure(analyzed), width="stretch", theme=None)
                st.markdown("##### FLAGGED ROWS")
                flagged = analyzed[analyzed["anomaly"]]
                if flagged.empty:
                    st.success("No intervals crossed the configured demo rules.")
                else:
                    st.dataframe(flagged, width="stretch", hide_index=True)
                report = report_text(analyzed, crew["name"])
                st.download_button(
                    "DOWNLOAD FLIGHT SURGEON TRIAGE SUMMARY · .TXT",
                    data=report,
                    file_name="astropulse_flight_surgeon_triage_summary.txt",
                    mime="text/plain",
                )
                st.caption(f"Source: {st.session_state.get('analyzer_source_name', 'CSV')} · rules are configurable demo logic, not validated clinical detection.")
            except (ValueError, KeyError) as error:
                st.error(f"Telemetry validation failed: {error}")
        else:
            st.markdown(
                '<div class="panel"><div class="eyebrow">CSV CONTRACT</div><div class="mono">timestamp, hr_bpm, rr_ms, spo2_pct, temp_c, co2_mmhg, rad_msv_hr</div><div class="subtle">Timestamp values must be parseable as UTC; measurement columns must be numeric.</div></div>',
                unsafe_allow_html=True,
            )

    with tab_arch:
        st.markdown("#### NASA DATA CITATIONS & DTN ARCHITECTURE")
        st.markdown("**Source provenance**")
        for source in standards["sources"]:
            st.markdown(f"- [{source['name']}]({source['url']}) — {source['citation']} {source['source_role']}")
        st.warning(
            "Provenance boundary: the transformed OSDR OSD-575 panel is loaded locally when present; generated live values and the bundled telemetry analyzer sample remain synthetic. Numeric alert bands are challenge-demo configuration, not verified NASA-STD-3001 requirement text, approved CHeCS procedure, or medical advice.",
            icon="⚠️",
        )
        osdr_paths = find_osdr_dataset_paths()
        if osdr_paths:
            st.markdown(
                "**ACTIVE EMBEDDED DATASET**  \n"
                "OSD-575 (LSDS-8 SpaceX Inspiration4 Cardiovascular Panel)  \n"
                f"`{osdr_paths[0].name}` · {len(osdr_paths)} matching transformed panel file(s) detected."
            )
        architecture_view_options = ["DTN ARCHITECTURE"]
        if osdr_paths:
            architecture_view_options.append(OSDR_DATASET_LABEL)
        if demo_config is not None and demo_config["tab"] == "arch" and osdr_paths:
            st.session_state["tab4_dataset_view"] = OSDR_DATASET_LABEL
        architecture_view = st.selectbox(
            "DATA VIEW",
            architecture_view_options,
            index=(
                architecture_view_options.index(OSDR_DATASET_LABEL)
                if demo_config is not None
                and demo_config["tab"] == "arch"
                and osdr_paths
                else 0
            ),
            key="tab4_dataset_view",
        )
        if architecture_view == OSDR_DATASET_LABEL:
            st.markdown("##### REAL NASA OSDR INSPIRATION4 CLINICAL DATASET · OSD-575")
            render_osdr_dataset_view("tab4_osdr")
            st.markdown("---")
        st.markdown("**Delay-tolerant telemetry queue**")
        packet_text, compressed_payload = build_dtn_packet(crew, values, readiness, scenario, light_time, met)
        raw_size = len(packet_text.encode("utf-8"))
        compressed_size = len(compressed_payload)
        q1, q2, q3 = st.columns(3)
        q1.metric("COMPACT JSON", f"{raw_size} B", delta="under 1.5 KB" if raw_size < 1536 else "over 1.5 KB")
        q2.metric("ZLIB PAYLOAD", f"{compressed_size} B", delta="under 1.5 KB" if compressed_size < 1536 else "over 1.5 KB")
        q3.metric("EARTH LIGHT-TIME", f"{light_time:.1f} min", delta="Local edge triage active")
        st.caption("Packet includes only the selected crew's synthetic summary; queued transport is represented locally and no network transmission occurs.")
        st.code(packet_text, language="json")
        st.download_button("DOWNLOAD CURRENT DTN JSON", data=packet_text, file_name="astropulse_dtn_packet.json", mime="application/json")
        st.markdown("**Onboard architecture**")
        st.markdown(
            """
            1. **Acquire:** approved sensors and data-quality metadata enter the onboard health-data interface.
            2. **Edge triage:** local threshold and trend logic remains available without Earth connectivity; this demo's rules are illustrative.
            3. **Queue:** a compact, timestamped JSON summary is queued for delay-tolerant forwarding; raw high-rate waveforms are not placed in this packet.
            4. **Review:** Earth-based flight surgeons review context, source quality, and mission procedure; a dashboard indicator is not an autonomous diagnosis.
            """
        )

    st.markdown("---")
    st.markdown(
        '<div class="eyebrow">ASTROPULSE // CHeCS-DS · SPACE APPS CHALLENGE DEMONSTRATOR · SYNTHETIC TELEMETRY · NOT FOR CLINICAL OR FLIGHT OPERATIONS</div>',
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
