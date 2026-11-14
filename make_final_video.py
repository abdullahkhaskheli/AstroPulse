# Install dependencies: pip install playwright edge-tts moviepy
# Install the Chromium browser once: playwright install chromium

from __future__ import annotations

import asyncio
import re
import subprocess
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path

import edge_tts
import imageio_ffmpeg
from playwright.sync_api import Browser, Error as PlaywrightError, sync_playwright


ROOT = Path(__file__).resolve().parent
APP_URL = "http://localhost:8501"
VIEWPORT = {"width": 1920, "height": 1080}
VIDEO_SECONDS = 28.0
VOICE = "en-US-AndrewNeural"
VOICEOVER_TEXT = (
    "Deep-space astronauts on Artemis and Mars missions aren't medical doctors, "
    "and face up to forty-minute communication delays with Earth. Meet AstroPulse—"
    "an offline-first autonomous medical console. When a cabin CO2 scrubber fails "
    "fourteen light-minutes away, AstroPulse fuses cabin sensors with vitals in "
    "twelve milliseconds offline, flagging critical anomalies and guiding "
    "non-doctor crews with step-by-step survival protocols. Built on "
    "NASA-STD-3001 limits and real SpaceX Inspiration4 data from NASA OSDR."
)
READINESS_SELECTOR = "text=CREW READINESS"
STAGES = [
    ("1", 7.0, "live"),
    ("2", 12.0, "live"),
    ("3", 5.0, "sop"),
    ("4", 4.0, "arch"),
]


def hold_until(started_at: float, offset: float) -> None:
    remaining = started_at + offset - time.monotonic()
    if remaining > 0:
        time.sleep(remaining)


def smooth_scroll_to(main, element, viewport_y: float) -> None:
    box = element.bounding_box()
    if box is None:
        raise RuntimeError("Could not locate a visible demo element to scroll to.")
    current_scroll = main.evaluate("(node) => node.scrollTop")
    target_scroll = max(0.0, current_scroll + box["y"] - viewport_y)
    main.evaluate(
        "(node, top) => node.scrollTo({top, behavior: 'smooth'})",
        target_scroll,
    )


def prime_plotly_charts(main, charts, chart_count: int) -> None:
    for index in range(chart_count):
        chart = charts.nth(index)
        smooth_scroll_to(main, chart, 140)
        chart.locator(".scatterlayer .trace").first.wait_for(state="visible")
    main.evaluate("(node) => node.scrollTo({top: 0, behavior: 'instant'})")
    time.sleep(0.3)


async def synthesize_voiceover(output_path: Path) -> None:
    communicate = edge_tts.Communicate(
        text=VOICEOVER_TEXT,
        voice=VOICE,
        rate="+0%",
    )
    await communicate.save(str(output_path))


def ffmpeg_executable() -> str:
    return imageio_ffmpeg.get_ffmpeg_exe()


def media_duration(ffmpeg: str, media_path: Path) -> float:
    probe = subprocess.run(
        [ffmpeg, "-hide_banner", "-i", str(media_path)],
        capture_output=True,
        text=True,
        check=False,
    )
    match = re.search(r"Duration:\s*(\d+):(\d+):(\d+(?:\.\d+)?)", probe.stderr)
    if match is None:
        raise RuntimeError(f"Could not read media duration for {media_path}.")
    hours, minutes, seconds = match.groups()
    return int(hours) * 3600 + int(minutes) * 60 + float(seconds)


def media_dimensions(ffmpeg: str, media_path: Path) -> tuple[int, int]:
    probe = subprocess.run(
        [ffmpeg, "-hide_banner", "-i", str(media_path)],
        capture_output=True,
        text=True,
        check=False,
    )
    match = re.search(r"\b(\d{2,5})x(\d{2,5})\b", probe.stderr)
    if match is None:
        raise RuntimeError(f"Could not read video dimensions for {media_path}.")
    return int(match.group(1)), int(match.group(2))


def decoded_video_duration(ffmpeg: str, media_path: Path) -> float:
    probe = subprocess.run(
        [
            ffmpeg,
            "-hide_banner",
            "-i",
            str(media_path),
            "-map",
            "0:v:0",
            "-an",
            "-f",
            "null",
            "NUL",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    matches = re.findall(
        r"time=(\d+):(\d+):(\d+(?:\.\d+)?)",
        probe.stderr,
    )
    if not matches:
        raise RuntimeError(f"Could not decode the video stream in {media_path}.")
    hours, minutes, seconds = matches[-1]
    return int(hours) * 3600 + int(minutes) * 60 + float(seconds)


def check_server() -> None:
    try:
        with urllib.request.urlopen(f"{APP_URL}/_stcore/health", timeout=5) as response:
            if response.status != 200:
                raise RuntimeError(f"Streamlit health check returned HTTP {response.status}.")
    except (urllib.error.URLError, TimeoutError) as error:
        raise RuntimeError(
            f"ASTROPULSE is not responding at {APP_URL}. Start it first with "
            "`streamlit run app.py`."
        ) from error


def record_segment(
    browser: Browser,
    video_dir: Path,
    step: str,
    duration: float,
) -> Path:
    context = browser.new_context(
        viewport=VIEWPORT,
        record_video_dir=str(video_dir),
        record_video_size=VIEWPORT,
        reduced_motion="reduce",
    )
    page = context.new_page()
    page.set_default_timeout(60_000)
    page.set_default_navigation_timeout(60_000)
    page.add_init_script(
        """
        (() => {
          document.documentElement.style.backgroundColor = "#06090F";
          document.documentElement.style.colorScheme = "dark";
          const style = document.createElement("style");
          style.textContent =
            "html,body{background:#06090F!important;color:#F1F5F9!important}";
          document.documentElement.appendChild(style);
        })();
        """
    )
    video = page.video
    if video is None:
        context.close()
        raise RuntimeError("Playwright did not initialize video recording.")
    video_path: Path | None = None
    try:
        page.goto(f"{APP_URL}/?demo_step={step}", wait_until="domcontentloaded")
        page.wait_for_selector(READINESS_SELECTOR)
        page.get_by_text("ASTROPULSE // CHeCS-DS", exact=False).last.wait_for(
            state="visible"
        )
        expected_tab = {
            "1": "01 // LIVE BIOMETRICS & TRIAGE",
            "2": "01 // LIVE BIOMETRICS & TRIAGE",
            "3": "02 // MEDICAL SOP & SELF-MEASUREMENT",
            "4": "04 // DATA CITATIONS & DTN",
        }[step]
        selected_tab = page.get_by_role("tab", name=expected_tab, exact=True)
        selected_tab.wait_for(state="visible")
        if selected_tab.get_attribute("aria-selected") != "true":
            raise RuntimeError(f"Demo step {step} did not activate its expected tab.")

        if step in {"1", "2"}:
            fault_selector = page.get_by_role(
                "combobox",
                name="FAULT-INJECTION SIMULATOR",
                exact=True,
            )
            fault_selector.wait_for(state="visible")
            expected_scenario = (
                "NOMINAL ORBIT" if step == "1" else "CDRA CO2 SCRUBBER DEGRADATION"
            )
            if fault_selector.input_value() != expected_scenario:
                raise RuntimeError(
                    f"Demo step {step} did not select {expected_scenario}."
                )
            main = page.locator('main[data-testid="stMain"]')
            main.evaluate(
                "(node) => node.scrollTo({top: node.scrollHeight, behavior: 'instant'})"
            )
            page.wait_for_function(
                "document.querySelectorAll('[data-testid=\"stPlotlyChart\"]').length >= 3",
                timeout=60_000,
            )
            live_charts = page.locator('[data-testid="stPlotlyChart"]')
            if live_charts.count() < 3:
                raise RuntimeError("The live telemetry charts did not render.")
            prime_plotly_charts(main, live_charts, 3)

        if step == "2":
            alert = page.get_by_text("AUTONOMOUS CDS", exact=False)
            alert.wait_for(state="visible")
            if "CRITICAL" not in alert.inner_text():
                raise RuntimeError(
                    "Demo step 2 did not produce the expected critical alert."
                )
        elif step == "3":
            page.get_by_text(
                "NON-DOCTOR MEASUREMENT PROTOCOL",
                exact=False,
            ).wait_for(state="visible")
        elif step == "4":
            architecture_panel = page.get_by_role("tabpanel").filter(
                has_text="COMPACT JSON"
            )
            dataset_title = architecture_panel.get_by_text(
                "REAL NASA OSDR INSPIRATION4 CLINICAL DATASET",
                exact=False,
            )
            architecture_panel.get_by_text(
                "COMPACT JSON",
                exact=True,
            ).wait_for(state="visible")
            main = page.locator('main[data-testid="stMain"]')
            smooth_scroll_to(main, dataset_title, 160)
            architecture_panel.locator('[data-testid="stDataFrame"]').wait_for(
                state="visible"
            )
            architecture_panel.locator(
                '[data-testid="stPlotlyChart"] .scatterlayer .trace'
            ).first.wait_for(state="visible")

        segment_started = time.monotonic()
        if step == "2":
            hold_until(segment_started, 4)
            smooth_scroll_to(main, live_charts.nth(0), 140)
            hold_until(segment_started, 8)
            smooth_scroll_to(main, live_charts.nth(2), 140)
        elif step == "4":
            hold_until(segment_started, 2)
            smooth_scroll_to(
                main,
                architecture_panel.get_by_text("COMPACT JSON", exact=True),
                200,
            )
        hold_until(segment_started, duration)
    except PlaywrightError as error:
        raise RuntimeError(
            f"ASTROPULSE demo step {step} did not render as expected. "
            f"Details: {error}"
        ) from error
    finally:
        context.close()
        video_path = Path(video.path())

    if video_path is None or not video_path.is_file():
        raise RuntimeError(f"Playwright did not save demo step {step}.")
    return video_path


def record_segments(video_dir: Path) -> list[tuple[Path, float]]:
    segments: list[tuple[Path, float]] = []
    ffmpeg = ffmpeg_executable()
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True)
        try:
            for step, duration, _ in STAGES:
                segment_path = record_segment(
                    browser,
                    video_dir,
                    step,
                    duration,
                )
                recorded_duration = media_duration(ffmpeg, segment_path)
                if recorded_duration + 0.1 < duration:
                    raise RuntimeError(
                        f"demo_step={step} produced a {recorded_duration:.2f}s "
                        f"recording; at least {duration:.2f}s is required."
                    )
                segments.append((segment_path, duration))
                print(
                    f"Recorded demo_step={step} ({duration:g}s; "
                    f"{recorded_duration:.2f}s including load)."
                )
        finally:
            browser.close()
    return segments


def assemble_raw_video(
    ffmpeg: str,
    segments: list[tuple[Path, float]],
    output: Path,
) -> None:
    if sum(duration for _, duration in segments) != VIDEO_SECONDS:
        raise RuntimeError("Demo segment lengths must add up to exactly 28 seconds.")
    filters: list[str] = []
    inputs: list[str] = []
    for index, (segment_path, duration) in enumerate(segments):
        recorded_duration = media_duration(ffmpeg, segment_path)
        trim_start = max(0.0, recorded_duration - duration)
        inputs.extend(["-i", str(segment_path)])
        filters.append(
            f"[{index}:v]trim=start={trim_start:.6f}:duration={duration:.6f},"
            "setpts=PTS-STARTPTS,"
            "scale=1920:1080:force_original_aspect_ratio=decrease,"
            "pad=1920:1080:(ow-iw)/2:(oh-ih)/2,setsar=1,fps=30"
            f"[v{index}]"
        )
    joined_streams = "".join(f"[v{index}]" for index in range(len(segments)))
    filters.append(f"{joined_streams}concat=n={len(segments)}:v=1:a=0[outv]")
    subprocess.run(
        [
            ffmpeg,
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            *inputs,
            "-filter_complex",
            ";".join(filters),
            "-map",
            "[outv]",
            "-t",
            "28",
            "-an",
            "-c:v",
            "libvpx-vp9",
            "-deadline",
            "realtime",
            "-cpu-used",
            "5",
            "-crf",
            "32",
            "-b:v",
            "0",
            "-pix_fmt",
            "yuv420p",
            str(output),
        ],
        check=True,
    )
    raw_duration = decoded_video_duration(ffmpeg, output)
    if abs(raw_duration - VIDEO_SECONDS) > 0.1:
        raise RuntimeError(
            f"Raw video stream is {raw_duration:.2f}s instead of 28 seconds."
        )


def mux_video_with_voiceover(
    ffmpeg: str,
    raw_video: Path,
    voiceover: Path,
    output: Path,
) -> None:
    subprocess.run(
        [
            ffmpeg,
            "-y",
            "-i",
            str(raw_video),
            "-i",
            str(voiceover),
            "-c:v",
            "libx264",
            "-c:a",
            "aac",
            "-ar",
            "44100",
            "-shortest",
            str(output),
        ],
        check=True,
    )


def main() -> None:
    check_server()
    ffmpeg = ffmpeg_executable()
    voiceover = ROOT / "voiceover.mp3"
    raw_video = ROOT / "raw_video.webm"
    output_video = ROOT / "AstroPulse_Final_Demo.mp4"
    asyncio.run(synthesize_voiceover(voiceover))
    audio_duration = media_duration(ffmpeg, voiceover)
    with tempfile.TemporaryDirectory(prefix="astropulse-browser-") as video_dir:
        segments = record_segments(Path(video_dir))
        assemble_raw_video(
            ffmpeg,
            segments,
            raw_video,
        )
    raw_duration = decoded_video_duration(ffmpeg, raw_video)
    mux_video_with_voiceover(ffmpeg, raw_video, voiceover, output_video)
    final_duration = media_duration(ffmpeg, output_video)
    if final_duration >= 29.5:
        raise RuntimeError(
            f"Final MP4 is {final_duration:.2f}s, not strictly under 29.5 seconds."
        )
    video_stream_duration = decoded_video_duration(ffmpeg, output_video)
    if abs(video_stream_duration - VIDEO_SECONDS) > 0.1:
        raise RuntimeError(
            f"Final MP4 video stream is {video_stream_duration:.2f}s instead of "
            "28 seconds."
        )
    dimensions = media_dimensions(ffmpeg, output_video)
    if dimensions != (VIEWPORT["width"], VIEWPORT["height"]):
        raise RuntimeError(
            f"Final MP4 dimensions are {dimensions[0]}x{dimensions[1]}, not 1920x1080."
        )
    if audio_duration > raw_duration:
        print(
            f"Warning: voiceover.mp3 is {audio_duration:.2f}s; -shortest ends the "
            f"final mix at {raw_duration:.2f}s, so the final "
            f"{audio_duration - raw_duration:.2f}s of narration is not in the MP4."
        )
    print(f"Voiceover saved: {voiceover} ({audio_duration:.2f}s at 1.0x)")
    print(f"Raw video saved: {raw_video} ({raw_duration:.2f}s)")
    print(
        f"Final video saved: {output_video} "
        f"({VIEWPORT['width']}x{VIEWPORT['height']}, {final_duration:.2f}s)"
    )


if __name__ == "__main__":
    main()
