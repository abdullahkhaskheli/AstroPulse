# Install the Python package: pip install playwright
# Install the Chromium browser: playwright install chromium

from pathlib import Path
import time

from playwright.sync_api import Error as PlaywrightError, sync_playwright


APP_URL = "http://localhost:8501"
VIDEO_DIR = Path(__file__).resolve().parent / "demo_video"
VIEWPORT = {"width": 1920, "height": 1080}
LIVE_TAB = "01 // LIVE BIOMETRICS & TRIAGE"
SOP_TAB = "02 // MEDICAL SOP & SELF-MEASUREMENT"
ANALYZER_TAB = "03 // OSDR BATCH TELEMETRY ANALYZER"
CITATIONS_TAB = "04 // DATA CITATIONS & DTN"
LIGHT_TIME_LABEL = "EARTH ONE-WAY LIGHT-TIME · MIN"
FAULT_LABEL = "FAULT-INJECTION SIMULATOR"
CDRA_FAULT = "CDRA CO2 SCRUBBER DEGRADATION"


def hold_until(started_at: float, offset_seconds: float) -> None:
    """Wait for an absolute point in the demo timeline."""
    remaining = started_at + offset_seconds - time.monotonic()
    if remaining > 0:
        time.sleep(remaining)


def set_light_time_to_14(page) -> None:
    slider = page.get_by_role("slider", name=LIGHT_TIME_LABEL, exact=False)
    slider.focus()
    slider.press("Home")
    for _ in range(28):  # 28 half-minute steps from 0.0 to 14.0.
        slider.press("ArrowRight")

    value = slider.get_attribute("aria-valuenow")
    if value is None or float(value) != 14.0:
        raise RuntimeError(
            "Could not set the Earth one-way light-time slider to 14 minutes."
        )


def main() -> None:
    VIDEO_DIR.mkdir(parents=True, exist_ok=True)
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=False)
        context = browser.new_context(
            viewport=VIEWPORT,
            record_video_dir=str(VIDEO_DIR),
            record_video_size=VIEWPORT,
            reduced_motion="reduce",
        )
        page = context.new_page()
        page.set_default_timeout(5_000)
        video = page.video
        if video is None:
            context.close()
            browser.close()
            raise RuntimeError("Playwright did not initialize video recording.")

        try:
            try:
                page.goto(
                    APP_URL,
                    wait_until="domcontentloaded",
                    timeout=30_000,
                )
                page.wait_for_selector("text=CREW READINESS", timeout=30_000)
                time.sleep(2)
                live_tab = page.get_by_role("tab", name=LIVE_TAB, exact=True)
                live_tab.wait_for(state="visible", timeout=10_000)
                page.get_by_role(
                    "slider", name=LIGHT_TIME_LABEL, exact=False
                ).wait_for(state="visible", timeout=10_000)
                page.get_by_role(
                    "combobox", name=FAULT_LABEL, exact=True
                ).wait_for(state="visible", timeout=10_000)
                fault_selector = page.get_by_role(
                    "combobox", name=FAULT_LABEL, exact=True
                )
                if "NOMINAL ORBIT" not in fault_selector.inner_text():
                    raise RuntimeError(
                        "The simulator did not start in NOMINAL ORBIT."
                    )
                if live_tab.get_attribute("aria-selected") != "true":
                    raise RuntimeError(
                        "The dashboard did not open on the live biometrics tab."
                    )
            except PlaywrightError as error:
                raise RuntimeError(
                    f"ASTROPULSE at {APP_URL} did not render within the opening "
                    "30-second dashboard wait. Start Streamlit on port 8501 and try "
                    f"again. Details: {error}"
                ) from error

            # The requested 28-second choreography starts after the dashboard,
            # live charts, and two-second settling pause are complete.
            started_at = time.monotonic()
            hold_until(started_at, 6)

            set_light_time_to_14(page)
            fault_selector = page.get_by_role(
                "combobox", name=FAULT_LABEL, exact=True
            )
            fault_selector.click()
            page.get_by_role("option", name=CDRA_FAULT, exact=True).click()
            page.get_by_text("AUTONOMOUS CDS · CRITICAL", exact=False).wait_for(
                state="visible", timeout=3_000
            )

            # Hold on the alert for four seconds, then scroll down to the charts.
            time.sleep(4)
            page.mouse.wheel(0, 900)
            hold_until(started_at, 18)

            page.evaluate("window.scrollTo(0, 0)")
            page.get_by_role("tab", name=SOP_TAB, exact=True).click()
            page.get_by_text("NON-DOCTOR MEASUREMENT PROTOCOL", exact=False).wait_for(
                state="visible", timeout=3_000
            )
            hold_until(started_at, 23)

            page.get_by_role("tab", name=ANALYZER_TAB, exact=True).click()
            hold_until(started_at, 25.5)
            page.get_by_role("tab", name=CITATIONS_TAB, exact=True).click()
            hold_until(started_at, 28)
        finally:
            # Closing the context finalizes and flushes its recorded video.
            context.close()
            video_path = video.path()
            browser.close()
        print(f"Demo video saved: {video_path}")


if __name__ == "__main__":
    main()
