"""Browser-level smoke test used by CI against the real web server."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

from playwright.sync_api import sync_playwright


def main() -> None:
    base_url = os.getenv("BASE_URL", "http://127.0.0.1:7860")
    dates = [f"{year}-{month:02d}-28" for year in range(2018, 2023) for month in range(1, 13)]
    prices = ["Date,SPY,IEF,GLD"]
    macro = ["Date,growth,inflation"]
    for index, date in enumerate(dates):
        prices.append(f"{date},{100 + index * 1.1:.2f},{80 + index * 0.35:.2f},{60 + index * 0.25:.2f}")
        macro.append(f"{date},{2.0 if index % 8 < 4 else -1.0},{1.5 if index % 12 < 6 else 4.0}")

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        prices_path = temp_path / "prices.csv"
        macro_path = temp_path / "macro.csv"
        prices_path.write_text("\n".join(prices), encoding="utf-8")
        macro_path.write_text("\n".join(macro), encoding="utf-8")

        with sync_playwright() as playwright:
            browser = playwright.chromium.launch()
            page = browser.new_page(viewport={"width": 1440, "height": 1000})
            browser_errors: list[str] = []
            page.on("console", lambda message: browser_errors.append(message.text) if message.type == "error" else None)
            page.goto(f"{base_url}/?skipAutoLoad=1", wait_until="networkidle")
            page.locator("#connection.badge-ok").wait_for()
            # Legacy local settings are migrated into a recurring real-spending phase.
            page.evaluate(
                """localStorage.setItem('mcq-controls', JSON.stringify({
                    schemaVersion: 13,
                    withdrawal: '5',
                    'withdrawal-start-period': '10',
                    periods: '12',
                    paths: '1000',
                    decumulationEnabled: true
                }))"""
            )
            page.reload(wait_until="networkidle")
            page.locator("#connection.badge-ok").wait_for()
            page.locator("#custom-data-toggle > summary").click()
            page.locator("#csv-prices").set_input_files(str(prices_path))
            page.locator("#csv-macro").set_input_files(str(macro_path))
            page.locator("#simulation-settings > summary").click()
            decumulation = page.locator("#decumulation-enabled")
            phase = page.locator(".phase-row").first
            assert decumulation.is_checked()
            assert phase.locator('[data-field="start_month"]').input_value() == "10"
            assert phase.locator('[data-field="end_month"]').input_value() == "12"
            assert phase.locator('[data-field="annual_real_amount"]').input_value() == "60"
            assert phase.locator('[data-field="annual_real_amount"]').is_enabled()
            page.locator("#decumulation-mode").select_option("safe_rate")
            assert page.locator("#safe-rate-btn").is_visible()
            assert phase.locator('[data-field="spending_multiplier"]').is_visible()
            page.locator("#decumulation-mode").select_option("manual")
            phase = page.locator(".phase-row").first
            assert phase.locator('[data-field="annual_real_amount"]').input_value() == "60"
            page.locator("#add-decumulation-event").click()
            assert page.locator(".event-row").count() == 1
            page.locator(".event-row .planner-remove").click()
            assert page.locator(".event-row").count() == 0
            page.locator("label.decumulation-toggle").click()
            assert not decumulation.is_checked()
            assert phase.locator('[data-field="annual_real_amount"]').is_disabled()
            page.locator("label.decumulation-toggle").click()
            assert decumulation.is_checked()
            page.locator("#target-wealth").fill("150")
            page.locator("#run-btn").click()
            page.locator("#growth-content").wait_for(state="visible", timeout=120_000)
            page.locator("#run-message").filter(has_text="Simulation complete").wait_for()
            assert page.locator("#metric-grid").get_by_text("Annualized return", exact=True).is_visible()
            assert page.locator("#goal-metric-grid").get_by_text("Target success", exact=True).is_visible()
            assert page.locator("#chart-goal-probability svg").is_visible()
            assert page.locator("#chart-rolling-horizon svg").is_visible()
            page.locator('[data-tab="tab-drawdowns"]').click()
            assert page.locator("#chart-drawdown-fan svg").is_visible()
            assert page.locator("#chart-drawdown-episodes svg").is_visible()
            assert page.locator("#chart-recovery-required svg").is_visible()
            page.locator('[data-tab="tab-retirement"]').click()
            assert page.locator("#retirement-content").is_visible()
            assert page.locator("#retirement-metrics").get_by_text("Funded-spending ratio", exact=True).is_visible()
            assert page.locator("#chart-retirement-survival svg").is_visible()
            assert page.locator("#retirement-comparison table").is_visible()
            assert not browser_errors, f"Browser console errors: {browser_errors}"

            page.set_viewport_size({"width": 390, "height": 844})
            page.wait_for_timeout(250)
            overflow = page.evaluate("document.documentElement.scrollWidth - document.documentElement.clientWidth")
            assert overflow <= 1, f"Mobile layout overflows horizontally by {overflow}px"
            browser.close()


if __name__ == "__main__":
    main()
