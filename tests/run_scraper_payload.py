#!/usr/bin/env python3
"""Dump AEAT scraper output for a sample factura payload.

This helper mirrors the style of the other scripts in ``tests/`` and is aimed at
quickly inspecting what the AEAT scrapers return. Provide a JSON invoice payload
(the default reuses ``tests/payload.json``) and the script will boot the shared
scraper pool, run ``search_comprehensive`` and print the structured result so it
can be reused as test data or fed into other tools.

Example:
    python tests/run_scraper_payload.py --payload tests/payload.json
Optional flags let you tweak pool size or toggle headless mode if you need to
observe the scrape in a real browser window.
"""

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import Any, Dict

from aeat_scraper_singleton import (
    ScraperConfig,
    async_search_comprehensive,
    initialize_aeat_scraper,
    shutdown_aeat_scraper,
)

DEFAULT_PAYLOAD = Path(__file__).with_name("payload.json")
DEFAULT_CONFIG = ScraperConfig()


def load_payload(path: Path) -> Dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except FileNotFoundError:
        raise SystemExit(f"Payload file not found: {path}")
    except json.JSONDecodeError as exc:
        raise SystemExit(f"Payload JSON is invalid: {exc}")


async def fetch_scraper_output(config: ScraperConfig, payload: Dict[str, Any]) -> Dict[str, Any]:
    if not initialize_aeat_scraper(config):
        raise RuntimeError("Could not initialize AEAT scraper pool. Check Chrome/Selenium setup.")

    try:
        return await async_search_comprehensive(payload)
    finally:
        shutdown_aeat_scraper()


async def async_main(args: argparse.Namespace) -> int:
    payload_path = Path(args.payload).expanduser().resolve()
    payload = load_payload(payload_path)

    config = ScraperConfig(
        headless=not args.show_browser,
        verbose=args.verbose,
        use_pool=not args.single_driver,
        pool_size=args.pool_size,
        max_retries=args.max_retries,
        timeout=args.timeout,
        auto_restart_interval=args.auto_restart,
    )

    try:
        result = await fetch_scraper_output(config, payload)
    except Exception as exc:  # noqa: BLE001
        sys.stderr.write(f"Error running AEAT scraper: {exc}\n")
        return 1

    output = json.dumps(result, ensure_ascii=False, indent=2)

    if args.output:
        output_path = Path(args.output).expanduser().resolve()
        output_path.write_text(output, encoding="utf-8")
        print(f"Scraper payload written to {output_path}")
    else:
        print(output)

    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run AEAT scraper and dump the payload")
    parser.add_argument(
        "--payload",
        default=str(DEFAULT_PAYLOAD),
        help="Path to the JSON factura payload (default: tests/payload.json)",
    )
    parser.add_argument(
        "--pool-size",
        type=int,
        default=DEFAULT_CONFIG.pool_size,
        help="Number of Selenium drivers in the pool",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=DEFAULT_CONFIG.timeout,
        help="Per-request timeout in seconds",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=DEFAULT_CONFIG.max_retries,
        help="Maximum retries before giving up",
    )
    parser.add_argument(
        "--auto-restart",
        type=int,
        default=DEFAULT_CONFIG.auto_restart_interval,
        help="Seconds between automatic pool restarts",
    )
    parser.add_argument(
        "--single-driver",
        action="store_true",
        help="Disable the driver pool and use a single Selenium session",
    )
    parser.add_argument(
        "--show-browser",
        action="store_true",
        help="Run Chrome in headed mode to observe interactions",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose scraper logging",
    )
    parser.add_argument(
        "--output",
        help="Optional path to store the resulting AEAT payload as JSON",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    try:
        exit_code = asyncio.run(async_main(args))
    except KeyboardInterrupt:
        exit_code = 130
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
