#!/usr/bin/env python3
"""
Download SKU images from the pattern:
  https://snap-on-products-hr.imgix.net/<itemsku>.jpg?w=600&dpr=2&auto=format&fit=max&q=25

Usage examples:
  python download_sku_images.py --list "YA1234,BC-5678,CA-9012" -o images
  python download_sku_images.py --file skus.csv --column sku -o output_images
  python download_sku_images.py --file skus.txt -o images --force

Requires: requests
  pip install requests

Outputs:
  - Images saved as <sku>.jpg in the output directory (default: ./images)
  - A results CSV (download_report.csv) with per-SKU status
"""
import argparse
import csv
import sys
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Iterable, List, Tuple
from urllib.parse import quote

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

URL_TEMPLATE = "https://snap-on-products-hr.imgix.net/{sku}.jpg?w=600&dpr=2&auto=format&fit=max&q=25"

def build_session() -> requests.Session:
    # Robust retry strategy for transient errors
    retry = Retry(
        total=5,
        backoff_factor=0.5,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=frozenset(["GET"]),
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry, pool_connections=100, pool_maxsize=100)
    s = requests.Session()
    s.mount("http://", adapter)
    s.mount("https://", adapter)
    s.headers.update({"User-Agent": "sku-image-downloader/1.0"})
    return s

_invalid_filename_chars = re.compile(r'[^A-Za-z0-9._\- ]+')

def sanitize_filename(name: str) -> str:
    name = name.strip()
    name = _invalid_filename_chars.sub("_", name)
    # Avoid empty filenames
    return name or "unnamed"

def read_skus_from_list(csv_list: str) -> List[str]:
    parts = [s.strip() for s in csv_list.split(",")]
    return [p for p in parts if p]

def read_skus_from_file(path: Path, column: str = None) -> List[str]:
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")
    # Detect csv vs txt by extension
    if path.suffix.lower() in {".csv"} or column:
        # CSV mode
        with path.open("r", newline="", encoding="utf-8-sig") as f:
            sample = f.read(4096)
            f.seek(0)
            # Try to sniff delimiter
            sniffer = csv.Sniffer()
            try:
                dialect = sniffer.sniff(sample)
            except csv.Error:
                dialect = csv.excel
            reader = csv.DictReader(f, dialect=dialect)
            if column and column not in reader.fieldnames:
                raise ValueError(f"Column '{column}' not found. Available: {reader.fieldnames}")
            col = column or reader.fieldnames[0]
            skus = [ (row.get(col) or "").strip() for row in reader ]
            return [s for s in skus if s]
    else:
        # TXT mode: one SKU per line
        return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]

def download_one(session: requests.Session, sku: str, out_dir: Path, force: bool=False, timeout=(5, 30)) -> Tuple[str, str, str]:
    """
    Returns: (sku, status, message)
    status ∈ {"ok","skip","not_found","error"}
    """
    safe_name = sanitize_filename(sku)
    out_path = out_dir / f"{safe_name}.jpg"
    if out_path.exists() and not force:
        return (sku, "skip", "exists")

    # Encode SKU for URL path
    url = URL_TEMPLATE.format(sku=quote(sku, safe=""))
    try:
        resp = session.get(url, stream=True, timeout=timeout)
    except requests.RequestException as e:
        return (sku, "error", f"request_failed: {e}")

    if resp.status_code == 404:
        return (sku, "not_found", "404")
    if resp.status_code != 200:
        return (sku, "error", f"http_{resp.status_code}")

    ctype = resp.headers.get("Content-Type", "")
    if "image" not in ctype.lower():
        return (sku, "error", f"unexpected_content_type:{ctype}")

    try:
        out_dir.mkdir(parents=True, exist_ok=True)
        with out_path.open("wb") as f:
            for chunk in resp.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
    except Exception as e:
        return (sku, "error", f"write_failed:{e}")
    finally:
        resp.close()

    return (sku, "ok", str(out_path))

def write_report(rows: Iterable[Tuple[str, str, str]], out_dir: Path) -> Path:
    report_path = out_dir / "download_report.csv"
    with report_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["sku", "status", "detail"])
        for r in rows:
            writer.writerow(r)
    return report_path

def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Download images for a list of SKUs.")
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--list", help="Comma-separated list of SKUs")
    src.add_argument("--file", help="Path to a TXT (one SKU per line) or CSV file")
    p.add_argument("--column", help="For CSV input, column name containing SKUs")
    p.add_argument("-o", "--out", default="images", help="Output directory (default: images)")
    p.add_argument("--workers", type=int, default=12, help="Number of parallel downloads (default: 12)")
    p.add_argument("--force", action="store_true", help="Re-download even if file exists")
    return p.parse_args(argv)

def main(argv=None) -> int:
    args = parse_args(argv)
    out_dir = Path(args.out)

    # Collect SKUs
    if args.list:
        skus = read_skus_from_list(args.list)
    else:
        skus = read_skus_from_file(Path(args.file), column=args.column)

    if not skus:
        print("No SKUs found to download.", file=sys.stderr)
        return 2

    session = build_session()

    results: List[Tuple[str, str, str]] = []
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as ex:
        futures = {ex.submit(download_one, session, sku, out_dir, args.force): sku for sku in skus}
        for fut in as_completed(futures):
            sku = futures[fut]
            try:
                res = fut.result()
            except Exception as e:
                res = (sku, "error", f"unhandled:{e}")
            results.append(res)
            # Minimal progress echo
            print(f"{res[0]} -> {res[1]} {('('+res[2]+')') if res[2] else ''}")

    report = write_report(results, out_dir)
    print(f"\nDone. Images in: {out_dir.resolve()}")
    print(f"Report: {report.resolve()}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())