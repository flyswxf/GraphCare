import csv
import time
import argparse
from pathlib import Path
from typing import List, Dict, Tuple

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, NoSuchElementException
from selenium.webdriver.edge.options import Options as EdgeOptions
from selenium.webdriver.edge.service import Service as EdgeService


SEARCH_URL = "https://data.pharnexcloud.com/6/table/31"
# XPaths provided by user
XPATH_INPUT = \
    "/html/body/div[1]/div[3]/div/div/div/div/div[1]/div/div[1]/div[2]/div[2]/div/div/input"
XPATH_TABLE_BODY = \
    "/html/body/div[1]/div[3]/div/div/div/div/div[3]/div[2]/div[2]/div/div[3]/table/tbody"
XPATH_FIRST_ROW = XPATH_TABLE_BODY + "/tr[1]"
XPATH_FIRST_ROW_CODE = XPATH_FIRST_ROW + "/td[1]/div"
XPATH_FIRST_ROW_NAME_ZH = XPATH_FIRST_ROW + "/td[3]/div"


def read_atc3_codes(atc_csv_path: Path) -> List[str]:
    """Read ATC.csv and return unique ATC3 codes (level==3.0 and code length==4)."""
    codes: List[str] = []
    seen = set()
    with atc_csv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            code = (row.get("code") or "").strip()
            level = (row.get("level") or "").strip()
            if level == "3.0" and len(code) == 4 and code not in seen:
                codes.append(code)
                seen.add(code)
    return codes


def setup_edge_driver(driver_path: Path, headless: bool = True) -> webdriver.Edge:
    options = EdgeOptions()
    # Headless mode can be toggled
    if headless:
        # newer chromium headless
        options.add_argument("--headless=new")
        options.add_argument("--disable-gpu")
    options.add_argument("--no-sandbox")
    options.add_argument("--window-size=1920,1080")
    service = EdgeService(str(driver_path))
    driver = webdriver.Edge(service=service, options=options)
    driver.set_page_load_timeout(30)
    return driver


def search_code_and_get_name(driver: webdriver.Edge, code: str, wait: WebDriverWait) -> Tuple[str, str]:
    """Search given ATC3 code and return (code, chinese_name). May return (code, "") if not found/mismatch."""
    # Focus and type into search input
    input_el = wait.until(EC.presence_of_element_located((By.XPATH, XPATH_INPUT)))
    # Clear existing text robustly
    try:
        input_el.clear()
    except Exception:
        pass
    input_el.send_keys(Keys.CONTROL, 'a')
    input_el.send_keys(Keys.DELETE)
    input_el.send_keys(code)
    input_el.send_keys(Keys.ENTER)

    # Wait for first row code cell to appear
    try:
        wait.until(EC.presence_of_element_located((By.XPATH, XPATH_FIRST_ROW_CODE)))
    except TimeoutException:
        return code, ""  # no results

    # Short settle delay for dynamic table rendering
    time.sleep(0.5)

    # Read first row displayed code and name
    try:
        displayed_code_el = driver.find_element(By.XPATH, XPATH_FIRST_ROW_CODE)
        displayed_code = displayed_code_el.text.strip()
    except NoSuchElementException:
        return code, ""

    # Verify code matches; if not, try to locate matching row within tbody
    name_zh = ""
    if displayed_code == code:
        try:
            name_el = driver.find_element(By.XPATH, XPATH_FIRST_ROW_NAME_ZH)
            name_zh = (name_el.text or "").strip()
        except NoSuchElementException:
            name_zh = ""
    else:
        # Fallback: scan table rows to find exact code match
        try:
            tbody = driver.find_element(By.XPATH, XPATH_TABLE_BODY)
            rows = tbody.find_elements(By.TAG_NAME, "tr")
            for row in rows:
                tds = row.find_elements(By.TAG_NAME, "td")
                if not tds or len(tds) < 3:
                    continue
                row_code = (tds[0].text or "").strip()
                if row_code == code:
                    name_zh = (tds[2].text or "").strip()
                    break
        except NoSuchElementException:
            pass

    return code, name_zh


def write_results(out_path: Path, results: List[Tuple[str, str]]):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["code", "name_zh"])  # header
        for code, name in results:
            writer.writerow([code, name])


def main():
    parser = argparse.ArgumentParser(description="Scrape ATC3 code to Chinese names using Selenium Edge.")
    # 修正仓库根目录为 GraphCare 根路径（utils -> SparseTest -> ehr_baselines -> GraphCare）
    repo_root = Path(__file__).resolve().parents[3]
    default_atc_csv = repo_root / "resources" / "ATC.csv"
    default_out_csv = repo_root / "resources" / "ATC_Chinese.csv"

    parser.add_argument("--driver", type=Path, default=Path(r"D:\edgedriver_win64\msedgedriver.exe"),
                        help="Path to msedgedriver.exe")
    parser.add_argument("--atc", type=Path, default=default_atc_csv, help="Path to ATC.csv")
    parser.add_argument("--out", type=Path, default=default_out_csv, help="Output CSV path for Chinese names")
    parser.add_argument("--no-headless", action="store_true", help="Run browser with UI (not headless)")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of codes to scrape (for testing)")

    args = parser.parse_args()

    codes = read_atc3_codes(args.atc)
    if args.limit:
        codes = codes[:args.limit]

    print(f"Total ATC3 codes to query: {len(codes)}")

    driver = setup_edge_driver(args.driver, headless=(not args.no_headless))
    wait = WebDriverWait(driver, 20)

    results: List[Tuple[str, str]] = []
    try:
        driver.get(SEARCH_URL)
        wait.until(EC.presence_of_element_located((By.XPATH, XPATH_INPUT)))

        for idx, code in enumerate(codes, start=1):
            try:
                found_code, name_zh = search_code_and_get_name(driver, code, wait)
                results.append((found_code, name_zh))
                print(f"[{idx}/{len(codes)}] {found_code} -> {name_zh}")
            except Exception as e:
                print(f"[{idx}/{len(codes)}] {code} -> ERROR: {e}")
                results.append((code, ""))
            # polite pacing to avoid triggering rate limits
            time.sleep(0.4)
    finally:
        driver.quit()

    write_results(args.out, results)
    print(f"Wrote {len(results)} rows to {args.out}")


if __name__ == "__main__":
    main()