"""
data/scrape_statement.py
=========================
Selenium-based fallback for scraping problem statements directly from Codeforces
when the API does not provide the full statement text.
"""
import logging
from typing import Any

logger = logging.getLogger(__name__)

# Try importing selenium; fail gracefully if not available
try:
    from selenium import webdriver
    from selenium.webdriver.common.by import By
    from selenium.webdriver.firefox.options import Options
    from selenium.webdriver.support import expected_conditions as EC
    from selenium.webdriver.support.ui import WebDriverWait
    from selenium.common.exceptions import WebDriverException, TimeoutException
    HAS_SELENIUM = True
except ImportError:
    HAS_SELENIUM = False


def scrape_problem(url: str) -> dict[str, Any] | None:
    """
    Scrape a Codeforces problem statement and samples.
    Gracefully returns None if Selenium or geckodriver is unavailable, or on timeout.
    
    Returns:
        dict with keys: "statement" (str), "samples" (list of dicts)
    """
    if not HAS_SELENIUM:
        logger.warning("Selenium is not installed. Scraper disabled.")
        return None

    options = Options()
    options.add_argument("--headless")
    options.add_argument("--disable-gpu")
    options.add_argument("--no-sandbox")

    try:
        # This will fail if geckodriver is missing or Firefox is not installed
        driver = webdriver.Firefox(options=options)
    except WebDriverException as e:
        logger.warning("Could not initialize Firefox driver: %s. Scraper disabled.", e)
        return None

    try:
        driver.get(url)
        wait = WebDriverWait(driver, 10)
        
        # Wait for the problem statement to load
        problem_element = wait.until(
            EC.presence_of_element_located((By.CLASS_NAME, "problem-statement"))
        )

        # Scrape statement text (all top-level paragraphs inside problem-statement)
        # Note: Codeforces puts the main statement directly under .problem-statement > div (second one usually)
        # We can extract all text from the main body, skipping headers/samples
        header = problem_element.find_element(By.CLASS_NAME, "header")
        
        # We'll just grab the full text of the problem statement for simplicity,
        # but filter out the header and sample inputs/outputs.
        # A more robust way is to select the div elements that are not part of header, sample, or notes.
        divs = problem_element.find_elements(By.XPATH, "./div")
        
        statement_parts = []
        samples = []
        
        for div in divs:
            cls = div.get_attribute("class") or ""
            if "header" in cls:
                continue
            elif "sample-tests" in cls:
                # Parse sample tests
                inputs = div.find_elements(By.CLASS_NAME, "input")
                outputs = div.find_elements(By.CLASS_NAME, "output")
                
                for inp, outp in zip(inputs, outputs):
                    # For pre elements inside input/output, CF uses <pre> with <br> for newlines
                    inp_pre = inp.find_element(By.TAG_NAME, "pre")
                    outp_pre = outp.find_element(By.TAG_NAME, "pre")
                    samples.append({
                        "input": inp_pre.text,
                        "output": outp_pre.text
                    })
            else:
                statement_parts.append(div.text)

        statement_text = "\n\n".join(statement_parts).strip()
        
        return {
            "statement": statement_text,
            "samples": samples
        }

    except TimeoutException:
        logger.warning("Timeout waiting for problem statement at %s", url)
        return None
    except Exception as e:
        logger.warning("Error scraping %s: %s", url, e)
        return None
    finally:
        driver.quit()
