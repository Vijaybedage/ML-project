"""
Convert Project_Report.html to PDF using a lightweight approach.
Tries multiple methods in order of preference.
"""
import subprocess
import sys
import os

HTML_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Project_Report.html")
PDF_PATH  = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Project_Report.pdf")

def try_edge_headless():
    """Use Microsoft Edge in headless mode to print to PDF."""
    edge_paths = [
        r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
        r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
    ]
    for edge in edge_paths:
        if os.path.exists(edge):
            file_url = "file:///" + HTML_PATH.replace("\\", "/").replace(" ", "%20")
            cmd = [
                edge,
                "--headless",
                "--disable-gpu",
                "--no-sandbox",
                f"--print-to-pdf={PDF_PATH}",
                "--print-to-pdf-no-header",
                file_url
            ]
            print(f"Using Edge: {edge}")
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            if os.path.exists(PDF_PATH) and os.path.getsize(PDF_PATH) > 1000:
                return True
    return False

def try_chrome_headless():
    """Use Google Chrome in headless mode to print to PDF."""
    chrome_paths = [
        r"C:\Program Files\Google\Chrome\Application\chrome.exe",
        r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
        os.path.expandvars(r"%LOCALAPPDATA%\Google\Chrome\Application\chrome.exe"),
    ]
    for chrome in chrome_paths:
        if os.path.exists(chrome):
            file_url = "file:///" + HTML_PATH.replace("\\", "/").replace(" ", "%20")
            cmd = [
                chrome,
                "--headless",
                "--disable-gpu",
                "--no-sandbox",
                f"--print-to-pdf={PDF_PATH}",
                "--print-to-pdf-no-header",
                file_url
            ]
            print(f"Using Chrome: {chrome}")
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            if os.path.exists(PDF_PATH) and os.path.getsize(PDF_PATH) > 1000:
                return True
    return False

if __name__ == "__main__":
    print(f"HTML: {HTML_PATH}")
    print(f"PDF:  {PDF_PATH}")
    print()

    # Try Edge first (comes with Windows)
    print("Attempting Microsoft Edge headless...")
    if try_edge_headless():
        size_kb = os.path.getsize(PDF_PATH) / 1024
        print(f"\n✅ PDF created successfully! ({size_kb:.0f} KB)")
        print(f"   Location: {PDF_PATH}")
        sys.exit(0)

    # Try Chrome
    print("Attempting Google Chrome headless...")
    if try_chrome_headless():
        size_kb = os.path.getsize(PDF_PATH) / 1024
        print(f"\n✅ PDF created successfully! ({size_kb:.0f} KB)")
        print(f"   Location: {PDF_PATH}")
        sys.exit(0)

    print("\n❌ Could not generate PDF automatically.")
    print(f"   Please open this file in your browser and press Ctrl+P to save as PDF:")
    print(f"   {HTML_PATH}")
    sys.exit(1)
