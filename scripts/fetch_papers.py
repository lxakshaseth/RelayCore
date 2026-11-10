"""Script to fetch exact versions of the 3 specified research papers from arXiv."""

import os
import sys
import urllib.request
import urllib.error

PAPERS = {
    "1706.03762v7": {
        "title": "Attention Is All You Need",
        "arxiv_id": "1706.03762",
        "version": "v7",
        "url": "https://arxiv.org/pdf/1706.03762v7.pdf",
        "filename": "1706.03762v7_Attention_Is_All_You_Need.pdf",
    },
    "2410.00037v1": {
        "title": "Moshi: a speech-text foundation model for real-time dialogue",
        "arxiv_id": "2410.00037",
        "version": "v1",
        "url": "https://arxiv.org/pdf/2410.00037v1.pdf",
        "filename": "2410.00037v1_Moshi.pdf",
    },
    "2507.07955v2": {
        "title": "Dynamic Chunking for End-to-End Hierarchical Sequence Modeling",
        "arxiv_id": "2507.07955",
        "version": "v2",
        "url": "https://arxiv.org/pdf/2507.07955v2.pdf",
        "alt_urls": [
            "https://arxiv.org/pdf/2407.07955v2.pdf",
            "https://arxiv.org/pdf/2307.07955v2.pdf",
        ],
        "filename": "2507.07955v2_Dynamic_Chunking.pdf",
    },
}


def download_paper(paper_key: str, data: dict, output_dir: str) -> str:
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, data["filename"])
    if os.path.exists(out_path) and os.path.getsize(out_path) > 1000:
        print(f"[{paper_key}] Already exists: {out_path} ({os.path.getsize(out_path)} bytes)")
        return out_path

    urls_to_try = [data["url"]] + data.get("alt_urls", [])
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36 RelayCorePaperFetcher/1.0"
    }

    success = False
    for url in urls_to_try:
        print(f"[{paper_key}] Downloading from {url}...")
        try:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=30) as response, open(out_path, "wb") as f:
                content = response.read()
                # Verify PDF header
                if content.startswith(b"%PDF"):
                    f.write(content)
                    print(f"[{paper_key}] Successfully downloaded ({len(content)} bytes) -> {out_path}")
                    success = True
                    break
                else:
                    print(f"[{paper_key}] Response was not a valid PDF (length: {len(content)}). Response snippet: {content[:200]}")
        except urllib.error.HTTPError as e:
            print(f"[{paper_key}] HTTP {e.code}: {e.reason} for {url}")
        except Exception as e:
            print(f"[{paper_key}] Error: {e} for {url}")

    if not success:
        print(f"[{paper_key}] Failed to download from any URL.")
    return out_path


def main():
    target_dir = os.path.join("data", "papers")
    print(f"Fetching 3 required research papers into {target_dir}...")
    for key, info in PAPERS.items():
        download_paper(key, info, target_dir)


if __name__ == "__main__":
    main()
