import requests
from bs4 import BeautifulSoup
from pathlib import Path


DOCUMENTS = {
    "fastapi_first_steps.md": "https://fastapi.tiangolo.com/tutorial/first-steps/",
    "fastapi_path_parameters.md": "https://fastapi.tiangolo.com/tutorial/path-params/",
    "fastapi_query_parameters.md": "https://fastapi.tiangolo.com/tutorial/query-params/",
    "fastapi_request_body.md": "https://fastapi.tiangolo.com/tutorial/body/",
}


OUTPUT_DIR = Path("data/documents")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def fetch_document(filename, url):
    print(f"Fetching: {url}")

    response = requests.get(
        url,
        headers={
            "User-Agent": "Mozilla/5.0"
        },
        timeout=30
    )

    response.raise_for_status()

    soup = BeautifulSoup(response.text, "html.parser")

    # Remove elements that are not useful for our RAG corpus
    for element in soup(["script", "style", "nav", "footer"]):
        element.decompose()

    main_content = soup.find("main")

    if main_content is None:
        raise ValueError(f"Could not find main content in {url}")

    title = main_content.find("h1")

    if title:
        title_text = title.get_text(" ", strip=True)
    else:
        title_text = filename.replace(".md", "").replace("_", " ").title()

    text = main_content.get_text("\n", strip=True)

    markdown_content = f"""# {title_text}

Source: {url}

{text}
"""

    output_file = OUTPUT_DIR / filename
    output_file.write_text(markdown_content, encoding="utf-8")

    print(f"Saved: {output_file}")


def main():
    for filename, url in DOCUMENTS.items():
        try:
            fetch_document(filename, url)
        except Exception as e:
            print(f"ERROR while processing {url}")
            print(e)


if __name__ == "__main__":
    main()