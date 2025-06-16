import argparse
import asyncio
from preprocessing.io_utils import load_documents, prepare_batches, save_json
from preprocessing.text_utils import async_summarize

async def batch_preprocess_and_summarize(
    directory: str,
    role: str = "mentor",
    batch_size: int = 5,
    extensions: list[str] = None,
) -> list[dict]:
    """
    Load all docs, asynchronously summarize each, return list of dicts with filename and summary.
    """
    docs = load_documents(directory, extensions)
    texts = [text for _, text in docs]
    batches = prepare_batches(texts, batch_size)
    summaries: list[dict] = []
    for b_idx, batch in enumerate(batches):
        results = await asyncio.gather(*[async_summarize(text, role) for text in batch])
        for j, summary in enumerate(results):
            file_name = docs[b_idx * batch_size + j][0]
            summaries.append({"filename": file_name, "summary": summary})
    return summaries


def main():
    parser = argparse.ArgumentParser(description="Preprocess and summarize documents in batch.")
    parser.add_argument("--dir", required=True, help="Directory of documents to process")
    parser.add_argument("--out", required=True, help="Output JSON file for summaries")
    parser.add_argument(
        "--role", choices=["mentor", "mentee"], default="mentor",
        help="Summary type"
    )
    parser.add_argument(
        "--batch-size", type=int, default=5,
        help="Number of documents per batch"
    )
    parser.add_argument(
        "--exts", nargs="+", default=[".pdf", ".docx", ".txt"],
        help="File extensions to include"
    )
    args = parser.parse_args()

    summaries = asyncio.run(
        batch_preprocess_and_summarize(
            args.dir, args.role, args.batch_size, args.exts
        )
    )
    save_json(summaries, args.out)

if __name__ == "__main__":
    main()