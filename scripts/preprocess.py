import gzip
import re
import random
from pathlib import Path

import pandas as pd
import spacy
from spacy.tokens import DocBin
import typer
from wasabi import msg


random.seed(42)

nlp = spacy.blank("en")


def process_spans(span_str: str):
    """
    Sanitize spans by converting them to lists of lists of two integer tuples.
    This seems to handle the best way to take care of both comma- and semicolon-based
    delimitation of spans. Examples:
    ['696 724'] -> [[(696, 724)]]
    ['501 517', '482 488;522 530'] -> [[(501, 517)], [(482, 488), (522, 530)]]
    """
    spans = []
    for s in span_str:
        ints = re.findall(r"\d+", s)
        assert len(ints) % 2 == 0
        current_s = [(int(ints[i]), int(ints[i + 1])) for i in range(0, len(ints), 2)]
        spans.append(current_s)
    return spans


def df_to_doc_list(in_df: pd.DataFrame):
    doc_list = []
    for _, row in in_df.iterrows():
        doc = nlp(row.pn_history)
        doc.user_data = {
            "id": row.id,
            "case_num": row.case_num,
            "pn_num": row.pn_num,
            "feature_num": row.feature_num,
            "feature_text": row.feature_text,
        }
        ents = []
        for span in row.location:
            for start, end in span:
                # NOTE: expand to spacy default tokenization
                char_span = doc.char_span(start, end, alignment_mode="expand", label="nbme")
                if char_span is None:
                    raise ValueError()
                ents.append(char_span)
        doc.ents = ents
        doc_list.append(doc)
    return doc_list


def main(
    input_dir: Path = typer.Argument(..., exists=True),
    output_dir: Path = typer.Argument(...),
    case_num: int = typer.Argument(0, help="The case_num to use, set -1 for all."),
    feature_num: int = typer.Argument(0, help="The feature_num to use, set -1 for all."),
    train_json_file_name: str = "train_split.json.gz",
):
    msg.info("Loading raw data.")
    with gzip.open(input_dir / train_json_file_name, "r") as fin:
        df = pd.read_json(fin)

    df.location = df.location.apply(process_spans)

    for i, r in df.iterrows():
        for expected, span in zip(r.annotation, r.location):
            actual = ""
            sep = ""
            for sub in span:
                actual += sep + r.pn_history[sub[0] : sub[1]]
                sep = " "
            if expected != actual:
                msg.warn("Mismatched span: ")
                msg.warn(f"Row: {i}")
                msg.warn(f"Expected: {expected}")
                msg.warn(f"Actual: {actual}")
                msg.warn("")

    if case_num != -1:
        df = df.loc[(df.case_num == case_num), :]

    if feature_num != -1:
        df = df.loc[(df.feature_num == feature_num), :]

    train_df = df.loc[
        df.fold.isin([0, 1, 2]),
    ]
    train_doc_list = df_to_doc_list(train_df)

    dev_df = df.loc[
        df.fold.isin([3]),
    ]
    dev_doc_list = df_to_doc_list(dev_df)

    test_df = df.loc[
        df.fold.isin([4]),
    ]
    test_doc_list = df_to_doc_list(test_df)

    msg.good(f"Num Train Docs: {len(train_doc_list)}")
    msg.good(f"Num Dev Docs: {len(dev_doc_list)}")
    msg.good(f"Num Test Docs: {len(test_doc_list)}")

    random.shuffle(train_doc_list)

    train_db = DocBin(docs=train_doc_list)
    dev_db = DocBin(docs=dev_doc_list)
    test_db = DocBin(docs=test_doc_list)

    with msg.loading(f"Saving docs to: {output_dir}..."):
        train_db.to_disk(
            output_dir / f"train-case_num-{case_num}-feature_num-{feature_num}.spacy"
        )
        dev_db.to_disk(
            output_dir / f"dev-case_num-{case_num}-feature_num-{feature_num}.spacy"
        )
        test_db.to_disk(
            output_dir / f"test-case_num-{case_num}-feature_num-{feature_num}.spacy"
        )
        msg.good("Done.")


if __name__ == "__main__":
    typer.run(main)
