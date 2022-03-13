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


def main(
    input_dir: Path = typer.Argument(..., exists=True),
    output_dir: Path = typer.Argument(...),
    case_num: int = typer.Argument(0),
    feature_num: int = typer.Argument(0),
    train_json_file_name: str = "train_split.json.gz",
):
    nlp = spacy.blank("en")

    with gzip.open(input_dir / train_json_file_name, "r") as fin:
        df = pd.read_json(fin)


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
        current_s = []
        ints = re.findall(r"\d+", s)
        assert len(ints) % 2 == 0
        for i in range(0, len(ints), 2):
            current_s.append((int(ints[i]), int(ints[i + 1])))
        spans.append(current_s)
    return spans


df.location = df.location.apply(process_spans)


for i, r in df.iterrows():
    for expected, span in zip(r.annotation, r.location):
        actual = ""
        sep = ""
        for sub in span:
            actual += sep + r.pn_history[sub[0] : sub[1]]
            sep = " "
        if expected != actual:
            print(f"Row: {i}")
            print(f"Expected: {expected}")
            print(f"Actual: {actual}")


sub_df = df.loc[(df.case_num == case_num) & (df.feature_num == feature_num), :]


feature_text = sub_df.feature_text.unique()
assert len(feature_text) == 1
feature_text[0]


assert len(sub_df.pn_num.unique()) == len(sub_df)


def df_to_docbin(in_df: pd.DataFrame):
    db = DocBin()
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
                char_span = doc.char_span(start, end, alignment_mode="expand")
                if char_span is None:
                    raise ValueError()
                ents.append(char_span)
        doc.ents = ents
        db.add(doc)
    return db


train_df = sub_df.loc[
    sub_df.fold.isin([0, 1, 2]),
]
train_db = df_to_docbin(train_df)

dev_df = sub_df.loc[
    sub_df.fold.isin([3]),
]
dev_db = df_to_docbin(dev_df)

test_df = sub_df.loc[
    sub_df.fold.isin([4]),
]
test_db = df_to_docbin(test_df)


assert len(train_db) == 60
assert len(dev_db) == 20
assert len(test_db) == 20


train_db.to_disk(
    output_dir / f"train-case_num-{case_num}-feature_num-{feature_num}.spacy"
)
dev_db.to_disk(output_dir / f"dev-case_num-{case_num}-feature_num-{feature_num}.spacy")
test_db.to_disk(
    output_dir / f"test-case_num-{case_num}-feature_num-{feature_num}.spacy"
)

if __name__ == "__main__":
    typer.run(main)
