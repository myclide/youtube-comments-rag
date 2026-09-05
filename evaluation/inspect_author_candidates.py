import pandas as pd
import re
from collections import Counter


DATA_PATH = r"data\Final Result.csv"

df = pd.read_csv(DATA_PATH)

pattern = re.compile(r"^(.+?)\s-\s(.*)$")

authors = []
suspicious = []

for raw in df["Comments"].fillna(""):
    for line in str(raw).split("\n"):

        line = line.strip()

        if not line:
            continue

        match = pattern.match(line)

        if not match:
            continue

        author = match.group(1).strip()
        text = match.group(2).strip()

        authors.append(author)

        # 临时用来发现可能的错误边界
        if (
            len(author) > 60
            or len(author) <= 1
            or author.lower() in {
                "pros",
                "cons",
                "pro",
                "con",
                "edit",
                "update",
                "note",
                "ps",
                "p.s."
            }
        ):
            suspicious.append(
                (author, text)
            )


print("=" * 60)
print("AUTHOR CANDIDATE AUDIT")
print("=" * 60)

print(f"Total candidate comment starts: {len(authors):,}")
print(f"Unique candidate authors:       {len(set(authors)):,}")

print()

lengths = pd.Series([len(x) for x in authors])

print("Author length statistics:")
print(lengths.describe())

print()

print("=" * 60)
print("MOST COMMON AUTHORS")
print("=" * 60)

for author, count in Counter(authors).most_common(20):
    print(count, repr(author))

print()

print("=" * 60)
print("SUSPICIOUS CANDIDATES")
print("=" * 60)

print(f"Suspicious count: {len(suspicious):,}")

for author, text in suspicious[:50]:
    print(
        "AUTHOR:",
        repr(author),
        "| TEXT:",
        repr(text[:150])
    )