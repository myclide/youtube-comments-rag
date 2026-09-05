import pandas as pd
import re


DATA_PATH = r"data\Final Result.csv"

df = pd.read_csv(DATA_PATH)

total_lines = 0
normal_starts = 0
empty_author_starts = 0
continuation_lines = 0

normal_examples = []
empty_author_examples = []
continuation_examples = []

# 新评论候选：
# 前面存在内容，然后出现 " - "
comment_start_pattern = re.compile(r"^.+?\s-\s")

for raw in df["Comments"].fillna(""):
    lines = str(raw).split("\n")

    for line in lines:
        line = line.strip()

        if not line:
            continue

        total_lines += 1

        if line.startswith("- "):
            empty_author_starts += 1

            if len(empty_author_examples) < 10:
                empty_author_examples.append(line)

        elif comment_start_pattern.match(line):
            normal_starts += 1

            if len(normal_examples) < 10:
                normal_examples.append(line)

        else:
            continuation_lines += 1

            if len(continuation_examples) < 20:
                continuation_examples.append(line)


print("=" * 60)
print("RAW COMMENT FORMAT")
print("=" * 60)

print(f"Total non-empty lines:        {total_lines:,}")
print(f"Normal author starts:         {normal_starts:,}")
print(f"Empty-author starts:          {empty_author_starts:,}")
print(f"Possible continuation lines:  {continuation_lines:,}")

print()

print(
    "Normal-start percentage:",
    f"{normal_starts / total_lines * 100:.2f}%"
)

print(
    "Continuation percentage:",
    f"{continuation_lines / total_lines * 100:.2f}%"
)

print()

print("=" * 60)
print("NORMAL START EXAMPLES")
print("=" * 60)

for x in normal_examples:
    print(repr(x))

print()

print("=" * 60)
print("EMPTY AUTHOR EXAMPLES")
print("=" * 60)

for x in empty_author_examples:
    print(repr(x))

print()

print("=" * 60)
print("CONTINUATION EXAMPLES")
print("=" * 60)

for x in continuation_examples:
    print(repr(x))