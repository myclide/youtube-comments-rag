import os
import re
import time
from pathlib import Path

import pandas as pd
import requests


COMMENTS_API = (
    "https://www.googleapis.com/youtube/v3/commentThreads"
)

VIDEOS_API = (
    "https://www.googleapis.com/youtube/v3/videos"
)

SOURCE_PATH = Path("data") / "Final Result.csv"

OUTPUT_PATH = (
    Path("data")
    / "structured_comments_test.jsonl"
)

# 现在只测试前 10 个视频
MAX_VIDEOS = 10

# 每个视频最多抓 300 条
MAX_COMMENTS_PER_VIDEO = 300


def get_api_key():
    api_key = os.getenv("YOUTUBE_API_KEY")

    if not api_key:
        raise ValueError(
            "YOUTUBE_API_KEY environment variable is not set."
        )

    return api_key


def extract_video_id(url):
    match = re.search(
        r"(?:v=|youtu\.be/)([^&?/]+)",
        str(url),
    )

    if match:
        return match.group(1)

    return None


def safe_request(url, params):
    response = requests.get(
        url,
        params=params,
        timeout=30,
    )

    if response.status_code != 200:
        try:
            error_data = response.json()

            message = (
                error_data
                .get("error", {})
                .get("message", "Unknown API error")
            )

        except ValueError:
            message = "Unknown API error"

        raise RuntimeError(
            f"YouTube API error "
            f"{response.status_code}: {message}"
        )

    return response.json()


def video_is_available(video_id, api_key):
    data = safe_request(
        VIDEOS_API,
        {
            "part": "status",
            "id": video_id,
            "key": api_key,
        },
    )

    return len(data.get("items", [])) > 0


def collect_video_comments(
    video_id,
    video_title,
    api_key,
    max_comments=None,
):
    records = []

    page_token = None
    pages = 0

    while True:
        params = {
            "part": "snippet",
            "videoId": video_id,
            "maxResults": 100,
            "textFormat": "plainText",
            "order": "time",
            "key": api_key,
        }

        if page_token:
            params["pageToken"] = page_token

        data = safe_request(
            COMMENTS_API,
            params,
        )

        pages += 1

        for item in data.get("items", []):
            thread_snippet = item["snippet"]

            top_comment = (
                thread_snippet["topLevelComment"]
            )

            snippet = top_comment["snippet"]

            text = (
                snippet.get("textOriginal")
                or snippet.get("textDisplay")
                or ""
            )

            records.append(
                {
                    "video_id": video_id,
                    "video_title": video_title,
                    "comment_id": top_comment["id"],
                    "author": snippet.get(
                        "authorDisplayName"
                    ),
                    "comment_text": text,
                    "like_count": snippet.get(
                        "likeCount", 0
                    ),
                    "published_at": snippet.get(
                        "publishedAt"
                    ),
                    "updated_at": snippet.get(
                        "updatedAt"
                    ),
                    "reply_count": (
                        thread_snippet.get(
                            "totalReplyCount", 0
                        )
                    ),
                }
            )

            if (
                max_comments is not None
                and len(records) >= max_comments
            ):
                break

        if (
            max_comments is not None
            and len(records) >= max_comments
        ):
            break

        page_token = data.get(
            "nextPageToken"
        )

        if not page_token:
            break

    return records, pages


def main():
    api_key = get_api_key()

    source_df = pd.read_csv(
        SOURCE_PATH
    ).head(MAX_VIDEOS)

    all_records = []

    total_pages = 0

    for index, row in source_df.iterrows():
        video_title = str(
            row["Video Title"]
        )

        video_id = extract_video_id(
            row["Video Link"]
        )

        print()
        print("-" * 70)
        print(
            f"[{index + 1}/{len(source_df)}] "
            f"{video_title}"
        )

        if not video_id:
            print("SKIP: invalid video URL")
            continue

        try:
            if not video_is_available(
                video_id,
                api_key,
            ):
                print("SKIP: video unavailable")
                continue

            records, pages = (
                collect_video_comments(
                    video_id=video_id,
                    video_title=video_title,
                    api_key=api_key,
                    max_comments=(
                        MAX_COMMENTS_PER_VIDEO
                    ),
                )
            )

            print(
                f"Collected: {len(records)}"
            )

            print(
                f"Pages requested: {pages}"
            )

            all_records.extend(records)
            total_pages += pages

        except RuntimeError as exc:
            # 例如 comments disabled
            # 不让一个视频导致整个任务停止
            print(f"SKIP: {exc}")

        # 对 API 稍微友好一点
        time.sleep(0.2)

    df = pd.DataFrame(all_records)

    if df.empty:
        print("No comments collected.")
        return

    before = len(df)

    df = (
        df
        .drop_duplicates(
            subset="comment_id",
            keep="first",
        )
        .reset_index(drop=True)
    )

    removed = before - len(df)

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    df.to_json(
        OUTPUT_PATH,
        orient="records",
        lines=True,
        force_ascii=False,
    )

    multiline = (
        df["comment_text"]
        .str.contains("\n", regex=False)
        .sum()
    )

    print()
    print("=" * 70)
    print("DATASET SUMMARY")
    print("=" * 70)

    print(
        f"Videos attempted: "
        f"{len(source_df)}"
    )

    print(
        f"Videos collected: "
        f"{df['video_id'].nunique()}"
    )

    print(
        f"Total comments: "
        f"{len(df):,}"
    )

    print(
        f"Unique comment IDs: "
        f"{df['comment_id'].nunique():,}"
    )

    print(
        f"Duplicates removed: "
        f"{removed:,}"
    )

    print(
        f"Multiline comments: "
        f"{multiline:,}"
    )

    print(
        "Multiline percentage:",
        round(
            multiline / len(df) * 100,
            2,
        ),
        "%",
    )

    print(
        f"Comment API pages: "
        f"{total_pages}"
    )

    print(
        f"Saved to: {OUTPUT_PATH}"
    )


if __name__ == "__main__":
    main()