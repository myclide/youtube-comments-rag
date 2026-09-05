import os
from pathlib import Path

import pandas as pd
import requests


API_URL = "https://www.googleapis.com/youtube/v3/commentThreads"

VIDEO_ID = "gyK9USvrvDQ"
VIDEO_TITLE = "Top 10 BEST SELLING Books In History"

# 现在只测试分页
MAX_COMMENTS = 300

OUTPUT_PATH = (
    Path("data")
    / f"test_comments_{VIDEO_ID}.jsonl"
)


def request_comment_page(
    video_id,
    api_key,
    page_token=None,
):
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

    response = requests.get(
        API_URL,
        params=params,
        timeout=30,
    )

    # 避免错误信息暴露 API key
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


def collect_comments(
    video_id,
    video_title,
    max_comments=None,
):
    api_key = os.getenv("YOUTUBE_API_KEY")

    if not api_key:
        raise ValueError(
            "YOUTUBE_API_KEY environment variable is not set."
        )

    records = []

    page_token = None
    pages_requested = 0

    while True:
        data = request_comment_page(
            video_id=video_id,
            api_key=api_key,
            page_token=page_token,
        )

        pages_requested += 1

        for item in data.get("items", []):
            thread_snippet = item["snippet"]

            top_comment = (
                thread_snippet["topLevelComment"]
            )

            comment_id = top_comment["id"]

            snippet = top_comment["snippet"]

            # 优先保留原始文本
            comment_text = (
                snippet.get("textOriginal")
                or snippet.get("textDisplay")
                or ""
            )

            records.append(
                {
                    "video_id": video_id,
                    "video_title": video_title,
                    "comment_id": comment_id,
                    "author": snippet.get(
                        "authorDisplayName"
                    ),
                    "comment_text": comment_text,
                    "like_count": snippet.get(
                        "likeCount", 0
                    ),
                    "published_at": snippet.get(
                        "publishedAt"
                    ),
                    "updated_at": snippet.get(
                        "updatedAt"
                    ),
                    "reply_count": thread_snippet.get(
                        "totalReplyCount", 0
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

        page_token = data.get("nextPageToken")

        if not page_token:
            break

    df = pd.DataFrame(records)

    # 理论上 API ID 应唯一，
    # 这里仍做防御性去重
    if not df.empty:
        df = (
            df
            .drop_duplicates(
                subset="comment_id",
                keep="first",
            )
            .reset_index(drop=True)
        )

    return df, pages_requested


if __name__ == "__main__":
    df, pages = collect_comments(
        video_id=VIDEO_ID,
        video_title=VIDEO_TITLE,
        max_comments=MAX_COMMENTS,
    )

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    # JSONL 非常适合保留多行评论：
    # 每个 JSON object = 一条真实 comment
    df.to_json(
        OUTPUT_PATH,
        orient="records",
        lines=True,
        force_ascii=False,
        date_format="iso",
    )

    print("=" * 70)
    print("COLLECTION RESULT")
    print("=" * 70)

    print(f"Pages requested: {pages}")
    print(f"Comments collected: {len(df)}")
    print(
        f"Unique comment IDs: "
        f"{df['comment_id'].nunique()}"
    )
    print(
        f"Duplicate comment IDs: "
        f"{df['comment_id'].duplicated().sum()}"
    )

    print()

    multiline = (
        df["comment_text"]
        .str.contains("\n", regex=False)
        .sum()
    )

    print(f"Multiline comments: {multiline}")

    print(
        "Multiline percentage:",
        round(multiline / len(df) * 100, 2),
        "%",
    )

    print()

    print(f"Saved to: {OUTPUT_PATH}")