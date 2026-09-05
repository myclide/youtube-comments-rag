import os
import re
import time

import pandas as pd
import requests


SOURCE_PATH = r"data\Final Result.csv"

VIDEOS_API = (
    "https://www.googleapis.com/youtube/v3/videos"
)


def extract_video_id(url):
    match = re.search(
        r"(?:v=|youtu\.be/)([^&?/]+)",
        str(url),
    )

    if match:
        return match.group(1)

    return None


def main():
    api_key = os.getenv("YOUTUBE_API_KEY")

    if not api_key:
        raise ValueError(
            "YOUTUBE_API_KEY environment variable is not set."
        )

    df = pd.read_csv(SOURCE_PATH)

    video_ids = []

    title_map = {}

    for _, row in df.iterrows():
        video_id = extract_video_id(
            row["Video Link"]
        )

        if video_id:
            video_ids.append(video_id)

            title_map[video_id] = str(
                row["Video Title"]
            )

    # 去重但保持顺序
    video_ids = list(dict.fromkeys(video_ids))

    records = []

    # videos.list 一次最多查 50 个
    for start in range(
        0,
        len(video_ids),
        50,
    ):
        batch = video_ids[
            start:start + 50
        ]

        response = requests.get(
            VIDEOS_API,
            params={
                "part": "snippet,statistics,status",
                "id": ",".join(batch),
                "key": api_key,
            },
            timeout=30,
        )

        if response.status_code != 200:
            raise RuntimeError(
                f"YouTube API error: "
                f"{response.status_code}"
            )

        data = response.json()

        returned = {
            item["id"]: item
            for item in data.get(
                "items", []
            )
        }

        for video_id in batch:
            if video_id not in returned:
                records.append(
                    {
                        "video_id": video_id,
                        "video_title": (
                            title_map[video_id]
                        ),
                        "available": False,
                        "comment_count": None,
                        "view_count": None,
                    }
                )

                continue

            item = returned[video_id]

            stats = item.get(
                "statistics", {}
            )

            records.append(
                {
                    "video_id": video_id,
                    "video_title": (
                        item["snippet"]["title"]
                    ),
                    "available": True,
                    "comment_count": int(
                        stats.get(
                            "commentCount", 0
                        )
                    ),
                    "view_count": int(
                        stats.get(
                            "viewCount", 0
                        )
                    ),
                }
            )

        print(
            f"Checked "
            f"{min(start + 50, len(video_ids))}"
            f"/{len(video_ids)} videos"
        )

        time.sleep(0.1)

    result = pd.DataFrame(records)

    result.to_csv(
        r"data\video_population.csv",
        index=False,
    )

    available = result[
        result["available"]
    ].copy()

    counts = available[
        "comment_count"
    ].dropna()

    print()
    print("=" * 70)
    print("VIDEO POPULATION SUMMARY")
    print("=" * 70)

    print(
        f"Unique video IDs: "
        f"{len(result):,}"
    )

    print(
        f"Available videos: "
        f"{len(available):,}"
    )

    print(
        f"Unavailable videos: "
        f"{(~result['available']).sum():,}"
    )

    print()

    print(
        f"Total reported comments: "
        f"{int(counts.sum()):,}"
    )

    print(
        f"Mean comments/video: "
        f"{counts.mean():.2f}"
    )

    print(
        f"Median comments/video: "
        f"{counts.median():.2f}"
    )

    print()

    print("Comment-count percentiles:")

    print(
        counts.quantile(
            [
                0.25,
                0.50,
                0.75,
                0.90,
                0.95,
                0.99,
            ]
        )
    )

    print()

    print("Top 10 videos by comment count:")

    print(
        available[
            [
                "video_title",
                "comment_count",
            ]
        ]
        .sort_values(
            "comment_count",
            ascending=False,
        )
        .head(10)
        .to_string(index=False)
    )

    print()

    for cap in [
        100,
        300,
        500,
        1000,
    ]:
        estimated = (
            counts.clip(
                upper=cap
            )
            .sum()
        )

        print(
            f"Estimated corpus "
            f"with cap={cap}: "
            f"{int(estimated):,} comments"
        )


if __name__ == "__main__":
    main()