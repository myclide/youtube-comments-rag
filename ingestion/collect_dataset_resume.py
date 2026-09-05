import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests


# ============================================================
# CONFIG
# ============================================================

VIDEOS_PATH = (
    Path("data")
    / "video_population.csv"
)

OUTPUT_DIR = (
    Path("data")
    / "structured_comments"
    / "by_video"
)

STATE_PATH = (
    Path("data")
    / "structured_comments"
    / "collection_state.json"
)

COMMENTS_API = (
    "https://www.googleapis.com/"
    "youtube/v3/commentThreads"
)


# 每个视频最多保存多少条 UNIQUE 顶层评论
COMMENTS_PER_VIDEO_CAP = 500


# 目前仍然只测试 20 个视频。
# 确认 resume 正常之后再改成 None。
MAX_VIDEOS_THIS_RUN = None


REQUEST_TIMEOUT = 30

MAX_RETRIES = 3


# ============================================================
# Custom Exceptions
# ============================================================

class CommentsDisabled(Exception):
    pass


class VideoUnavailable(Exception):
    pass


class QuotaExceeded(Exception):
    pass


# ============================================================
# General Utilities
# ============================================================

def utc_now():
    """
    Return current UTC time as ISO string.
    """
    return datetime.now(
        timezone.utc
    ).isoformat()


def get_api_key():
    """
    Read YouTube API key from environment variable.
    """

    api_key = os.getenv(
        "YOUTUBE_API_KEY"
    )

    if not api_key:
        raise ValueError(
            "YOUTUBE_API_KEY environment "
            "variable is not set."
        )

    return api_key


# ============================================================
# State Management
# ============================================================

def load_state():
    """
    Load resumable collection state.

    completed:
        successfully collected videos

    skipped:
        videos we should not retry
        e.g. comments disabled / unavailable

    failed:
        temporary or unexpected failures;
        these will be retried next run
    """

    if not STATE_PATH.exists():
        return {
            "completed": {},
            "skipped": {},
            "failed": {},
        }

    with open(
        STATE_PATH,
        "r",
        encoding="utf-8",
    ) as file:
        return json.load(file)


def save_state(state):
    """
    Save state atomically.

    Write to temp file first,
    then replace the real state file.

    This reduces the chance of corrupting
    collection_state.json if the program
    stops during writing.
    """

    STATE_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temp_path = (
        STATE_PATH.with_suffix(
            ".tmp"
        )
    )

    with open(
        temp_path,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            state,
            file,
            ensure_ascii=False,
            indent=2,
        )

    os.replace(
        temp_path,
        STATE_PATH,
    )


# ============================================================
# YouTube API Request
# ============================================================

def request_json(
    session,
    api_key,
    video_id,
    page_token=None,
):
    """
    Request one page of top-level
    YouTube comment threads.
    """

    params = {
        "part": "snippet",
        "videoId": video_id,
        "maxResults": 100,
        "textFormat": "plainText",

        # We want a systematic recent-comment
        # sample rather than YouTube relevance
        # ranking.
        "order": "time",

        "key": api_key,
    }

    if page_token:
        params["pageToken"] = (
            page_token
        )

    for attempt in range(
        MAX_RETRIES
    ):

        try:
            response = session.get(
                COMMENTS_API,
                params=params,
                timeout=REQUEST_TIMEOUT,
            )

        except requests.RequestException as exc:

            if (
                attempt
                == MAX_RETRIES - 1
            ):
                raise RuntimeError(
                    f"Network error: {exc}"
                )

            wait_seconds = (
                2 ** attempt
            )

            time.sleep(
                wait_seconds
            )

            continue

        # ----------------------------
        # Success
        # ----------------------------

        if response.status_code == 200:
            return response.json()

        # ----------------------------
        # Parse YouTube API error
        # WITHOUT printing full URL
        # or exposing API key
        # ----------------------------

        try:
            error_data = (
                response.json()
            )

            error = (
                error_data.get(
                    "error",
                    {},
                )
            )

            message = error.get(
                "message",
                "Unknown API error",
            )

            errors = error.get(
                "errors",
                [],
            )

            if errors:
                reason = (
                    errors[0].get(
                        "reason",
                        "",
                    )
                )
            else:
                reason = ""

        except ValueError:

            message = (
                "Unknown API error"
            )

            reason = ""

        # ----------------------------
        # Known permanent conditions
        # ----------------------------

        if reason == "commentsDisabled":
            raise CommentsDisabled(
                "Comments are disabled."
            )

        if (
            response.status_code == 404
            or reason
            in {
                "videoNotFound",
                "notFound",
            }
        ):
            raise VideoUnavailable(
                "Video unavailable."
            )

        # ----------------------------
        # Quota
        # ----------------------------

        if reason in {
            "quotaExceeded",
            "dailyLimitExceeded",
        }:
            raise QuotaExceeded(
                "YouTube API quota exceeded."
            )

        # ----------------------------
        # Temporary server/rate errors
        # Retry using exponential backoff
        # ----------------------------

        if response.status_code in {
            429,
            500,
            502,
            503,
            504,
        }:

            if (
                attempt
                < MAX_RETRIES - 1
            ):
                wait_seconds = (
                    2 ** attempt
                )

                time.sleep(
                    wait_seconds
                )

                continue

        # ----------------------------
        # Other API errors
        # ----------------------------

        raise RuntimeError(
            f"YouTube API error "
            f"{response.status_code}: "
            f"{message}"
        )

    raise RuntimeError(
        "Request failed after retries."
    )


# ============================================================
# Collect One Video
# ============================================================

def collect_video(
    session,
    api_key,
    video_id,
    video_title,
):
    """
    Collect up to COMMENTS_PER_VIDEO_CAP
    UNIQUE top-level comments.

    Deduplication happens during pagination,
    not after reaching the cap.
    """

    records = []

    # Track IDs during collection.
    # This ensures cap=500 means:
    # 500 UNIQUE comments,
    # not 500 API records.
    seen_comment_ids = set()

    duplicate_count = 0

    page_token = None

    pages = 0

    collection_time = utc_now()

    while True:

        data = request_json(
            session=session,
            api_key=api_key,
            video_id=video_id,
            page_token=page_token,
        )

        pages += 1

        items = data.get(
            "items",
            [],
        )

        for item in items:

            thread_snippet = (
                item["snippet"]
            )

            top_comment = (
                thread_snippet[
                    "topLevelComment"
                ]
            )

            comment_id = (
                top_comment["id"]
            )

            # --------------------------------
            # Online ID deduplication
            # --------------------------------

            if (
                comment_id
                in seen_comment_ids
            ):
                duplicate_count += 1
                continue

            seen_comment_ids.add(
                comment_id
            )

            snippet = (
                top_comment["snippet"]
            )

            # Prefer original/plain text
            comment_text = (
                snippet.get(
                    "textOriginal"
                )
                or snippet.get(
                    "textDisplay"
                )
                or ""
            ).strip()

            # Do not save empty comments
            if not comment_text:
                continue

            record = {
                "video_id": (
                    video_id
                ),

                "video_title": (
                    video_title
                ),

                "source_video_url": (
                    "https://www.youtube.com/"
                    f"watch?v={video_id}"
                ),

                "thread_id": (
                    item["id"]
                ),

                "comment_id": (
                    comment_id
                ),

                "author": (
                    snippet.get(
                        "authorDisplayName"
                    )
                ),

                "comment_text": (
                    comment_text
                ),

                "like_count": (
                    snippet.get(
                        "likeCount",
                        0,
                    )
                ),

                "published_at": (
                    snippet.get(
                        "publishedAt"
                    )
                ),

                "updated_at": (
                    snippet.get(
                        "updatedAt"
                    )
                ),

                "reply_count": (
                    thread_snippet.get(
                        "totalReplyCount",
                        0,
                    )
                ),

                "collection_order": (
                    "time"
                ),

                "sample_cap": (
                    COMMENTS_PER_VIDEO_CAP
                ),

                "collected_at": (
                    collection_time
                ),
            }

            records.append(
                record
            )

            # --------------------------------
            # Important:
            #
            # records now contains only
            # UNIQUE comment IDs.
            # --------------------------------

            if (
                len(records)
                >= COMMENTS_PER_VIDEO_CAP
            ):
                break

        # We already reached cap
        if (
            len(records)
            >= COMMENTS_PER_VIDEO_CAP
        ):
            break

        # Get next page
        page_token = data.get(
            "nextPageToken"
        )

        # No more comments available
        if not page_token:
            break

    df = pd.DataFrame(
        records
    )

    return (
        df,
        pages,
        duplicate_count,
    )


# ============================================================
# Save Per-video Dataset
# ============================================================

def save_video_file(
    video_id,
    df,
):
    """
    Save each video independently.

    This prevents one failed future video
    from corrupting all previously collected
    data.
    """

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = (
        OUTPUT_DIR
        / f"{video_id}.jsonl"
    )

    temp_path = (
        OUTPUT_DIR
        / f"{video_id}.tmp"
    )

    df.to_json(
        temp_path,
        orient="records",
        lines=True,
        force_ascii=False,
    )

    os.replace(
        temp_path,
        output_path,
    )

    return output_path


# ============================================================
# Main Collection Pipeline
# ============================================================

def main():

    api_key = get_api_key()

    # --------------------------------------------------------
    # Load previously audited video population
    # --------------------------------------------------------

    population = pd.read_csv(
        VIDEOS_PATH
    )

    # video_population.csv may read bool values
    # as Python bool or strings depending on pandas.
    available_mask = (
        population["available"]
        .astype(str)
        .str.lower()
        .eq("true")
    )

    population = (
        population[
            available_mask
        ]
        .copy()
        .reset_index(drop=True)
    )

    # --------------------------------------------------------
    # Load resume state
    # --------------------------------------------------------

    state = load_state()

    completed = set(
        state[
            "completed"
        ].keys()
    )

    skipped = set(
        state[
            "skipped"
        ].keys()
    )

    # Failed videos are intentionally
    # NOT excluded.
    #
    # They will automatically retry
    # in future runs.

    candidates = population[
        ~population[
            "video_id"
        ].isin(
            completed | skipped
        )
    ].copy()

    # --------------------------------------------------------
    # Testing limit
    # --------------------------------------------------------

    if (
        MAX_VIDEOS_THIS_RUN
        is not None
    ):
        candidates = (
            candidates.head(
                MAX_VIDEOS_THIS_RUN
            )
        )

    # --------------------------------------------------------
    # Starting summary
    # --------------------------------------------------------

    print("=" * 70)

    print(
        "RESUMABLE COLLECTION"
    )

    print("=" * 70)

    print(
        f"Available video population: "
        f"{len(population)}"
    )

    print(
        f"Previously completed: "
        f"{len(completed)}"
    )

    print(
        f"Previously skipped: "
        f"{len(skipped)}"
    )

    print(
        f"Previously failed: "
        f"{len(state['failed'])}"
    )

    print(
        f"Videos scheduled this run: "
        f"{len(candidates)}"
    )

    print()

    # --------------------------------------------------------
    # Session + run counters
    # --------------------------------------------------------

    session = requests.Session()

    run_comments = 0

    run_pages = 0

    run_completed = 0

    run_duplicates = 0

    run_skipped = 0

    run_failed = 0

    # --------------------------------------------------------
    # Collection Loop
    # --------------------------------------------------------

    try:

        for position, row in enumerate(
            candidates.itertuples(
                index=False
            ),
            start=1,
        ):

            video_id = str(
                row.video_id
            )

            video_title = str(
                row.video_title
            )

            print("-" * 70)

            print(
                f"[{position}/"
                f"{len(candidates)}] "
                f"{video_title}"
            )

            # ---------------------------------------------
            # Collect one video
            # ---------------------------------------------

            try:

                (
                    df,
                    pages,
                    duplicates,
                ) = collect_video(
                    session=session,
                    api_key=api_key,
                    video_id=video_id,
                    video_title=video_title,
                )

            # ---------------------------------------------
            # Comments disabled
            # ---------------------------------------------

            except CommentsDisabled:

                print(
                    "SKIP: comments disabled"
                )

                state[
                    "skipped"
                ][video_id] = {
                    "title": (
                        video_title
                    ),

                    "reason": (
                        "comments_disabled"
                    ),

                    "timestamp": (
                        utc_now()
                    ),
                }

                # Remove old failed status
                state[
                    "failed"
                ].pop(
                    video_id,
                    None,
                )

                run_skipped += 1

                save_state(
                    state
                )

                continue

            # ---------------------------------------------
            # Video disappeared since population audit
            # ---------------------------------------------

            except VideoUnavailable:

                print(
                    "SKIP: video unavailable"
                )

                state[
                    "skipped"
                ][video_id] = {
                    "title": (
                        video_title
                    ),

                    "reason": (
                        "video_unavailable"
                    ),

                    "timestamp": (
                        utc_now()
                    ),
                }

                state[
                    "failed"
                ].pop(
                    video_id,
                    None,
                )

                run_skipped += 1

                save_state(
                    state
                )

                continue

            # ---------------------------------------------
            # Daily quota used up
            # ---------------------------------------------

            except QuotaExceeded:

                print()

                print(
                    "STOP: YouTube API "
                    "quota exceeded."
                )

                print(
                    "All completed progress "
                    "has already been saved."
                )

                save_state(
                    state
                )

                return

            # ---------------------------------------------
            # Unexpected / temporary failure
            # ---------------------------------------------

            except Exception as exc:

                print(
                    f"FAILED: {exc}"
                )

                state[
                    "failed"
                ][video_id] = {
                    "title": (
                        video_title
                    ),

                    "error": (
                        str(exc)
                    ),

                    "timestamp": (
                        utc_now()
                    ),
                }

                run_failed += 1

                save_state(
                    state
                )

                # Do NOT put this video into skipped.
                # It will retry next run.
                continue

            # ---------------------------------------------
            # Save successful result
            # ---------------------------------------------

            output_path = (
                save_video_file(
                    video_id,
                    df,
                )
            )

            state[
                "completed"
            ][video_id] = {
                "title": (
                    video_title
                ),

                "comments": (
                    int(len(df))
                ),

                "pages": (
                    int(pages)
                ),

                "duplicate_api_records": (
                    int(duplicates)
                ),

                "output": (
                    str(output_path)
                ),

                "timestamp": (
                    utc_now()
                ),
            }

            # If it previously failed,
            # successful completion clears it.
            state[
                "failed"
            ].pop(
                video_id,
                None,
            )

            # Save immediately after each video
            save_state(
                state
            )

            # ---------------------------------------------
            # Update run statistics
            # ---------------------------------------------

            run_completed += 1

            run_comments += (
                len(df)
            )

            run_pages += (
                pages
            )

            run_duplicates += (
                duplicates
            )

            # ---------------------------------------------
            # Terminal output
            # ---------------------------------------------

            print(
                f"Collected: "
                f"{len(df)}"
            )

            print(
                f"Pages: "
                f"{pages}"
            )

            print(
                "Duplicate API records "
                f"skipped: {duplicates}"
            )

            print(
                f"Saved: "
                f"{output_path}"
            )

            # Mild delay between videos
            time.sleep(
                0.1
            )

    # --------------------------------------------------------
    # Ctrl+C support
    # --------------------------------------------------------

    except KeyboardInterrupt:

        print()
        print()

        print(
            "Collection interrupted "
            "by user."
        )

        print(
            "Completed videos have "
            "already been saved."
        )

        save_state(
            state
        )

        return

    # --------------------------------------------------------
    # Always close HTTP session
    # --------------------------------------------------------

    finally:

        session.close()

    # --------------------------------------------------------
    # Run Summary
    # --------------------------------------------------------

    print()

    print("=" * 70)

    print(
        "RUN SUMMARY"
    )

    print("=" * 70)

    print(
        f"Completed this run: "
        f"{run_completed}"
    )

    print(
        f"Skipped this run: "
        f"{run_skipped}"
    )

    print(
        f"Failed this run: "
        f"{run_failed}"
    )

    print(
        f"Comments this run: "
        f"{run_comments:,}"
    )

    print(
        f"API pages this run: "
        f"{run_pages:,}"
    )

    print(
        "Duplicate API records "
        f"skipped this run: "
        f"{run_duplicates:,}"
    )

    print()

    print(
        f"Total completed: "
        f"{len(state['completed'])}"
    )

    print(
        f"Total skipped: "
        f"{len(state['skipped'])}"
    )

    print(
        f"Currently failed: "
        f"{len(state['failed'])}"
    )


if __name__ == "__main__":
    main()