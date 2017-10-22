-- rDepression_comments_2017_06.sql
/* contains my query for Google BigQuery to get comments for r/depression from
2017-06 */
SELECT
    body,
    author,
    created_utc,
    parent_id,
    score,
    subreddit,
    subreddit_id
    FROM
       [fh-bigquery:reddit_comments.2017_06],
WHERE subreddit='depression'
