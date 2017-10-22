--rDepression_all.sql
/* a Google BigQuery select statement that pulls all comments from the history
of the r/depression subreddit */

SELECT
    *
    FROM
       [fh-bigquery:reddit_comments.all]
WHERE subreddit='depression'
