WITH engagement AS (
  SELECT
    user_pseudo_id,
    event_date,
    (SELECT ep.value.int_value
     FROM UNNEST(event_params) ep
     WHERE ep.key = 'engagement_time_msec') AS engagement_time_msec
  FROM `bigquery-public-data.ga4_obfuscated_sample_ecommerce.events_*`
  WHERE _TABLE_SUFFIX BETWEEN '20210101' AND '20210107'
    AND event_name = 'user_engagement'
),
engaged_7d AS (
  SELECT DISTINCT user_pseudo_id
  FROM engagement
  WHERE engagement_time_msec > 0
),
engaged_2d AS (
  SELECT DISTINCT user_pseudo_id
  FROM engagement
  WHERE engagement_time_msec > 0
    AND event_date BETWEEN '20210106' AND '20210107'
)
SELECT COUNT(*) AS distinct_users
FROM engaged_7d
WHERE user_pseudo_id NOT IN (SELECT user_pseudo_id FROM engaged_2d)