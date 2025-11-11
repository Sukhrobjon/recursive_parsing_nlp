WITH engagement_events AS (
  SELECT
    user_pseudo_id,
    PARSE_DATE('%Y%m%d', event_date) AS event_dt,
    (SELECT SAFE_CAST(param.value.int_value AS INT64)
     FROM UNNEST(event_params) AS param
     WHERE param.key = 'engagement_time_msec') AS engagement_time_msec
  FROM `bigquery-public-data.ga4_obfuscated_sample_ecommerce.events_*`
  WHERE _TABLE_SUFFIX BETWEEN '20210101' AND '20210107'
),
user_engagement AS (
  SELECT
    user_pseudo_id,
    SUM(CASE WHEN event_dt BETWEEN DATE '2021-01-01' AND DATE '2021-01-07'
             THEN engagement_time_msec ELSE 0 END) AS engagement_7d,
    SUM(CASE WHEN event_dt BETWEEN DATE '2021-01-06' AND DATE '2021-01-07'
             THEN engagement_time_msec ELSE 0 END) AS engagement_2d
  FROM engagement_events
  WHERE engagement_time_msec IS NOT NULL
        AND engagement_time_msec > 0
  GROUP BY user_pseudo_id
)
SELECT COUNT(*) AS distinct_users
FROM user_engagement
WHERE engagement_7d > 0
  AND engagement_2d = 0