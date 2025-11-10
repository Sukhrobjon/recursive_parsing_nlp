WITH engagement AS (
  SELECT user_pseudo_id, event_date
  FROM (
    SELECT
      user_pseudo_id,
      event_date,
      (
        SELECT MAX(CAST(ep.value.int_value AS INT64))
        FROM UNNEST(event_params) AS ep
        WHERE ep.key = 'engagement_time_msec'
      ) AS engagement_time_msec
    FROM `bigquery-public-data.ga4_obfuscated_sample_ecommerce.events_20210101`
    UNION ALL
    SELECT
      user_pseudo_id,
      event_date,
      (
        SELECT MAX(CAST(ep.value.int_value AS INT64))
        FROM UNNEST(event_params) AS ep
        WHERE ep.key = 'engagement_time_msec'
      ) AS engagement_time_msec
    FROM `bigquery-public-data.ga4_obfuscated_sample_ecommerce.events_20210102`
    UNION ALL
    SELECT
      user_pseudo_id,
      event_date,
      (
        SELECT MAX(CAST(ep.value.int_value AS INT64))
        FROM UNNEST(event_params) AS ep
        WHERE ep.key = 'engagement_time_msec'
      ) AS engagement_time_msec
    FROM `bigquery-public-data.ga4_obfuscated_sample_ecommerce.events_20210103`
    UNION ALL
    SELECT
      user_pseudo_id,
      event_date,
      (
        SELECT MAX(CAST(ep.value.int_value AS INT64))
        FROM UNNEST(event_params) AS ep
        WHERE ep.key = 'engagement_time_msec'
      ) AS engagement_time_msec
    FROM `bigquery-public-data.ga4_obfuscated_sample_ecommerce.events_20210104`
    UNION ALL
    SELECT
      user_pseudo_id,
      event_date,
      (
        SELECT MAX(CAST(ep.value.int_value AS INT64))
        FROM UNNEST(event_params) AS ep
        WHERE ep.key = 'engagement_time_msec'
      ) AS engagement_time_msec
    FROM `bigquery-public-data.ga4_obfuscated_sample_ecommerce.events_20210105`
    UNION ALL
    SELECT
      user_pseudo_id,
      event_date,
      (
        SELECT MAX(CAST(ep.value.int_value AS INT64))
        FROM UNNEST(event_params) AS ep
        WHERE ep.key = 'engagement_time_msec'
      ) AS engagement_time_msec
    FROM `bigquery-public-data.ga4_obfuscated_sample_ecommerce.events_20210106`
    UNION ALL
    SELECT
      user_pseudo_id,
      event_date,
      (
        SELECT MAX(CAST(ep.value.int_value AS INT64))
        FROM UNNEST(event_params) AS ep
        WHERE ep.key = 'engagement_time_msec'
      ) AS engagement_time_msec
    FROM `bigquery-public-data.ga4_obfuscated_sample_ecommerce.events_20210107`
  )
  WHERE engagement_time_msec > 0
    AND user_pseudo_id IS NOT NULL
),
users_7d AS (
  SELECT DISTINCT user_pseudo_id
  FROM engagement
  WHERE event_date BETWEEN '20210101' AND '20210107'
),
users_2d AS (
  SELECT DISTINCT user_pseudo_id
  FROM engagement
  WHERE event_date BETWEEN '20210106' AND '20210107'
)
SELECT COUNT(*) AS distinct_pseudo_users
FROM users_7d
LEFT JOIN users_2d USING (user_pseudo_id)
WHERE users_2d.user_pseudo_id IS NULL