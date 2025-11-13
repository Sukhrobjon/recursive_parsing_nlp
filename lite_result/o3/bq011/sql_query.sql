WITH seven_day_engaged AS (
  SELECT DISTINCT user_pseudo_id
  FROM `bigquery-public-data.ga4_obfuscated_sample_ecommerce.events_202101*`,
  UNNEST(event_params) AS ep
  WHERE _TABLE_SUFFIX BETWEEN '20210101' AND '20210107'
    AND ep.key = 'engagement_time_msec'
    AND (
      (ep.value.int_value IS NOT NULL AND ep.value.int_value > 0) OR
      (ep.value.double_value IS NOT NULL AND ep.value.double_value > 0) OR
      (ep.value.float_value IS NOT NULL AND ep.value.float_value > 0)
    )
),
two_day_engaged AS (
  SELECT DISTINCT user_pseudo_id
  FROM `bigquery-public-data.ga4_obfuscated_sample_ecommerce.events_202101*`,
  UNNEST(event_params) AS ep
  WHERE _TABLE_SUFFIX BETWEEN '20210106' AND '20210107'
    AND ep.key = 'engagement_time_msec'
    AND (
      (ep.value.int_value IS NOT NULL AND ep.value.int_value > 0) OR
      (ep.value.double_value IS NOT NULL AND ep.value.double_value > 0) OR
      (ep.value.float_value IS NOT NULL AND ep.value.float_value > 0)
    )
)
SELECT COUNT(*) AS distinct_users
FROM seven_day_engaged
WHERE user_pseudo_id NOT IN (SELECT user_pseudo_id FROM two_day_engaged)