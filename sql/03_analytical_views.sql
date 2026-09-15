-- ============================================================
-- Analytical Views
-- Healthcare Operations & Patient Outcome Intelligence
-- ============================================================

-- View: Prior healthcare utilization and readmission outcomes

CREATE OR REPLACE VIEW vw_utilization_summary AS

WITH utilization_groups AS (
    SELECT
        CASE
            WHEN total_prior_visits = 0 THEN '0'
            WHEN total_prior_visits BETWEEN 1 AND 2 THEN '1-2'
            WHEN total_prior_visits BETWEEN 3 AND 5 THEN '3-5'
            ELSE '6+'
        END AS utilization_group,

        time_in_hospital,
        readmitted_30

    FROM encounters
)

SELECT
    utilization_group,

    COUNT(*) AS encounters,

    ROUND(
        AVG(time_in_hospital),
        2
    ) AS avg_length_of_stay,

    SUM(readmitted_30) AS readmissions,

    ROUND(
        AVG(readmitted_30) * 100,
        2
    ) AS readmission_rate_pct

FROM utilization_groups

GROUP BY utilization_group;

-- For Testing
SELECT *
FROM vw_utilization_summary;

-- ============================================================
-- View: Hospital operations by admission type
-- ============================================================

CREATE OR REPLACE VIEW vw_admission_summary AS

SELECT
    COALESCE(admission_type, 'Unknown') AS admission_type,

    COUNT(*) AS encounters,

    ROUND(
        AVG(time_in_hospital),
        2
    ) AS avg_length_of_stay,

    SUM(readmitted_30) AS readmissions,

    ROUND(
        AVG(readmitted_30) * 100,
        2
    ) AS readmission_rate_pct

FROM encounters

GROUP BY COALESCE(admission_type, 'Unknown');

-- Test view
SELECT *
FROM vw_admission_summary
ORDER BY readmission_rate_pct DESC;