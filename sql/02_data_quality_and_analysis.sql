-- ============================================================
-- Healthcare Operations & Patient Outcome Intelligence
-- SQL Data Quality and Analysis
-- ============================================================

-- 1. Verify total encounters
SELECT COUNT(*) AS encounter_count
FROM encounters;


-- 2. Verify unique encounters and patients
SELECT
    COUNT(*) AS total_rows,
    COUNT(DISTINCT encounter_id) AS unique_encounters,
    COUNT(DISTINCT patient_nbr) AS unique_patients
FROM encounters;


-- 3. Check important fields for missing values
SELECT
    COUNT(*) AS total_rows,
    COUNT(*) FILTER (WHERE readmitted_30 IS NULL) AS missing_target,
    COUNT(*) FILTER (WHERE age_numeric IS NULL) AS missing_age,
    COUNT(*) FILTER (WHERE gender IS NULL) AS missing_gender,
    COUNT(*) FILTER (WHERE admission_type IS NULL) AS missing_admission_type,
    COUNT(*) FILTER (WHERE time_in_hospital IS NULL) AS missing_length_of_stay
FROM encounters;


-- 4. 30-day readmission distribution
SELECT
    readmitted_30,
    COUNT(*) AS encounter_count,
    ROUND(
        COUNT(*) * 100.0 / SUM(COUNT(*)) OVER (),
        2
    ) AS percentage
FROM encounters
GROUP BY readmitted_30
ORDER BY readmitted_30;


-- 5. Prior inpatient utilization vs. readmission
SELECT
    prior_inpatient_visit,
    COUNT(*) AS encounters,
    SUM(readmitted_30) AS readmissions,
    ROUND(
        AVG(readmitted_30) * 100,
        2
    ) AS readmission_rate_pct
FROM encounters
GROUP BY prior_inpatient_visit
ORDER BY prior_inpatient_visit;


-- 6. Prior healthcare utilization groups vs. readmission
SELECT
    CASE
        WHEN total_prior_visits = 0 THEN '0'
        WHEN total_prior_visits BETWEEN 1 AND 2 THEN '1-2'
        WHEN total_prior_visits BETWEEN 3 AND 5 THEN '3-5'
        ELSE '6+'
    END AS utilization_group,

    COUNT(*) AS encounters,

    SUM(readmitted_30) AS readmissions,

    ROUND(
        AVG(readmitted_30) * 100,
        2
    ) AS readmission_rate_pct

FROM encounters

GROUP BY
    CASE
        WHEN total_prior_visits = 0 THEN '0'
        WHEN total_prior_visits BETWEEN 1 AND 2 THEN '1-2'
        WHEN total_prior_visits BETWEEN 3 AND 5 THEN '3-5'
        ELSE '6+'
    END

ORDER BY
    CASE
        WHEN total_prior_visits = 0 THEN 1
        WHEN total_prior_visits BETWEEN 1 AND 2 THEN 2
        WHEN total_prior_visits BETWEEN 3 AND 5 THEN 3
        ELSE 4
    END;