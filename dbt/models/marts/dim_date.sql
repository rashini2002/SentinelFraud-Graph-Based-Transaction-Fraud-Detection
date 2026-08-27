{{ config(materialized='table') }}

-- dim_date.sql
--
-- IEEE-CIS's TransactionDT is a relative offset in seconds from an
-- unspecified reference point, NOT a real calendar timestamp. Per common
-- practice with this dataset (documented in the original competition
-- discussion), we anchor it to an arbitrary reference date purely to get
-- usable calendar attributes (day-of-week, month) for BI purposes. This is
-- a synthetic calendar, not a claim about when these transactions "really"
-- happened — documented here to avoid any confusion later.

with reference as (

    select date '2017-12-01' as reference_date

),

transaction_dates as (

    select distinct
        (select reference_date from reference) + (transaction_dt / 86400)::int as calendar_date
    from {{ ref('stg_transactions') }}

)

select
    dense_rank() over (order by calendar_date) as date_key,
    calendar_date,
    extract(year from calendar_date) as year,
    extract(month from calendar_date) as month,
    extract(day from calendar_date) as day,
    extract(dow from calendar_date) as day_of_week,
    to_char(calendar_date, 'Day') as day_name,
    case when extract(dow from calendar_date) in (0, 6) then true else false end as is_weekend
from transaction_dates