{{ config(materialized='table') }}

-- dim_device.sql
-- One row per distinct device_type + device_info combination.
-- Only rows with has_device_identity = true have meaningful values here;
-- transactions without device identity data link to no row in this
-- dimension (handled as a left join / null device_key in fact_transactions).

with distinct_devices as (

    select distinct
        device_type,
        device_info
    from {{ ref('stg_transactions') }}
    where has_device_identity = true

)

select
    dense_rank() over (order by device_type, device_info) as device_key,
    device_type,
    device_info
from distinct_devices