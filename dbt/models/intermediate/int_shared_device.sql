{{
  config(
    materialized='table'
  )
}}

-- int_shared_device.sql
--
-- Builds Tier 1 (high-confidence) linkage edges: pairs of transactions
-- sharing the same DeviceInfo value. Restricted to rows where
-- has_device_identity = true (~20% of the dataset, per Day 2 profiling).
-- DeviceInfo is a high-cardinality field (1,786 unique values in the
-- original profiling), so device sharing is a much stronger signal than
-- card/addr sharing alone — two unrelated people are unlikely to
-- coincidentally report the exact same device fingerprint string.
--
-- Same group-size capping logic as int_shared_card_addr.sql applies here,
-- using the same {{ var('max_linkage_group_size') }} threshold, though in
-- practice device-based groups are expected to be much smaller given the
-- high cardinality of DeviceInfo.

with base as (

    select
        transaction_id,
        device_info,
        is_fraud,
        ring_id,
        ring_tier
    from {{ ref('stg_transactions') }}
    where has_device_identity = true
      and device_info is not null

),

group_sizes as (

    select
        device_info,
        count(*) as group_size
    from base
    group by device_info

),

eligible_groups as (

    select device_info
    from group_sizes
    where group_size between 2 and {{ var('max_linkage_group_size') }}

),

eligible_transactions as (

    select b.*
    from base b
    inner join eligible_groups g
        on b.device_info = g.device_info

),

pairs as (

    select
        a.transaction_id as transaction_id_a,
        b.transaction_id as transaction_id_b,
        a.device_info,
        'device' as shared_attribute_type,
        case
            when a.ring_id is not null
                and a.ring_id = b.ring_id
            then true
            else false
        end as is_true_ring_pair
    from eligible_transactions a
    inner join eligible_transactions b
        on a.device_info = b.device_info
        and a.transaction_id < b.transaction_id

)

select * from pairs