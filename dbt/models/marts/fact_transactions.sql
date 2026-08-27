{{ config(materialized='table') }}

-- fact_transactions.sql
-- Grain: one row per transaction. Joins to dim_date, dim_card, dim_device.
-- ring_id / ring_tier are retained here as ground-truth evaluation fields
-- ONLY — see docs/DECISIONS.md. They must never be used as classifier
-- features (Day 12-13) or as graph-construction inputs (Day 9-10).

with txn as (

    select * from {{ ref('stg_transactions') }}

),

dates as (

    select * from {{ ref('dim_date') }}

),

cards as (

    select * from {{ ref('dim_card') }}

),

devices as (

    select * from {{ ref('dim_device') }}

),

reference as (

    select date '2017-12-01' as reference_date

)

select
    txn.transaction_id,
    d.date_key,
    c.card_key,
    dev.device_key,
    txn.transaction_amt,
    txn.product_cd,
    txn.p_email_domain,
    txn.is_fraud,
    txn.has_device_identity,
    txn.ring_id,
    txn.ring_tier
from txn
left join dates d
    on d.calendar_date = (select reference_date from reference) + (txn.transaction_dt / 86400)::int
left join cards c
    on c.card1 = txn.card1
    and coalesce(c.card2, '') = coalesce(txn.card2, '')
    and coalesce(c.card3, '') = coalesce(txn.card3, '')
    and coalesce(c.card4, '') = coalesce(txn.card4, '')
    and coalesce(c.card5, '') = coalesce(txn.card5, '')
    and coalesce(c.card6, '') = coalesce(txn.card6, '')
left join devices dev
    on coalesce(dev.device_type, '') = coalesce(txn.device_type, '')
    and coalesce(dev.device_info, '') = coalesce(txn.device_info, '')
    and txn.has_device_identity = true