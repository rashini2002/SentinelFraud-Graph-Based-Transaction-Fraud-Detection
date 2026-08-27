{{ config(materialized='table') }}

-- fact_entity_edges.sql
--
-- Unifies int_shared_card_addr (Tier 2, weak signal) and int_shared_device
-- (Tier 1, high confidence) into a single edge list — this is what Day 9's
-- NetworkX graph construction loads directly. Each row is one edge between
-- two transactions that share an identity attribute.
--
-- edge_weight reflects confidence tier: device-based edges (Tier 1) are
-- weighted higher than card/addr-based edges (Tier 2), since DeviceInfo
-- sharing, while noisier than originally assumed (see Day 6 findings), is
-- still generally a stronger signal than card/addr overlap alone. This
-- weight is used later by community detection (Day 10) to favor
-- higher-confidence connections when forming clusters.

with card_addr_edges as (

    select
        transaction_id_a,
        transaction_id_b,
        shared_attribute_type,
        1.0 as edge_weight,
        is_true_ring_pair
    from {{ ref('int_shared_card_addr') }}

),

device_edges as (

    select
        transaction_id_a,
        transaction_id_b,
        shared_attribute_type,
        2.0 as edge_weight,
        is_true_ring_pair
    from {{ ref('int_shared_device') }}

),

unioned as (

    select * from card_addr_edges
    union all
    select * from device_edges

)

select
    row_number() over (order by transaction_id_a, transaction_id_b, shared_attribute_type) as edge_id,
    transaction_id_a,
    transaction_id_b,
    shared_attribute_type,
    edge_weight,
    is_true_ring_pair
from unioned