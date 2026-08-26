{{
  config(
    materialized='table'
  )
}}

-- int_shared_card_addr.sql
--
-- Builds Tier 2 (weak-signal) linkage edges: pairs of transactions sharing
-- both card1 AND addr1. This combination is used (rather than either alone)
-- because card1 or addr1 individually are common enough that many unrelated
-- transactions coincidentally share one — requiring both together is a much
-- stronger, more specific signal of a genuine relationship.
--
-- Group-size capping: groups larger than {{ var('max_linkage_group_size') }}
-- are excluded entirely, not just truncated. A very large group is more
-- likely a common demographic/regional pattern (e.g. a popular billing
-- address shared by an apartment complex or a card-issuing bank's default
-- test value) than a real fraud ring, and including it would flood the
-- graph with a low-value fully-connected clique.

with base as (

    select
        transaction_id,
        card1,
        addr1,
        is_fraud,
        ring_id,
        ring_tier
    from {{ ref('stg_transactions') }}
    where card1 is not null
      and addr1 is not null

),

group_sizes as (

    select
        card1,
        addr1,
        count(*) as group_size
    from base
    group by card1, addr1

),

eligible_groups as (

    select card1, addr1
    from group_sizes
    where group_size between 2 and {{ var('max_linkage_group_size') }}

),

eligible_transactions as (

    select b.*
    from base b
    inner join eligible_groups g
        on b.card1 = g.card1
        and b.addr1 = g.addr1

),

pairs as (

    select
        a.transaction_id as transaction_id_a,
        b.transaction_id as transaction_id_b,
        a.card1,
        a.addr1,
        'card_addr' as shared_attribute_type,
        -- true ring pair only if both members belong to the SAME injected
        -- ring — used later for evaluation only, never as a model feature
        case
            when a.ring_id is not null
                and a.ring_id = b.ring_id
            then true
            else false
        end as is_true_ring_pair
    from eligible_transactions a
    inner join eligible_transactions b
        on a.card1 = b.card1
        and a.addr1 = b.addr1
        and a.transaction_id < b.transaction_id  -- avoid duplicate symmetric pairs and self-pairs

)

select * from pairs