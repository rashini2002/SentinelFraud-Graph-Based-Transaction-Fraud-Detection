{{ config(materialized='table') }}

-- dim_card.sql
-- One row per distinct card1-card6 combination observed in the data.

with distinct_cards as (

    select distinct
        card1,
        card2,
        card3,
        card4,
        card5,
        card6
    from {{ ref('stg_transactions') }}
    where card1 is not null

)

select
    dense_rank() over (order by card1, card2, card3, card4, card5, card6) as card_key,
    card1,
    card2,
    card3,
    card4,
    card5,
    card6
from distinct_cards