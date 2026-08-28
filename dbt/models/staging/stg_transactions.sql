with source as (

    select * from {{ source('raw', 'raw_transactions') }}

),

renamed as (

    select
        "TransactionID"::bigint                as transaction_id,
        "isFraud"::int                          as is_fraud,
        "TransactionDT"::bigint                 as transaction_dt,
        "TransactionAmt"::numeric               as transaction_amt,
        "ProductCD"::text                       as product_cd,

        -- primary linkage keys
        "card1"::text                           as card1,
        "card2"::text                           as card2,
        "card3"::text                           as card3,
        "card4"::text                           as card4,
        "card5"::text                           as card5,
        "card6"::text                           as card6,
        "addr1"::text                           as addr1,
        "addr2"::text                           as addr2,
        "P_emaildomain"::text                   as p_email_domain,

        -- identity / device linkage
        "DeviceType"::text                      as device_type,
        "DeviceInfo"::text                       as device_info,
        "has_device_identity"::boolean          as has_device_identity,

        -- ground truth — evaluation only, never a model feature
        "ring_id"::text                         as ring_id,
        "ring_tier"::text                       as ring_tier,

        -- sampled V-columns retained from Day 3 for classifier use
        -- (these were generated in the synthetic data but never selected
        -- into staging until this fix — see docs/DECISIONS.md Day 12)
        "V1"::numeric                           as v1,
        "V12"::numeric                          as v12,
        "V45"::numeric                          as v45,
        "V78"::numeric                          as v78,
        "V100"::numeric                         as v100,
        "V130"::numeric                         as v130,
        "V160"::numeric                         as v160,
        "V200"::numeric                         as v200,
        "V250"::numeric                         as v250,
        "V300"::numeric                         as v300

    from source

)

select * from renamed