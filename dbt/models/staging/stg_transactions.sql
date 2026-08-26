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
        "ring_tier"::text                       as ring_tier

    from source

)

select * from renamed