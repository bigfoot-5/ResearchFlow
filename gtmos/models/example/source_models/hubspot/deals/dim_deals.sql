with deals as (
    select *
    from {{ ref('stg_deals') }}
)
select * from deals