with contacts as (
    select * 
    from {{ ref('stg_contacts') }}
)
select * from contacts