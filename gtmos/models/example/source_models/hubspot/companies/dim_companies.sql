with companies as (
    select * 
    from {{ ref('stg_companies') }}
)
select * from companies