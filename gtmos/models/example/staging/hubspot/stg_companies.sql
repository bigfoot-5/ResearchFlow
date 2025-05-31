select 
    cast(id as bigint) as company_id,
    industry,
    country,
    cast(numberofemployees as bigint) as numberofemployees,
    case
        when cast(numberofemployees as bigint) between 1 and 1000 then '1-1000'
        when cast(numberofemployees as bigint) between 1001 and 2000 then '1001-2000'
        when cast(numberofemployees as bigint) between 2001 and 3000 then '2001-3000'
        when cast(numberofemployees as bigint) between 3001 and 4000 then '3001-4000'
        when cast(numberofemployees as bigint) between 4001 and 5000 then '4001-5000'
        when cast(numberofemployees as bigint) > 5000 then '5000+'
        else null
    end as employee_bucket
from {{ source('hubspot_data', 'companies') }}