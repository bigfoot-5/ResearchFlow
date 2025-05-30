select 
    cast(id as bigint) as company_id,
    industry,
    cast(numberofemployees as bigint)
from {{ source('hubspot_data', 'companies') }}