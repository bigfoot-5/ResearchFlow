select 
    cast(id as bigint), 
    cast(company_id as bigint),
    first_name,
    last_name, 
    job_title
    
from {{ source('hubspot_data', 'contacts') }}