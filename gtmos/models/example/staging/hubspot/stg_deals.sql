select 
    cast(id as bigint),
    cast(company_id as bigint),
    company_name,
    deal_name,
    amount,
    dealstage,
    closedate,
    hs_is_closed_lost,
    hs_is_closed,
    hs_is_closed_won,
    hs_analytics_source, 
    closed_lost_reason,
    closed_won_reason,
    createdate,
    days_to_close
from {{ source('hubspot_data', 'deals') }}