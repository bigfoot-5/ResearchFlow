response = requests.post(
                    f"{backend_url}/api/v1/icp/triangulation",
                    json={"filter_condition": filter_condition}
                )
if response.status_code == 200:
    triangulation_data = response.json()