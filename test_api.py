import requests

url = "http://127.0.0.1:5000/api/fuel-route/"
payload = {
    "start_location": "Los Angeles, CA",
    "end_location": "New York, NY"
}
headers = {
    "Content-Type": "application/json"
}

response = requests.post(url, json=payload, headers=headers)
print("Status:", response.status_code)
print("Response:", response.json())
