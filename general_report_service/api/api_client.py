import requests
from django.urls import reverse
from django.conf import settings  # Import the settings module

def get_options_from_index_service_api():
    try:
        base_url = settings.BASE_URL  # Replace with your actual setting name
        api_endpoint = reverse('index-api')  # Assuming 'index-api' is the name of your URL pattern

        # Construct the full URL
        full_url = f"{base_url}{api_endpoint}"

        response = requests.get(full_url)
        response.raise_for_status()
        options = response.json().get('options', [])
        return options
    except requests.RequestException as e:
        # Log the error details
        print(f"Error during API request: {e}")
        return []
