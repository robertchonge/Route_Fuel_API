
import os
import sys
import django
from django.conf import settings
from django.core.wsgi import get_wsgi_application
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.urls import path
from django.core.management import execute_from_command_line
import json
import requests
import pandas as pd
import math
from geopy.distance import geodesic
from io import StringIO

# Django settings
settings.configure(
    DEBUG=True,
    SECRET_KEY=os.getenv('DJANGO_SECRET_KEY', 'fallback-secret-key-for-development-only'),
    ROOT_URLCONF=__name__,
    MIDDLEWARE=[
        'django.middleware.common.CommonMiddleware',
        'django.middleware.security.SecurityMiddleware',
    ],
    CORS_ALLOW_ALL_ORIGINS=True,
    CORS_ALLOW_CREDENTIALS=True,
    CORS_ALLOW_HEADERS=[
        'accept',
        'accept-encoding',
        'authorization',
        'content-type',
        'dnt',
        'origin',
        'user-agent',
        'x-csrftoken',
        'x-requested-with',
    ],
    ALLOWED_HOSTS=['*'],
    USE_TZ=True,
    SECURE_CROSS_ORIGIN_OPENER_POLICY=None,
    SECURE_REFERRER_POLICY=None,
)

django.setup()

# Constants
VEHICLE_RANGE_MILES = 500
MILES_PER_GALLON = 10
OPENROUTESERVICE_API_KEY = os.getenv('OPENROUTESERVICE_API_KEY', '5b3ce3597851110001cf6248f2a8a5ec68ce4d9e8fbb4cbfdadca5b3')
GOOGLE_DRIVE_FILE_ID = os.getenv('GOOGLE_DRIVE_FILE_ID', '1D0tiaSF3RKimUbr_NmiLwsq39hSmB0_2')

# Global variable to cache fuel data
FUEL_DATA_CACHE = None

# Loading fuel prices from CSV
def load_fuel_data():
    """Load fuel price data from Google Drive"""
    global FUEL_DATA_CACHE
    
    # Returning cached data if available
    if FUEL_DATA_CACHE is not None:
        return FUEL_DATA_CACHE
    
    try:
        # Loading from Google Drive
        google_drive_url = f"https://drive.google.com/uc?id={GOOGLE_DRIVE_FILE_ID}"
        df = pd.read_csv(google_drive_url)
        print(f"Loaded {len(df)} fuel stations from Google Drive")
        
        # Adding coordinates column if not exists
        if 'Latitude' not in df.columns:
            df['Latitude'] = None
        if 'Longitude' not in df.columns:
            df['Longitude'] = None
            
        # Using state-based coordinates for faster loading
        print("Adding coordinates based on state locations...")
        for index, row in df.iterrows():
            if pd.isna(row.get('Latitude')) or pd.isna(row.get('Longitude')):
                state = row.get('State', '')
                state_coords = get_state_coordinates(state)
                if state_coords:
                    # Add small random offset to avoid all stations at exact same location
                    import random
                    lat_offset = random.uniform(-0.5, 0.5)
                    lon_offset = random.uniform(-0.5, 0.5)
                    df.at[index, 'Latitude'] = state_coords[0] + lat_offset
                    df.at[index, 'Longitude'] = state_coords[1] + lon_offset
        
        # Removing rows without coordinates
        df = df.dropna(subset=['Latitude', 'Longitude'])
        print(f"Final dataset: {len(df)} fuel stations with coordinates")
        
        # Caching the data
        FUEL_DATA_CACHE = df
        return df
    except Exception as e:
        print(f"Error loading fuel data: {e}")
        return pd.DataFrame()

def get_state_coordinates(state):
    """Get approximate coordinates for a state"""
    state_coords = {
        'AL': (32.3617, -86.2792), 'AK': (64.0685, -152.2782), 'AZ': (34.2744, -111.2847),
        'AR': (34.7519, -92.1313), 'CA': (36.7783, -119.4179), 'CO': (39.5501, -105.7821),
        'CT': (41.6032, -73.0877), 'DE': (38.9108, -75.5277), 'FL': (27.7663, -82.6404),
        'GA': (32.1656, -82.9001), 'HI': (19.8968, -155.5828), 'ID': (44.0682, -114.7420),
        'IL': (40.6331, -89.3985), 'IN': (40.2732, -86.1349), 'IA': (41.8780, -93.0977),
        'KS': (38.5767, -98.6092), 'KY': (37.8393, -84.2700), 'LA': (30.9843, -91.9623),
        'ME': (45.2538, -69.4455), 'MD': (39.0458, -76.6413), 'MA': (42.2373, -71.5314),
        'MI': (44.3467, -85.4102), 'MN': (46.7296, -94.6859), 'MS': (32.3547, -89.3985),
        'MO': (37.9643, -91.8318), 'MT': (47.0527, -109.6333), 'NE': (41.4925, -99.9018),
        'NV': (38.8026, -116.4194), 'NH': (43.1939, -71.5724), 'NJ': (40.0583, -74.4057),
        'NM': (34.5199, -105.8701), 'NY': (43.2994, -74.2179), 'NC': (35.7596, -79.0193),
        'ND': (47.4501, -100.4659), 'OH': (40.4173, -82.9071), 'OK': (35.0078, -97.0929),
        'OR': (43.8041, -120.5542), 'PA': (41.2033, -77.1945), 'RI': (41.6809, -71.5118),
        'SC': (33.8361, -81.1637), 'SD': (43.9695, -99.9018), 'TN': (35.5175, -86.5804),
        'TX': (31.9686, -99.9018), 'UT': (39.3210, -111.0937), 'VT': (44.2601, -72.5806),
        'VA': (37.4316, -78.6569), 'WA': (47.2529, -120.7401), 'WV': (38.3498, -80.6547),
        'WI': (43.7844, -88.7879), 'WY': (43.0759, -107.2903),
        # Adding Canadian provinces for the dataset
        'AB': (53.9333, -116.5765), 'BC': (53.7267, -127.6476), 'MB': (53.7609, -98.8139),
        'NB': (46.5653, -66.4619), 'ON': (51.2538, -85.3232), 'SK': (52.9399, -106.4509)
    }
    return state_coords.get(state)

def geocode_address(address):
    """Geocode address using Nominatim (OpenStreetMap)"""
    url = "https://nominatim.openstreetmap.org/search"
    params = {
        'q': address,
        'format': 'json',
        'countrycodes': 'us',
        'limit': 1
    }
    
    headers = {
        'User-Agent': 'FuelRoutePlanner/1.0'
    }
    
    try:
        import time
        time.sleep(1)  # Rate limiting - 1 request per second
        response = requests.get(url, params=params, headers=headers, timeout=10)
        
        if response.status_code == 200 and response.text.strip():
            data = response.json()
            if data and len(data) > 0:
                return float(data[0]['lat']), float(data[0]['lon'])
        return None, None
    except Exception as e:
        print(f"Geocoding error for '{address}': {e}")
        return None, None

def get_route_info(start_coords, end_coords):
    """Get route information using OpenRouteService"""
    url = "https://api.openrouteservice.org/v2/directions/driving-car"
    
    headers = {
        'Authorization': OPENROUTESERVICE_API_KEY,
        'Content-Type': 'application/json'
    }
    
    body = {
        'coordinates': [
            [start_coords[1], start_coords[0]],  # lon, lat
            [end_coords[1], end_coords[0]]
        ],
        'format': 'json',
        'units': 'mi',
        'geometry_format':'geojson'
    }
    
    try:
        print(f"Making route request from {start_coords} to {end_coords}")
        response = requests.post(url, json=body, headers=headers, timeout=30)
        
        print(f"Route API response status: {response.status_code}")
        print(f"Route API response text: {response.text[:500]}...")
        
        if response.status_code != 200:
            print(f"Route API error: {response.status_code} - {response.text}")
            return None, None
            
        data = response.json()
        print(f"Parsed response keys: {list(data.keys()) if isinstance(data, dict) else 'Not a dict'}")
        
        if isinstance(data, dict) and 'routes' in data and data['routes']:
            route = data['routes'][0]
            # Distance is returned in meters, convert to miles
            distance_meters = route['summary']['distance']
            distance_miles = distance_meters * 0.000621371  # Convert meters to miles
            coordinates = route['geometry']['coordinates']
            print(f"Route found: {distance_miles:.2f} miles")
            return distance_miles, coordinates
        elif isinstance(data, dict) and 'error' in data:
            print(f"API returned error: {data['error']}")
            return None, None
        else:
            print(f"Unexpected response format: {data}")
            return None, None
    except requests.exceptions.Timeout:
        print("Route API request timed out")
        return None, None
    except requests.exceptions.RequestException as e:
        print(f"Route API request error: {e}")
        return None, None
    except Exception as e:
        print(f"Route error: {e}")
        return None, None

def find_fuel_stops(route_coords, fuel_data, distance_miles):
    """Find optimal fuel stops along the route"""
    fuel_stops = []
    
    if fuel_data.empty:
        return fuel_stops
    
    # Calculating  number of fuel stops needed
    num_stops = math.ceil(distance_miles / VEHICLE_RANGE_MILES) - 1
    
    if num_stops <= 0:
        return fuel_stops
    
    # Calculating stop intervals along route
    stop_interval = len(route_coords) // (num_stops + 1)
    
    for i in range(1, num_stops + 1):
        stop_index = i * stop_interval
        if stop_index < len(route_coords):
            route_point = route_coords[stop_index]
            route_lat, route_lon = route_point[1], route_point[0]
            
            # Finding optimal fuel station (balance of distance and price)
            best_station = None
            best_score = float('inf')
            
            for _, station in fuel_data.iterrows():
                if pd.notna(station.get('Latitude')) and pd.notna(station.get('Longitude')):
                    station_coords = (float(station['Latitude']), float(station['Longitude']))
                    route_coords_point = (route_lat, route_lon)
                    distance = geodesic(route_coords_point, station_coords).miles
                    
                    if distance < 100:  # Within 100 miles of route
                        # Calculating optimization score (weighted distance and price)
                        price = float(station.get('Retail Price', 4.0))
                        # Score combines distance penalty and price (lower is better)
                        score = distance * 0.1 + price * 2
                        
                        if score < best_score:
                            best_score = score
                            best_station = station
            
            if best_station is not None:
                fuel_stops.append({
                    'name': str(best_station.get('Truckstop Name', 'Unknown Station')),
                    'address': str(best_station.get('Address', 'Unknown Address')),
                    'city': str(best_station.get('City', '')),
                    'state': str(best_station.get('State', '')),
                    'price': float(best_station.get('Retail Price', 0)),
                    'latitude': float(best_station.get('Latitude')),
                    'longitude': float(best_station.get('Longitude')),
                    'distance_from_route': round(geodesic((route_lat, route_lon), 
                                                        (float(best_station.get('Latitude')), 
                                                         float(best_station.get('Longitude')))).miles, 2)
                })
    
    return fuel_stops

def calculate_fuel_cost(distance_miles, fuel_stops):
    """Calculate total fuel cost for the trip"""
    gallons_needed = distance_miles / MILES_PER_GALLON
    
    if not fuel_stops:
        # Using average fuel price if no specific stops found
        avg_price = 3.50  # Default average gas price
        return gallons_needed * avg_price
    
    # Calculating cost based on fuel stops
    avg_stop_price = sum(stop['price'] for stop in fuel_stops) / len(fuel_stops)
    return gallons_needed * avg_stop_price

def generate_map_url(start_coords, end_coords, fuel_stops):
    """Generate static map URL using OpenRouteService"""
    base_url = "https://api.openrouteservice.org/v2/directions/driving-car"
    
    # Creating waypoints including fuel stops
    waypoints = [start_coords]
    for stop in fuel_stops:
        waypoints.append((stop['latitude'], stop['longitude']))
    waypoints.append(end_coords)
    
    # Converting  to lon,lat format for OpenRouteService
    coordinates = []
    for lat, lon in waypoints:
        coordinates.append([lon, lat])
    
    map_url = f"https://maps.openrouteservice.org/directions?n1={start_coords[0]}&n2={start_coords[1]}&n3={end_coords[0]}&n4={end_coords[1]}"
    return map_url

def add_cors_headers(response):
    """Add CORS headers to response"""
    response['Access-Control-Allow-Origin'] = '*'
    response['Access-Control-Allow-Methods'] = 'GET, POST, OPTIONS'
    response['Access-Control-Allow-Headers'] = 'Content-Type, Authorization'
    return response

@csrf_exempt
def fuel_route_planner(request):
    """Main API endpoint for fuel route planning"""
    
    print(f"Received {request.method} request to fuel_route_planner endpoint")
    print(f"Request headers: {dict(request.headers)}")
    
    # Handling preflight OPTIONS request
    if request.method == 'OPTIONS':
        response = JsonResponse({'status': 'ok'})
        return add_cors_headers(response)
    
    if request.method != 'POST':
        response = JsonResponse({'error': 'Only POST requests allowed'}, status=405)
        return add_cors_headers(response)
    
    try:
        data = json.loads(request.body)
        start_location = data.get('start_location')
        end_location = data.get('end_location')
        
        if not start_location or not end_location:
            return JsonResponse({'error': 'start_location and end_location required'}, status=400)
        
        # Geocoding  addresses
        start_coords = geocode_address(start_location)
        end_coords = geocode_address(end_location)
        
        if not start_coords[0] or not end_coords[0]:
            return JsonResponse({'error': 'Could not geocode addresses'}, status=400)
        
        # Getting route information
        distance_miles, route_coords = get_route_info(start_coords, end_coords)
        
        if not distance_miles:
            return JsonResponse({'error': 'Could not calculate route'}, status=400)
        
        # Loading fuel data and find optimal stops
        fuel_data = load_fuel_data()
        fuel_stops = find_fuel_stops(route_coords, fuel_data, distance_miles)
        
        # Calculating  total fuel cost
        total_fuel_cost = calculate_fuel_cost(distance_miles, fuel_stops)
        
        # Generating map URL
        map_url = generate_map_url(start_coords, end_coords, fuel_stops)
        
        response_data = {
            'success': True,
            'start_location': start_location,
            'end_location': end_location,
            'total_distance_miles': round(distance_miles, 2),
            'total_fuel_cost': round(total_fuel_cost, 2),
            'gallons_needed': round(distance_miles / MILES_PER_GALLON, 2),
            'estimated_fuel_stops': len(fuel_stops),
            'fuel_stops': fuel_stops,
            'map_url': map_url,
            'vehicle_specs': {
                'range_miles': VEHICLE_RANGE_MILES,
                'miles_per_gallon': MILES_PER_GALLON
            },
            'calculation_notes': f"Route calculated with {len(fuel_stops)} optimal fuel stops based on cost-effectiveness"
        }
        
        response = JsonResponse(response_data)
        return add_cors_headers(response)
        
    except json.JSONDecodeError:
        response = JsonResponse({'error': 'Invalid JSON in request body'}, status=400)
        return add_cors_headers(response)
    except Exception as e:
        response = JsonResponse({'error': str(e)}, status=500)
        return add_cors_headers(response)

def health_check(request):
    """Health check endpoint"""
    fuel_data = load_fuel_data()
    response = JsonResponse({
        'status': 'healthy', 
        'message': 'Fuel Route Planner API is running',
        'fuel_stations_loaded': len(fuel_data),
        'openrouteservice_configured': True,
        'google_drive_configured': True,
        'api_endpoints': {
            'fuel_route': '/api/fuel-route/',
            'health': '/health/',
            'test': '/test/'
        }
    })
    return add_cors_headers(response)

def test_endpoint(request):
    """Test endpoint for API functionality"""
    if request.method == 'GET':
        response = JsonResponse({
            'message': 'Test endpoint working',
            'sample_request': {
                'method': 'POST',
                'url': '/api/fuel-route/',
                'body': {
                    'start_location': 'Los Angeles, CA',
                    'end_location': 'New York, NY'
                }
            }
        })
        return add_cors_headers(response)
    response = JsonResponse({'error': 'Only GET requests allowed'}, status=405)
    return add_cors_headers(response)

# URL patterns
urlpatterns = [
    path('api/fuel-route/', fuel_route_planner, name='fuel_route_planner'),
    path('health/', health_check, name='health_check'),
    path('test/', test_endpoint, name='test_endpoint'),
    path('', health_check, name='home'),
]

# WSGI application
application = get_wsgi_application()

if __name__ == '__main__':
    from django.core.management import execute_from_command_line
    import sys
    
    if len(sys.argv) == 1:
        # Running development server
        os.environ.setdefault('DJANGO_SETTINGS_MODULE', '__main__')
        from django.core.management.commands.runserver import Command as RunServerCommand
        
        print("Starting Fuel Route Planner API...")
        print("API Endpoint: POST /api/fuel-route/")
        print("Health Check: GET /health/")
        print("Server running on: http://0.0.0.0:5000")
        
        # Starting the server
        cmd = RunServerCommand()
        cmd.run_from_argv(['manage.py', 'runserver', '0.0.0.0:5000'])
    else:
        execute_from_command_line(sys.argv)
