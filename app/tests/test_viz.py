from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)

post_routes_to_test = ['/viz']


def test_all_routes_200():
    """Return 200 Success with valid input."""
    post_body = {'title': 'foo bar bar barrrr',
                 'selftext': 'banjo didjeridoo djembe khomuz igil'
                 }
    for route in post_routes_to_test:
        response = client.post(route, json=post_body)
        body = response.json()
        assert response.status_code == 200


def test_invalid_input():
    """Return 404 if the endpoint isn't valid US state postal code."""
    response = client.get('/viz/ZZ')
    body = response.json()
    assert response.status_code == 404
