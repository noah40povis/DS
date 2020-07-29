from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_predict_routes():
    """Test all routes return 200 Success with known input 
    """
    routes_to_test = ['/predict','/nsfw_predict', '/test_predict']
    post_body = {'title': 'foo bar bar barrrr',
                'selftext': 'banjo didjeridoo djembe khomuz igil',
                }
    for route in routes_to_test:
        response = client.post(route,json=post_body)
        body = response.json()
        assert response.status_code == 200
    


