from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


# todo

def test_valid_input():
    """Return 200 Success when input is valid."""
    response = client.post(
        '/predict',
        json={
            'title': 'foo bar bar barrrr',
            'selftext': 'banjo didjeridoo djembe khomuz igil',
            'n_results': 4
        }
    )
    body = response.json()
    assert response.status_code == 200
    assert body['prediction'] in [True, False]
    assert 0.50 <= body['probability'] < 1


def test_invalid_input():
    """Return 422 Validation Error when n_results is negative."""
    response = client.post(
        '/predict',
        json={
            'title': 'foo bar bar barrrr',
            'selftext': 'banjo didjeridoo djembe khomuz igil',
            'n_results': -3
        }
    )
    body = response.json()
    assert response.status_code == 422
    assert 'selftext' in body['detail'][1]['loc']
