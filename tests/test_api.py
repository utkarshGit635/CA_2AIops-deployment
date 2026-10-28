from fastapi.testclient import TestClient

from app.main import app


client = TestClient(app)


def test_root_endpoint():
    response = client.get("/")

    assert response.status_code == 200
    assert response.json()["message"] == (
        "Student Performance Prediction API is running!"
    )


def test_health_endpoint():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "healthy"
    assert response.json()["model_loaded"] is True


def test_prediction_endpoint():
    payload = {
        "school": "GP",
        "sex": "F",
        "age": 18,
        "address": "U",
        "famsize": "GT3",
        "Pstatus": "A",
        "Medu": 4,
        "Fedu": 4,
        "Mjob": "at_home",
        "Fjob": "teacher",
        "reason": "course",
        "guardian": "mother",
        "traveltime": 2,
        "studytime": 2,
        "failures": 0,
        "schoolsup": "no",
        "famsup": "yes",
        "paid": "no",
        "activities": "yes",
        "nursery": "yes",
        "higher": "yes",
        "internet": "yes",
        "romantic": "no",
        "famrel": 4,
        "freetime": 3,
        "goout": 3,
        "Dalc": 1,
        "Walc": 1,
        "health": 3,
        "absences": 4,
        "G1": 10,
        "G2": 11,
    }

    response = client.post("/predict", json=payload)

    assert response.status_code == 200

    result = response.json()

    assert "predicted_final_grade" in result
    assert isinstance(result["predicted_final_grade"], (int, float))
    assert 0 <= result["predicted_final_grade"] <= 20


def test_prediction_rejects_invalid_age():
    payload = {
        "school": "GP",
        "sex": "F",
        "age": 10,
        "address": "U",
        "famsize": "GT3",
        "Pstatus": "A",
        "Medu": 4,
        "Fedu": 4,
        "Mjob": "at_home",
        "Fjob": "teacher",
        "reason": "course",
        "guardian": "mother",
        "traveltime": 2,
        "studytime": 2,
        "failures": 0,
        "schoolsup": "no",
        "famsup": "yes",
        "paid": "no",
        "activities": "yes",
        "nursery": "yes",
        "higher": "yes",
        "internet": "yes",
        "romantic": "no",
        "famrel": 4,
        "freetime": 3,
        "goout": 3,
        "Dalc": 1,
        "Walc": 1,
        "health": 3,
        "absences": 4,
        "G1": 10,
        "G2": 11,
    }

    response = client.post("/predict", json=payload)

    assert response.status_code == 422
