from fastapi import FastAPI
import joblib
import uvicorn
from pydantic import BaseModel
from pathlib import Path

student_app = FastAPI()



model_app = joblib.load('model.pkl')
scaler = joblib.load('scaler.pkl')


class StudentSchema(BaseModel):
    gender: str
    race_ethnicity: str
    parent_education: str
    lunch: str
    test_preparation: str
    math_score: int
    reading_score: int


@student_app.post('/predict/')
async def predict(student: StudentSchema):
    student_dict = student.dict()

    new_gender = student_dict.pop('gender')
    gender1or_0 = [
        1 if new_gender == 'male' else 0
    ]

    new_race = student_dict.pop('race_ethnicity')
    race1or_0 = [
        1 if new_race == 'group B' else 0,
        1 if new_race == 'group C' else 0,
        1 if new_race == 'group D' else 0,
        1 if new_race == 'group E' else 0,
    ]

    new_lunch = student_dict.pop('lunch')
    lunch1_0 = [
        1 if new_lunch == "standard" else 0,
        ]

    new_test = student_dict.pop('test_preparation')
    test1_0 = [
        1 if new_test == "none" else 0,
    ]

    features = list(student_dict.values()) + gender1or_0 + race1or_0 + lunch1_0 + test1_0

    scaler_data = scaler.transform([features])
    print(model_app.predict(scaler_data))
    pred = model_app.predict(scaler_data)[0]
    return {'predict': round(pred, 2)}


if __name__ == '__main__':
    uvicorn.run(student_app, host='127.0.0.1', port=8000)