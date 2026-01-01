import joblib
import pandas as pd
import gradio as gr

# --------------------------------------------------
# Load trained pipeline
# --------------------------------------------------
model = joblib.load("student_score_model.joblib")

# --------------------------------------------------
# Prediction function
# --------------------------------------------------
def predict_exam_score(
    age,
    gender,
    course,
    study_hours,
    class_attendance,
    internet_access,
    sleep_hours,
    sleep_quality,
    study_method,
    facility_rating,
    exam_difficulty
):
    input_df = pd.DataFrame([{
        "id": 0,  # dummy id
        "age": age,
        "gender": gender,
        "course": course,
        "study_hours": study_hours,
        "class_attendance": class_attendance,
        "internet_access": internet_access,
        "sleep_hours": sleep_hours,
        "sleep_quality": sleep_quality,
        "study_method": study_method,
        "facility_rating": facility_rating,
        "exam_difficulty": exam_difficulty
    }])

    prediction = model.predict(input_df)[0]
    return round(float(prediction), 2)

# --------------------------------------------------
# Gradio UI
# --------------------------------------------------
inputs = [
    gr.Slider(15, 30, step=1, label="Age"),
    gr.Dropdown(["male", "female", "other"], label="Gender"),
    gr.Dropdown(["b.sc", "b.tech", "b.com", "bca", "ba", "bba", "diploma"], label="Course"),
    gr.Slider(0, 15, step=0.1, label="Study Hours per Day"),
    gr.Slider(0, 100, step=1, label="Class Attendance (%)"),
    gr.Dropdown(["yes", "no"], label="Internet Access"),
    gr.Slider(0, 12, step=0.1, label="Sleep Hours"),
    gr.Dropdown(["poor", "average", "good"], label="Sleep Quality"),
    gr.Dropdown(
        ["self-study", "online videos", "group study", "coaching", "mixed"],
        label="Study Method"
    ),
    gr.Dropdown(["low", "medium", "high"], label="Facility Rating"),
    gr.Dropdown(["easy", "moderate", "hard"], label="Exam Difficulty")
]

app = gr.Interface(
    fn=predict_exam_score,
    inputs=inputs,
    outputs=gr.Number(label="Predicted Exam Score"),
    title="🎓 Student Exam Score Predictor",
    description="Predict exam score using academic, lifestyle, and study-behavior features.",
)

# --------------------------------------------------
# Launch (HF compatible)
# --------------------------------------------------
if __name__ == "__main__":
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        show_error=True
    )
