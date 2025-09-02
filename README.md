# Streamlit — Exact-model-ready Hospital Readmission App

This project will attempt to load any exported `.joblib` model located in the `models/` folder.
If no exported model can be loaded (due to binary/version mismatches), the app will fall back to the bundled `readmission_pipeline.joblib` if present.

## How to run locally
```
pip install -r requirements.txt
streamlit run app.py
```

## Notes
- For exact reproduction of predictions using your original exported model, ensure the Python environment uses the same library versions used when the model was saved.
- If you encounter model loading errors, check the `app` UI to see attempts/errors for each `.joblib` file.