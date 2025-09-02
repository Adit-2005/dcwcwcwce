
import streamlit as st
import joblib, os, glob, pandas as pd, numpy as np, json, traceback

st.set_page_config(page_title="Hospital Readmission Predictor (Exact-model ready)", layout="wide", page_icon="🏥")
st.title("🏥 Hospital Readmission Predictor — Exact-model ready")
st.markdown("This app will attempt to load one of the exported `.joblib` models included in the `models/` directory. "\
            "If no compatible model is found (due to library/version mismatches), it will fall back to a bundled retrained pipeline.")

MODEL_DIR = "models"
loaded_model = None
loaded_model_name = None
load_errors = {}

# Try to load each .joblib model until one succeeds
for path in sorted(glob.glob(os.path.join(MODEL_DIR, "*.joblib"))):
    name = os.path.basename(path)
    try:
        m = joblib.load(path)
        # Basic sanity checks
        if hasattr(m, "predict") or hasattr(m, "predict_proba"):
            loaded_model = m
            loaded_model_name = name
            break
    except Exception as e:
        load_errors[name] = str(e)

# If no model loaded, show errors and stop
if loaded_model is None:
    st.error("No exported model could be loaded successfully due to compatibility issues.")
    st.write("Attempted models and errors:")
    for k,v in load_errors.items():
        st.write(f"- **{k}**: {v[:400]}")
    st.info("If you want exact predictions from one of your exported models, please ensure the environment has the same library versions used when the model was saved. "\
            "As a convenience, this project also includes a fallback retrained pipeline named `readmission_pipeline.joblib`.")
    # Try to load fallback if present
    fallback = os.path.join(MODEL_DIR, "readmission_pipeline.joblib")
    if os.path.exists(fallback):
        try:
            loaded_model = joblib.load(fallback)
            loaded_model_name = "readmission_pipeline.joblib (fallback retrained)"
            st.success("Loaded fallback retrained pipeline.")
        except Exception as e:
            st.error("Failed to load fallback pipeline as well.")
            st.write(traceback.format_exc())
            st.stop()
    else:
        st.stop()

st.success(f"Loaded model: `{loaded_model_name}`")

# Try to infer expected feature names (if pipeline or estimator exposes feature_names_in_)
feature_names = None
if hasattr(loaded_model, "feature_names_in_"):
    try:
        feature_names = list(loaded_model.feature_names_in_)
    except Exception:
        feature_names = None

# If it's a sklearn Pipeline, try to get preprocessors to find column names via named_transformers_
from sklearn.pipeline import Pipeline
if isinstance(loaded_model, Pipeline):
    try:
        # Many pipelines don't expose feature names; but if there's a ColumnTransformer we can inspect
        from sklearn.compose import ColumnTransformer
        for name, step in loaded_model.named_steps.items():
            if isinstance(step, ColumnTransformer):
                # Attempt to extract column lists
                try:
                    cols = []
                    for trans_name, trans, cols_sel in step.transformers:
                        if isinstance(cols_sel, (list, tuple)):
                            cols.extend(list(cols_sel))
                    if cols:
                        feature_names = cols
                except Exception:
                    pass
    except Exception:
        pass

st.write("**Detected feature names (if available):**")
st.write(feature_names if feature_names else "Feature names not available from this model. You can still provide input via CSV batch file matching your original columns.")

tab1, tab2 = st.tabs(["Single prediction", "Batch CSV scoring"])

with tab1:
    st.header("Single prediction")
    if feature_names:
        st.write("Provide values for the following features:")
        user_vals = {}
        cols = st.columns(3)
        for i, f in enumerate(feature_names):
            with cols[i % 3]:
                user_vals[f] = st.text_input(f, key=f"input_{f}")
        if st.button("Predict (single)"):
            X = pd.DataFrame([user_vals])
            # Try convert numeric-like columns
            for c in X.columns:
                try:
                    X[c] = pd.to_numeric(X[c])
                except Exception:
                    pass
            try:
                if hasattr(loaded_model, "predict_proba"):
                    p = loaded_model.predict_proba(X)[:,1][0]
                    st.metric("Readmission probability", f"{p:.2%}")
                    st.write("Prediction:", "Readmitted" if p>=0.5 else "Not Readmitted")
                else:
                    pred = loaded_model.predict(X)[0]
                    st.write("Prediction:", pred)
            except Exception as e:
                st.error("Error during prediction: " + str(e))
                st.text(traceback.format_exc())
    else:
        st.info("Feature names unavailable. Use batch CSV upload in the next tab.")

with tab2:
    st.header("Batch CSV scoring")
    st.write("Upload a CSV containing the same feature columns used at model training (no target column).")
    uploaded = st.file_uploader("Upload CSV", type=["csv"])
    if uploaded is not None:
        try:
            df = pd.read_csv(uploaded)
            # Ensure model receives expected columns if known
            if feature_names:
                missing = [c for c in feature_names if c not in df.columns]
                if missing:
                    st.error(f"Uploaded CSV is missing required columns: {missing}")
                else:
                    X = df[feature_names]
                    pass
            else:
                X = df
            if hasattr(loaded_model, "predict_proba"):
                probs = loaded_model.predict_proba(X)[:,1]
                preds = (probs >= 0.5).astype(int)
                df_out = df.copy()
                df_out["readmission_probability"] = probs
                df_out["prediction"] = np.where(preds==1, "Readmitted", "Not Readmitted")
            else:
                preds = loaded_model.predict(X)
                df_out = df.copy()
                df_out["prediction"] = preds
            st.success(f"Scored {len(df_out)} rows.")
            st.dataframe(df_out.head(200))
            st.download_button("Download Results CSV", df_out.to_csv(index=False).encode("utf-8"), "predictions.csv", "text/csv")
        except Exception as e:
            st.error("Failed to score uploaded CSV: " + str(e))
            st.write(traceback.format_exc())

st.write("---")
st.write("Model load attempts and errors (truncated):")
for k,v in list(load_errors.items())[:50]:
    st.write(f"- **{k}**: {str(v)[:400]}")
