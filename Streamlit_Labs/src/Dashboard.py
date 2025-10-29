import json
import requests
import streamlit as st
from streamlit.logger import get_logger

FASTAPI_BACKEND_ENDPOINT = "http://localhost:8000"
LOGGER = get_logger(__name__)

WINE_FEATURES = [
    "alcohol", "malic_acid", "ash", "alcalinity_of_ash", "magnesium",
    "total_phenols", "flavanoids", "nonflavanoid_phenols", "proanthocyanins",
    "color_intensity", "hue", "od280_od315", "proline",
]

# Fallback label map if backend only returns a number
FALLBACK_LABELS = ["class_0", "class_1", "class_2"]

def run():
    st.set_page_config(page_title="Wine Classification Demo", page_icon="🍷", layout="wide")

    st.markdown("<h1 style='text-align:center;margin-bottom:0;'>🍷 Wine Classification Demo</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align:center;color:#6b7280;'>Streamlit front-end + FastAPI back-end</p>", unsafe_allow_html=True)
    st.markdown("---")

    col_status, col_upload, col_preview, col_action = st.columns([1.1, 1.4, 2.2, 0.9])

    # Backend status
    with col_status:
        st.markdown("#### Backend")
        with st.container(border=True):
            try:
                r = requests.get(FASTAPI_BACKEND_ENDPOINT, timeout=2.5)
                if r.status_code == 200:
                    st.success("Online ✅")
                else:
                    st.warning(f"Issue: {r.status_code} 😕")
            except requests.RequestException as e:
                LOGGER.error(e)
                st.error("Offline 😱")
            st.caption(f"Endpoint: {FASTAPI_BACKEND_ENDPOINT}")

    # Upload JSON
    with col_upload:
        st.markdown("#### Upload")
        with st.container(border=True):
            uploaded = st.file_uploader(
                "Test input (JSON)",
                type=["json"],
                label_visibility="collapsed",
                accept_multiple_files=False,
                help="Upload a JSON payload with wine feature values only.",
            )
            st.caption("One file only")

    # Preview JSON
    with col_preview:
        st.markdown("#### Preview")
        with st.container(border=True, height=220):
            if uploaded:
                try:
                    uploaded.seek(0)
                    data = json.load(uploaded)
                    st.json(data)
                    st.session_state["TEST_INPUT_DATA"] = data
                except json.JSONDecodeError as e:
                    st.error(f"Invalid JSON: {e}")
                    st.session_state["TEST_INPUT_DATA"] = None
            else:
                st.info("Upload JSON to preview")
                st.session_state["TEST_INPUT_DATA"] = None

    # Action
    with col_action:
        st.markdown("#### Action")
        with st.container(border=True):
            predict_button = st.button("Predict", use_container_width=True)

    st.markdown("---")

    tab_home, tab_predict, tab_about = st.tabs(["Home", "Predict", "About"])

    with tab_home:
        st.subheader("Overview")
        st.markdown(
            """
            - Upload a JSON payload with 13 wine features.
            - Click **Predict** to call the FastAPI `/predict` endpoint.
            - Result shows both the class id and a human-readable label.
            """
        )
        st.code(
            json.dumps(
                {
                    "alcohol": 13.2,
                    "malic_acid": 2.3,
                    "ash": 2.4,
                    "alcalinity_of_ash": 19.5,
                    "magnesium": 102,
                    "total_phenols": 2.6,
                    "flavanoids": 2.0,
                    "nonflavanoid_phenols": 0.3,
                    "proanthocyanins": 1.5,
                    "color_intensity": 5.2,
                    "hue": 1.05,
                    "od280_od315": 2.8,
                    "proline": 750
                },
                indent=2,
            ),
            language="json",
        )

    with tab_predict:
        st.subheader("Prediction")
        if predict_button:
            payload = st.session_state.get("TEST_INPUT_DATA")

            if not payload:
                st.warning("Upload a JSON file first.")
            else:
                missing = [f for f in WINE_FEATURES if f not in payload]
                if missing:
                    st.error(f"Missing keys in uploaded JSON: {missing}")
                else:
                    with st.spinner("Calling backend…"):
                        try:
                            resp = requests.post(f"{FASTAPI_BACKEND_ENDPOINT}/predict", json=payload, timeout=6)
                            if resp.ok:
                                out = resp.json()
                                # Prefer backend-provided label; fall back to local map
                                label = out.get("predicted_label")
                                cls = out.get("predicted_class")
                                if label is None and isinstance(cls, int) and 0 <= cls < len(FALLBACK_LABELS):
                                    label = FALLBACK_LABELS[cls]

                                if label is not None:
                                    st.success(f"Prediction: **{label}**")
                                if cls is not None:
                                    st.caption(f"class id: {cls}")
                                st.json(out)
                            else:
                                st.error(f"Backend error: {resp.status_code}")
                                st.write(resp.text)
                        except requests.RequestException as e:
                            st.error("Failed to reach backend")
                            st.caption(str(e))

    with tab_about:
        st.subheader("About")
        st.markdown(
            """
            - Dataset: `sklearn.datasets.load_wine()`
            - Model: RandomForestClassifier
            - API: `/predict` → returns `predicted_class` and `predicted_label`
            """
        )

    st.markdown("<hr/>", unsafe_allow_html=True)
    st.caption("Built with Streamlit + FastAPI • Wine Demo")

if __name__ == "__main__":
    run()
