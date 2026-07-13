import streamlit as st
import requests
import time
import pandas as pd

URL = "https://agentic-hvac-classroom.onrender.com/api/v1/telemetry"

st.set_page_config(page_title="Smart HVAC Digital Twin", layout="wide")

st.title("Smart Classroom HVAC Digital Twin")
st.write("Simulated multi-zone classroom telemetry connected to live Render cloud controller.")

if "results" not in st.session_state:
    st.session_state.results = []

col1, col2 = st.columns(2)

with col1:
    st.subheader("Window Zone")
    window_temp = st.slider("Window Temperature (°C)", 18.0, 35.0, 27.5, 0.1)
    window_humidity = st.slider("Window Humidity (%)", 20.0, 80.0, 55.0, 1.0)
    window_co2 = st.slider("Window CO₂ (ppm)", 400, 2000, 1200, 10)
    window_pref = st.slider("Window Occupant Preferred Temp (°C)", 18.0, 26.0, 21.5, 0.1)

with col2:
    st.subheader("Interior Zone")
    interior_temp = st.slider("Interior Temperature (°C)", 18.0, 35.0, 25.0, 0.1)
    interior_humidity = st.slider("Interior Humidity (%)", 20.0, 80.0, 50.0, 1.0)
    interior_co2 = st.slider("Interior CO₂ (ppm)", 400, 2000, 850, 10)
    interior_pref = st.slider("Interior Occupant Preferred Temp (°C)", 18.0, 26.0, 22.0, 0.1)

payload = {
    "scenario_id": f"dashboard_test_{len(st.session_state.results) + 1}",
    "description": "Software digital twin dashboard test",
    "window_zone": {
        "air_temp_c": window_temp,
        "relative_humidity": window_humidity,
        "co2_ppm": window_co2,
        "occupant_preferences": [window_pref, window_pref + 0.5, window_pref - 0.5],
    },
    "interior_zone": {
        "air_temp_c": interior_temp,
        "relative_humidity": interior_humidity,
        "co2_ppm": interior_co2,
        "occupant_preferences": [interior_pref, interior_pref + 0.5, interior_pref - 0.5],
    },
}

st.subheader("Payload Sent to Cloud")
st.json(payload)

if st.button("Send Telemetry to Render Cloud"):
    start = time.time()

    try:
        response = requests.post(URL, json=payload, timeout=90)
        end = time.time()
        latency_ms = round((end - start) * 1000, 2)

        st.write("HTTP Status:", response.status_code)
        st.write("Latency:", latency_ms, "ms")

        if response.status_code == 200:
            data = response.json()

            st.subheader("Cloud Agent Response")
            st.json(data)

            control = data.get("control_signals", {})

            result = {
                "run": len(st.session_state.results) + 1,
                "window_temp_c": window_temp,
                "window_humidity": window_humidity,
                "window_co2": window_co2,
                "interior_temp_c": interior_temp,
                "interior_humidity": interior_humidity,
                "interior_co2": interior_co2,
                "latency_ms": latency_ms,
                "hvac_mode": control.get("hvac_mode"),
                "window_damper_pct": control.get("window_damper_pct"),
                "interior_damper_pct": control.get("interior_damper_pct"),
                "fan_speed": control.get("fan_speed"),
                "supervisor_reasoning": data.get("supervisor_reasoning"),
            }

            st.session_state.results.append(result)

        else:
            st.error(response.text)

    except Exception as e:
        st.error(f"Request failed: {e}")

if st.session_state.results:
    st.subheader("Experiment Results Table")

    df = pd.DataFrame(st.session_state.results)
    st.dataframe(df)

    csv = df.to_csv(index=False).encode("utf-8")

    st.download_button(
        label="Download Results CSV",
        data=csv,
        file_name="digital_twin_results.csv",
        mime="text/csv",
    )

    avg_latency = df["latency_ms"].mean()
    st.metric("Average Latency", f"{avg_latency:.2f} ms")
