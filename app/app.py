import streamlit as st
import pandas as pd
import numpy as np
import pickle
import shap
import matplotlib.pyplot as plt

st.set_page_config(
    page_title="Predictive Maintenance",
    layout="wide"
)

@st.cache_resource
def load_model():
    with open('app/model/xgb_model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('app/model/feature_names.pkl', 'rb') as f:
        features = pickle.load(f)
    return model, features

model, feature_names = load_model()

st.title("Industrial Equipment Failure Predictor")
st.markdown(
    "Enter current sensor readings to assess equipment failure risk. "
    "Model trained on AI4I 2020 dataset- XGBoost with engineered features, "
    "ROC-AUC 0.978, Recall 83.8%."
)
st.divider()

col_input, col_result = st.columns([1, 1], gap="large")

with col_input:
    st.subheader("Sensor Inputs")

    type_map = {'L- Low Quality': 1, 'M- Medium Quality': 2, 'H- High Quality': 0}
    product_type = st.selectbox("Product Type", list(type_map.keys()))

    air_temp = st.slider(
        "Air Temperature [K]", 295.0, 305.0, 300.0, step=0.1,
        help="Ambient temperature around the machine"
    )
    process_temp = st.slider(
        "Process Temperature [K]", 305.0, 315.0, 310.0, step=0.1,
        help="Machine operating temperature"
    )
    rot_speed = st.slider(
        "Rotational Speed [rpm]", 1168, 2886, 1500, step=1,
        help="Spindle rotational speed"
    )
    torque = st.slider(
        "Torque [Nm]", 3.8, 76.6, 40.0, step=0.1,
        help="Applied torque"
    )
    tool_wear = st.slider(
        "Tool Wear [min]", 0, 253, 100, step=1,
        help="Cumulative tool usage time"
    )

    power = rot_speed * torque
    temp_diff = process_temp - air_temp

    st.divider()
    st.caption("Derived features:")
    dcol1, dcol2 = st.columns(2)
    dcol1.metric("Power [W]", f"{power:,.0f}")
    dcol2.metric("Temp Differential [K]", f"{temp_diff:.1f}")

input_dict = {
    'Type': type_map[product_type],
    'Air temperature K': air_temp,
    'Process temperature K': process_temp,
    'Rotational speed rpm': rot_speed,
    'Torque Nm': torque,
    'Tool wear min': tool_wear,
    'Power': power,
    'Temp_diff': temp_diff,
}

input_df = pd.DataFrame([input_dict])[feature_names]

prob = model.predict_proba(input_df)[0][1]
pred = (prob >= 0.5).astype(int)

with col_result:
    st.subheader("Risk Assessment")

    st.metric(
        label="Failure Probability",
        value=f"{prob:.1%}",
        delta=None
    )

    if prob >= 0.7:
        st.error("HIGH RISK- Immediate maintenance recommended")
    elif prob >= 0.4:
        st.warning("MODERATE RISK- Schedule inspection soon")
    else:
        st.success("LOW RISK- Equipment operating normally")

    st.progress(float(prob))

    st.divider()

    fig, ax = plt.subplots(figsize=(5, 1.2))
    bar_color = '#ef4444' if prob >= 0.7 else ('#f59e0b' if prob >= 0.4 else '#22c55e')
    ax.barh(['Risk'], [prob], color=bar_color, height=0.5)
    ax.barh(['Risk'], [1 - prob], left=[prob], color='#e5e7eb', height=0.5)
    ax.set_xlim(0, 1)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xticklabels(['0%', '25%', '50%', '75%', '100%'], fontsize=8)
    ax.axvline(0.5, color='gray', linestyle='--', linewidth=0.8, alpha=0.6)
    ax.set_yticks([])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close()

    st.divider()

    st.subheader("Key Risk Factors")
    factors = []
    if tool_wear > 200:
        factors.append(f"Tool wear ({tool_wear} min) is critically high failure rate "
                       f"above 16% at this level")
    elif tool_wear > 150:
        factors.append(f"Tool wear ({tool_wear} min) is elevated monitor closely")

    if torque > 60:
        factors.append(f"High torque ({torque} Nm) overstrain risk")
    elif torque < 20:
        factors.append(f"Very low torque ({torque} Nm) with current speed "f"check for overspeed condition")

    if rot_speed < 1400:
        factors.append(f"Low rotational speed ({rot_speed} rpm) overstrain failure mode")
    elif rot_speed > 2500:
        factors.append(f"High rotational speed ({rot_speed} rpm) overspeed risk")

    if temp_diff < 8.6:
        factors.append(f"Low temperature differential ({temp_diff:.1f}K) "f"heat dissipation failure risk")

    if not factors:
        factors.append("All readings within normal operating range")

    for f in factors:
        st.markdown(f)

st.divider()
st.subheader("Why did the model make this prediction?")
st.caption("SHAP values show how each sensor reading contributed to the failure probability")

with st.spinner("Computing explanation..."):
    explainer = shap.Explainer(model)
    shap_vals = explainer(input_df)

    fig2, ax2 = plt.subplots(figsize=(9, 4))
    shap.waterfall_plot(shap_vals[0], show=False, max_display=8)
    plt.tight_layout()
    st.pyplot(fig2, use_container_width=True)
    plt.close()