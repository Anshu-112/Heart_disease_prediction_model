"""
Heart Stroke Risk Predictor — modern Streamlit UI
=================================================
Drop-in replacement for your existing app.py.

Expects (in the same folder):
    knn_model.pkl   -> your trained KNN (or a full sklearn Pipeline)
    scaler.pkl      -> optional StandardScaler, if not baked into a Pipeline
    columns.pkl     -> optional list of training column names (after get_dummies)

If none are found, the app runs in DEMO MODE with a transparent rule-based
score so you can still show off the UI.

Run:  streamlit run app.py
Deps: streamlit pandas numpy scikit-learn joblib plotly
"""

import os
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

try:
    import joblib
except ImportError:
    joblib = None


# ──────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ──────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Heart Stroke Risk Predictor",
    page_icon="💓",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ──────────────────────────────────────────────────────────────────────────
# STYLING
# ──────────────────────────────────────────────────────────────────────────
CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"]  { font-family: 'Inter', sans-serif; }

#MainMenu, footer {visibility: hidden;}

.block-container { padding-top: 2rem; padding-bottom: 3rem; max-width: 1150px; }

/* ---------- Hero ---------- */
.hero {
    background: linear-gradient(120deg, #e11d48 0%, #be123c 45%, #7f1d1d 100%);
    border-radius: 20px;
    padding: 2.2rem 2.4rem;
    color: #fff;
    box-shadow: 0 18px 40px -18px rgba(190, 18, 60, .6);
    margin-bottom: 1.6rem;
}
.hero h1 {
    font-size: 2.15rem; font-weight: 800; margin: 0 0 .45rem 0;
    letter-spacing: -.02em; line-height: 1.15; color: #fff;
}
.hero p { font-size: .98rem; opacity: .92; margin: 0; max-width: 62ch; }
.hero .pill {
    display: inline-block; background: rgba(255,255,255,.18);
    border: 1px solid rgba(255,255,255,.28);
    padding: .22rem .7rem; border-radius: 999px;
    font-size: .74rem; font-weight: 600; letter-spacing: .04em;
    text-transform: uppercase; margin-bottom: .9rem;
}

/* ---------- Section cards ---------- */
.section {
    background: rgba(148, 163, 184, .08);
    border: 1px solid rgba(148, 163, 184, .22);
    border-radius: 16px;
    padding: 1.1rem 1.3rem .4rem 1.3rem;
    margin-bottom: 1.1rem;
}
.section-title {
    font-weight: 700; font-size: 1.02rem; margin: 0 0 .15rem 0;
    display: flex; align-items: center; gap: .5rem;
}
.section-sub {
    font-size: .82rem; opacity: .65; margin: 0 0 .9rem 0;
}

/* ---------- Result card ---------- */
.result {
    border-radius: 18px; padding: 1.5rem 1.7rem; margin-top: .4rem;
    border: 1px solid; text-align: left;
}
.result h2 { margin: .15rem 0 .35rem 0; font-size: 1.6rem; font-weight: 800; }
.result p  { margin: 0; font-size: .92rem; opacity: .9; }
.res-high { background: rgba(225,29,72,.10);  border-color: rgba(225,29,72,.45);  color:#e11d48; }
.res-mid  { background: rgba(245,158,11,.10); border-color: rgba(245,158,11,.45); color:#d97706; }
.res-low  { background: rgba(16,185,129,.10); border-color: rgba(16,185,129,.45); color:#059669; }

/* ---------- Glossary ---------- */
.gloss {
    background: rgba(148,163,184,.07);
    border-left: 3px solid #e11d48;
    border-radius: 0 12px 12px 0;
    padding: .9rem 1.1rem; margin-bottom: .75rem;
}
.gloss h4 { margin: 0 0 .3rem 0; font-size: .95rem; font-weight: 700; }
.gloss p  { margin: 0; font-size: .86rem; opacity: .8; line-height: 1.55; }
.gloss .tag {
    display:inline-block; font-size:.7rem; font-weight:600; letter-spacing:.03em;
    background: rgba(225,29,72,.12); color:#e11d48;
    padding:.12rem .5rem; border-radius:6px; margin-left:.45rem;
}

/* ---------- Flag chips ---------- */
.chip {
    display:inline-block; padding:.3rem .7rem; border-radius:999px;
    font-size:.8rem; font-weight:600; margin:.2rem .35rem .2rem 0;
}
.chip-bad  { background: rgba(225,29,72,.12);  color:#e11d48; }
.chip-warn { background: rgba(245,158,11,.14); color:#d97706; }
.chip-ok   { background: rgba(16,185,129,.12); color:#059669; }

/* ---------- Buttons ---------- */
.stButton > button {
    width: 100%; border-radius: 12px; font-weight: 700; font-size: 1rem;
    padding: .7rem 1rem; border: none; color: #fff;
    background: linear-gradient(120deg, #e11d48, #9f1239);
    box-shadow: 0 10px 22px -12px rgba(225,29,72,.9);
    transition: transform .12s ease, box-shadow .12s ease;
}
.stButton > button:hover {
    transform: translateY(-1px); color:#fff;
    box-shadow: 0 14px 26px -12px rgba(225,29,72,1);
}

.disclaimer {
    font-size: .78rem; opacity: .6; line-height: 1.6;
    border-top: 1px solid rgba(148,163,184,.25);
    padding-top: .9rem; margin-top: 1.6rem;
}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────────────────
# MODEL LOADING
# ──────────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_artifacts():
    """Load model / scaler / column order if they exist."""
    art = {"model": None, "scaler": None, "columns": None}
    if joblib is None:
        return art
    for key, fname in [
        ("model", "KNN_heart.pkl"),
        ("scaler", "scaler.pkl"),
        ("columns", "columns.pkl"),
    ]:
        if os.path.exists(fname):
            try:
                art[key] = joblib.load(fname)
            except Exception:
                pass
    return art


ART = load_artifacts()
DEMO_MODE = ART["model"] is None


def build_frame(d: dict) -> pd.DataFrame:
    """Raw single-row frame using the original dataset's column names."""
    return pd.DataFrame([{
        "Age": d["age"],
        "Sex": d["sex"],
        "ChestPainType": d["cp"],
        "RestingBP": d["bp"],
        "Cholesterol": d["chol"],
        "FastingBS": d["fbs"],
        "RestingECG": d["ecg"],
        "MaxHR": d["hr"],
        "ExerciseAngina": d["angina"],
        "Oldpeak": d["oldpeak"],
        "ST_Slope": d["slope"],
    }])


def demo_score(d: dict) -> float:
    """Transparent heuristic used only when no model file is present."""
    s = 0.0
    s += np.clip((d["age"] - 35) / 45, 0, 1) * 0.20
    s += 0.08 if d["sex"] == "M" else 0.0
    s += {"ASY": 0.26, "NAP": 0.10, "ATA": 0.05, "TA": 0.08}[d["cp"]]
    s += np.clip((d["bp"] - 110) / 90, 0, 1) * 0.10
    s += np.clip((d["chol"] - 180) / 220, 0, 1) * 0.08
    s += 0.07 if d["fbs"] == 1 else 0.0
    s += np.clip((150 - d["hr"]) / 90, 0, 1) * 0.12
    s += 0.14 if d["angina"] == "Y" else 0.0
    s += np.clip(d["oldpeak"] / 4.0, 0, 1) * 0.14
    s += {"Up": 0.0, "Flat": 0.14, "Down": 0.18}[d["slope"]]
    return float(np.clip(s, 0.02, 0.98))


def predict(d: dict):
    """Return (probability_of_disease, label_int)."""
    if DEMO_MODE:
        p = demo_score(d)
        return p, int(p >= 0.5)

    raw = build_frame(d)
    model, scaler, cols = ART["model"], ART["scaler"], ART["columns"]

    # A full Pipeline can usually eat the raw frame directly.
    try:
        X = raw
        if cols is not None:
            X = pd.get_dummies(raw, drop_first=True)
            X = X.reindex(columns=cols, fill_value=0)
        if scaler is not None:
            X = scaler.transform(X)
        label = int(model.predict(X)[0])
        if hasattr(model, "predict_proba"):
            prob = float(model.predict_proba(X)[0][1])
        else:
            prob = float(label)
        return prob, label
    except Exception as e:
        st.error(
            f"Could not run the saved model ({type(e).__name__}: {e}). "
            "Check that `columns.pkl` matches your training columns, or save "
            "your preprocessing + KNN as a single sklearn Pipeline."
        )
        st.stop()


# ──────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ──────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 💓 Heart Risk Predictor")
    st.caption("K-Nearest Neighbours classifier trained on the Heart Failure Prediction dataset.")
    st.divider()

    st.markdown("**Model status**")
    if DEMO_MODE:
        st.warning("Demo mode — no `knn_model.pkl` found. Scores come from a "
                   "transparent heuristic, not your trained model.")
    else:
        st.success("Trained KNN loaded ✓")

    st.divider()
    st.markdown("**How to read the score**")
    st.markdown(
        "- **0–30%** — low estimated risk\n"
        "- **30–60%** — borderline, worth a check-up\n"
        "- **60–100%** — high, see a cardiologist"
    )
    st.divider()
    st.caption("Built by Anshu · Educational use only.")


# ──────────────────────────────────────────────────────────────────────────
# HERO
# ──────────────────────────────────────────────────────────────────────────
st.markdown(
    """
    <div class="hero">
      <span class="pill">Machine Learning · KNN Classifier</span>
      <h1>Heart Stroke Risk Predictor</h1>
      <p>Enter eleven clinical measurements and the model estimates the likelihood
      of heart disease. Every field has a plain-English explanation — hover the
      <b>?</b> icon, or open the <b>Understand the Inputs</b> tab for the full guide.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

tab_predict, tab_learn, tab_about = st.tabs(
    ["🩺  Risk Assessment", "📖  Understand the Inputs", "⚙️  About the Model"]
)


# ──────────────────────────────────────────────────────────────────────────
# TAB 1 — ASSESSMENT
# ──────────────────────────────────────────────────────────────────────────
with tab_predict:

    # ---------- Section 1: Patient profile ----------
    st.markdown('<div class="section">', unsafe_allow_html=True)
    st.markdown('<p class="section-title">👤 Patient Profile</p>'
                '<p class="section-sub">Basic demographics — age and sex are both '
                'independent risk factors for cardiac disease.</p>',
                unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        age = st.slider(
            "Age", 18, 100, 45,
            help="Age in years. Cardiac risk climbs steadily after 45 for men "
                 "and after 55 for women."
        )
    with c2:
        sex = st.radio(
            "Sex", ["M", "F"], horizontal=True,
            format_func=lambda x: "Male" if x == "M" else "Female",
            help="Biological sex. Men develop heart disease earlier on average; "
                 "women's risk rises sharply after menopause."
        )
    st.markdown('</div>', unsafe_allow_html=True)

    # ---------- Section 2: Vitals & labs ----------
    st.markdown('<div class="section">', unsafe_allow_html=True)
    st.markdown('<p class="section-title">🧪 Vitals &amp; Blood Work</p>'
                '<p class="section-sub">Measurements taken at rest, plus a fasting '
                'blood sample.</p>',
                unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    with c1:
        bp = st.number_input(
            "Resting Blood Pressure (mm Hg)", 80, 220, 120, step=1,
            help="Systolic pressure while seated and at rest. Normal is under 120. "
                 "130+ is classed as hypertension, which strains the heart."
        )
        st.caption("🟢 <120 normal · 🟡 120–139 elevated · 🔴 140+ high")
    with c2:
        chol = st.number_input(
            "Cholesterol (mg/dL)", 0, 700, 200, step=5,
            help="Total serum cholesterol. Under 200 is desirable, 240+ is high. "
                 "A value of 0 means the measurement was missing in the dataset."
        )
        st.caption("🟢 <200 desirable · 🟡 200–239 borderline · 🔴 240+ high")
    with c3:
        fbs = st.selectbox(
            "Fasting Blood Sugar > 120 mg/dL", [0, 1],
            format_func=lambda x: "No — normal" if x == 0 else "Yes — elevated",
            help="Was blood sugar above 120 mg/dL after an overnight fast? "
                 "Elevated fasting glucose suggests diabetes, a major heart-disease risk factor."
        )
        st.caption("Proxy indicator for diabetes")
    st.markdown('</div>', unsafe_allow_html=True)

    # ---------- Section 3: Cardiac testing ----------
    st.markdown('<div class="section">', unsafe_allow_html=True)
    st.markdown('<p class="section-title">❤️‍🔥 Cardiac Testing</p>'
                '<p class="section-sub">Results from an ECG and a supervised '
                'exercise stress test — the strongest signals in this model.</p>',
                unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        cp = st.selectbox(
            "Chest Pain Type",
            ["ATA", "NAP", "ASY", "TA"],
            format_func=lambda x: {
                "TA":  "TA — Typical Angina",
                "ATA": "ATA — Atypical Angina",
                "NAP": "NAP — Non-Anginal Pain",
                "ASY": "ASY — Asymptomatic",
            }[x],
            help="TA: classic pressure on exertion, relieved by rest. "
                 "ATA: chest pain that doesn't fit the classic pattern. "
                 "NAP: pain unlikely to be heart-related. "
                 "ASY: no chest pain at all — counter-intuitively the highest-risk "
                 "group here, because silent disease goes undetected."
        )
        ecg = st.selectbox(
            "Resting ECG",
            ["Normal", "ST", "LVH"],
            format_func=lambda x: {
                "Normal": "Normal",
                "ST":     "ST — ST-T wave abnormality",
                "LVH":    "LVH — Left ventricular hypertrophy",
            }[x],
            help="Electrocardiogram taken at rest. "
                 "ST abnormalities can indicate reduced blood flow to the heart muscle. "
                 "LVH means the heart's main pumping chamber has thickened, often from "
                 "long-standing high blood pressure."
        )
        angina = st.radio(
            "Exercise-Induced Angina", ["N", "Y"], horizontal=True,
            format_func=lambda x: "No" if x == "N" else "Yes",
            help="Did chest pain appear during the exercise stress test? "
                 "Pain triggered by exertion strongly suggests narrowed coronary arteries."
        )
    with c2:
        hr = st.slider(
            "Max Heart Rate Achieved", 60, 210, 150,
            help="Highest heart rate reached during the stress test. A LOW maximum "
                 "is the warning sign — it means the heart couldn't respond to demand. "
                 "Rough expected peak ≈ 220 − age."
        )
        expected = 220 - age
        st.caption(f"Age-predicted maximum ≈ **{expected} bpm** "
                   f"({hr/expected*100:.0f}% achieved)")

        oldpeak = st.slider(
            "Oldpeak (ST Depression)", -2.0, 7.0, 1.0, 0.1,
            help="How far the ECG's ST segment dropped during exercise compared to rest, "
                 "in millimetres. 0 is ideal. Above 2.0 suggests significant oxygen "
                 "starvation of the heart muscle (ischemia)."
        )
        slope = st.selectbox(
            "ST Slope (peak exercise)",
            ["Up", "Flat", "Down"],
            format_func=lambda x: {
                "Up":   "Up — upsloping (healthy)",
                "Flat": "Flat — flat (concerning)",
                "Down": "Down — downsloping (high risk)",
            }[x],
            help="Direction of the ST segment at peak exercise. Upsloping is the normal "
                 "healthy response; flat and especially downsloping patterns are "
                 "associated with coronary artery disease."
        )
    st.markdown('</div>', unsafe_allow_html=True)

    go_btn = st.button("🔍  Predict Heart Disease Risk", type="primary")

    # ---------- Results ----------
    if go_btn:
        data = dict(age=age, sex=sex, cp=cp, bp=bp, chol=chol, fbs=fbs,
                    ecg=ecg, hr=hr, angina=angina, oldpeak=oldpeak, slope=slope)

        with st.spinner("Analysing patient profile…"):
            prob, label = predict(data)

        pct = prob * 100
        if pct >= 60:
            band, cls, head = "High", "res-high", "Elevated risk detected"
            msg = ("The model places this profile in the high-risk group. This is a "
                   "screening signal, not a diagnosis — please consult a cardiologist.")
            colour = "#e11d48"
        elif pct >= 30:
            band, cls, head = "Borderline", "res-mid", "Borderline risk"
            msg = ("Several indicators sit outside the ideal range. A routine "
                   "cardiac check-up would be sensible.")
            colour = "#f59e0b"
        else:
            band, cls, head = "Low", "res-low", "Low estimated risk"
            msg = ("Most indicators look healthy for this profile. Keep up regular "
                   "exercise, balanced diet and periodic screening.")
            colour = "#10b981"

        st.markdown("###")
        r1, r2 = st.columns([1.05, 1])

        with r1:
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=pct,
                number={"suffix": "%", "font": {"size": 46, "color": colour}},
                title={"text": f"<b>{band} Risk</b>", "font": {"size": 17}},
                gauge={
                    "axis": {"range": [0, 100], "tickwidth": 1,
                             "tickcolor": "rgba(148,163,184,.5)"},
                    "bar": {"color": colour, "thickness": 0.28},
                    "bgcolor": "rgba(0,0,0,0)",
                    "borderwidth": 0,
                    "steps": [
                        {"range": [0, 30],   "color": "rgba(16,185,129,.16)"},
                        {"range": [30, 60],  "color": "rgba(245,158,11,.16)"},
                        {"range": [60, 100], "color": "rgba(225,29,72,.16)"},
                    ],
                    "threshold": {
                        "line": {"color": colour, "width": 4},
                        "thickness": 0.82, "value": pct,
                    },
                },
            ))
            fig.update_layout(height=290, margin=dict(t=50, b=10, l=25, r=25),
                              paper_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig, use_container_width=True)

        with r2:
            st.markdown(
                f'<div class="result {cls}"><h2>{head}</h2>'
                f'<p>{msg}</p></div>',
                unsafe_allow_html=True,
            )

            st.markdown("**What pushed this result**")
            flags = []
            if cp == "ASY":
                flags.append(("bad", "Asymptomatic chest pain"))
            if angina == "Y":
                flags.append(("bad", "Exercise-induced angina"))
            if slope == "Down":
                flags.append(("bad", "Downsloping ST segment"))
            elif slope == "Flat":
                flags.append(("warn", "Flat ST segment"))
            if oldpeak >= 2.0:
                flags.append(("bad", f"ST depression {oldpeak:.1f} mm"))
            elif oldpeak >= 1.0:
                flags.append(("warn", f"ST depression {oldpeak:.1f} mm"))
            if bp >= 140:
                flags.append(("bad", f"High BP {bp} mm Hg"))
            elif bp >= 130:
                flags.append(("warn", f"Elevated BP {bp} mm Hg"))
            if chol >= 240:
                flags.append(("bad", f"High cholesterol {chol}"))
            elif chol >= 200:
                flags.append(("warn", f"Borderline cholesterol {chol}"))
            if fbs == 1:
                flags.append(("warn", "Elevated fasting blood sugar"))
            if hr < (220 - age) * 0.7:
                flags.append(("warn", f"Low peak heart rate {hr} bpm"))
            if ecg != "Normal":
                flags.append(("warn", f"Abnormal resting ECG ({ecg})"))

            if not flags:
                flags = [("ok", "No individual red flags detected")]

            chips = "".join(
                f'<span class="chip chip-{k}">{t}</span>' for k, t in flags
            )
            st.markdown(chips, unsafe_allow_html=True)

        st.markdown("###")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Risk score", f"{pct:.1f}%")
        m2.metric("Model verdict", "Disease" if label == 1 else "No disease")
        m3.metric("Peak HR vs expected",
                  f"{hr/(220-age)*100:.0f}%",
                  delta=f"{hr - (220-age)} bpm", delta_color="normal")
        m4.metric("Red flags", str(sum(1 for k, _ in flags if k == "bad")))

        with st.expander("📋  See the exact values sent to the model"):
            st.dataframe(build_frame(data), use_container_width=True, hide_index=True)

        st.markdown(
            '<p class="disclaimer">⚠️ <b>This is a student machine-learning project, '
            'not a medical device.</b> The output is a statistical estimate from a '
            'KNN model trained on a public dataset of roughly 900 patients. It cannot '
            'diagnose anything. If you have symptoms or concerns about your heart, '
            'please speak to a qualified doctor.</p>',
            unsafe_allow_html=True,
        )


# ──────────────────────────────────────────────────────────────────────────
# TAB 2 — GLOSSARY
# ──────────────────────────────────────────────────────────────────────────
with tab_learn:
    st.markdown("### What each input actually means")
    st.caption("Plain-English explanations of the eleven clinical features the model uses.")

    GLOSSARY = [
        ("Age", "Demographic",
         "Age in years. Arteries stiffen and plaque accumulates over time, so risk "
         "rises steadily — noticeably after 45 in men and 55 in women."),
        ("Sex", "Demographic",
         "Biological sex, recorded as M or F. Men tend to develop coronary disease "
         "about a decade earlier; oestrogen offers women partial protection until menopause."),
        ("Chest Pain Type", "Symptom",
         "TA (typical angina) is classic exertional chest pressure relieved by rest. "
         "ATA (atypical angina) partially fits that pattern. NAP (non-anginal pain) is "
         "probably not cardiac. ASY (asymptomatic) means no chest pain — and in this "
         "dataset it is the single strongest predictor of disease, because silent "
         "blockages are only caught once they are advanced."),
        ("Resting Blood Pressure", "Vital sign",
         "Systolic pressure in mm Hg, measured seated and at rest. Below 120 is normal, "
         "120–139 is elevated, 140 and above is hypertension. Sustained high pressure "
         "damages artery walls and forces the heart to work harder."),
        ("Cholesterol", "Lab value",
         "Total serum cholesterol in mg/dL. Under 200 is desirable, 200–239 borderline, "
         "240+ high. Excess LDL cholesterol deposits in artery walls and forms the "
         "plaque that narrows coronary vessels. Note: a value of 0 in the original "
         "dataset means the reading was missing, not that cholesterol was zero."),
        ("Fasting Blood Sugar", "Lab value",
         "A yes/no flag for whether blood glucose exceeded 120 mg/dL after an overnight "
         "fast. A 'yes' points towards diabetes or pre-diabetes, which roughly doubles "
         "cardiovascular risk by damaging small blood vessels."),
        ("Resting ECG", "Test result",
         "An electrocardiogram recorded at rest. 'Normal' is a clean trace. 'ST' means "
         "ST-T wave abnormalities, which can reflect reduced blood supply to the heart "
         "muscle. 'LVH' means left ventricular hypertrophy — the main pumping chamber "
         "has thickened, usually from years of pumping against high pressure."),
        ("Max Heart Rate Achieved", "Stress test",
         "The highest heart rate reached during a supervised exercise test. Here a LOW "
         "number is the worrying one: a healthy heart should accelerate to roughly "
         "220 minus your age. Falling well short suggests the heart cannot meet demand, "
         "often because of blocked arteries or medication."),
        ("Exercise-Induced Angina", "Stress test",
         "Whether chest pain appeared during exertion. Pain that shows up reliably on "
         "exercise and fades with rest is the textbook sign of coronary arteries too "
         "narrow to supply the heart under load."),
        ("Oldpeak (ST Depression)", "Stress test",
         "How many millimetres the ECG's ST segment dropped during exercise relative to "
         "rest. Zero is ideal. Values above 1 mm suggest ischemia — the heart muscle is "
         "being starved of oxygen — and above 2 mm is considered clearly significant."),
        ("ST Slope", "Stress test",
         "The direction the ST segment takes at peak exercise. 'Up' (upsloping) is the "
         "normal, healthy response. 'Flat' is concerning. 'Down' (downsloping) carries "
         "the strongest association with significant coronary artery disease."),
    ]

    gl, gr = st.columns(2)
    for i, (name, tag, body) in enumerate(GLOSSARY):
        target = gl if i % 2 == 0 else gr
        with target:
            st.markdown(
                f'<div class="gloss"><h4>{name}<span class="tag">{tag}</span></h4>'
                f'<p>{body}</p></div>',
                unsafe_allow_html=True,
            )


# ──────────────────────────────────────────────────────────────────────────
# TAB 3 — ABOUT
# ──────────────────────────────────────────────────────────────────────────
with tab_about:
    st.markdown("### How the prediction works")
    st.markdown(
        """
**K-Nearest Neighbours (KNN)** doesn't learn equations. It memorises the training
patients, and when you submit a new profile it finds the *k* most similar patients
in that memory and takes a vote.

1. **Encode** — categorical fields (Sex, ChestPainType, RestingECG,
   ExerciseAngina, ST_Slope) become one-hot columns.
2. **Scale** — every feature is standardised, because KNN measures raw distance.
   Without scaling, cholesterol (values in the hundreds) would drown out Oldpeak
   (values between 0 and 6).
3. **Find neighbours** — the *k* closest training patients are located by
   Euclidean distance.
4. **Vote** — the majority class among those neighbours becomes the prediction;
   the proportion becomes the probability shown on the gauge.
        """
    )

    st.divider()
    st.markdown("### Wiring in your trained model")
    st.markdown(
        "Save your artifacts next to `app.py` and the demo banner disappears "
        "automatically:"
    )
    st.code(
        """import joblib

# Cleanest option — one pipeline handles scaling + prediction
joblib.dump(pipeline, "knn_model.pkl")

# Or, if you scaled separately:
joblib.dump(knn,     "knn_model.pkl")
joblib.dump(scaler,  "scaler.pkl")
joblib.dump(list(X_train.columns), "columns.pkl")   # post-get_dummies order
""",
        language="python",
    )

    st.divider()
    st.markdown("### Dataset")
    st.markdown(
        "Heart Failure Prediction Dataset — 918 patient records combined from the "
        "Cleveland, Hungarian, Switzerland, Long Beach VA and Statlog datasets, "
        "with 11 clinical features and a binary `HeartDisease` target."
    )

    st.markdown(
        '<p class="disclaimer">Built by <b>Anshu</b> · Educational project. '
        'Not a diagnostic tool and not a substitute for professional medical advice.</p>',
        unsafe_allow_html=True,
    )