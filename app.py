
from pathlib import Path
import pickle

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go


from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

# ============================================================
# 1. PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="SalaryWise | ML Analytics",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "linear_regression_model.pkl"
DATA_PATH = BASE_DIR / "Salary_Data.csv"

# ============================================================
# 2. ENTERPRISE DESIGN SYSTEM
# ============================================================

st.markdown("""
<style>
@import url(
'https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap'
);

:root {
    --primary: #2457C5;
    --primary-light: #EAF1FF;
    --background: #F5F7FB;
    --surface: #FFFFFF;
    --text: #17243A;
    --muted: #68768B;
    --border: #E1E7F0;
    --success: #16865C;
}

.stApp {
    background: var(--background);
    color: var(--text);
    font-family: 'Inter', sans-serif;
}

.block-container {
    max-width: 1500px;
    padding: 1.7rem 2rem 2rem;
}

[data-testid="stSidebar"] {
    background: #14233D;
}

[data-testid="stSidebar"] * {
    color: #F1F5FC;
}

[data-testid="stSidebar"] [data-testid="stRadio"] label {
    color: #E5EDFC;
}

h1, h2, h3 {
    color: var(--text);
    letter-spacing: -0.5px;
}

p, label {
    color: var(--muted);
}

[data-testid="stMetric"] {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 17px 18px;
    box-shadow: 0 3px 12px rgba(20, 35, 61, 0.035);
}

[data-testid="stMetricLabel"] {
    color: var(--muted);
    font-size: 0.83rem;
}

[data-testid="stMetricValue"] {
    color: var(--text);
    font-weight: 750;
}

[data-testid="stVerticalBlockBorderWrapper"] {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 14px;
}

.stButton > button,
.stDownloadButton > button {
    border-radius: 9px;
    min-height: 42px;
    font-weight: 600;
}

.stButton > button[kind="primary"] {
    background: var(--primary);
    border-color: var(--primary);
}

.section-label {
    color: var(--primary);
    font-size: 0.75rem;
    font-weight: 800;
    letter-spacing: 1.4px;
    text-transform: uppercase;
}

.page-description {
    color: var(--muted);
    font-size: 0.95rem;
    line-height: 1.65;
}

.info-card {
    background: white;
    border: 1px solid var(--border);
    padding: 18px;
    border-radius: 12px;
    height: 100%;
}

div[data-testid="stDataFrame"] {
    border: 1px solid var(--border);
    border-radius: 10px;
}

hr {
    border-color: var(--border);
}

footer {
    visibility: hidden;
}

@media (max-width: 800px) {
    .block-container {
        padding: 1rem 0.8rem;
    }
}
</style>
""", unsafe_allow_html=True)


# ============================================================
# 3. LOAD MODEL AND DATA
# ============================================================

@st.cache_resource
def load_model(path, modified_time):
    with open(path, "rb") as file:
        return pickle.load(file)


@st.cache_data
def load_dataset(path, modified_time):
    return pd.read_csv(path)


@st.cache_data
def create_evaluation_data(csv_path, modified_time):
    df = pd.read_csv(csv_path)

    if df.empty or df.shape[1] < 2:
        raise ValueError(
            "Dataset must contain at least one feature and one target."
        )

    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=0
    )

    return X_train, X_test, y_train, y_test


def money(value):
    return f"${value:,.2f}"


def number(value, digits=3):
    if pd.isna(value):
        return "N/A"
    return f"{value:,.{digits}f}"


model = None
dataset = None
load_error = None

if not MODEL_PATH.exists():
    load_error = (
        f"Model file not found: {MODEL_PATH.name}. "
        "Run regression_machine_learning.py first."
    )
else:
    try:
        model = load_model(
            str(MODEL_PATH), MODEL_PATH.stat().st_mtime
        )
    except Exception as exc:
        load_error = f"Could not load model: {exc}"

if not DATA_PATH.exists():
    load_error = (
        f"Dataset file not found: {DATA_PATH.name}. "
        "Place Salary_Data.csv in the same folder as app.py."
    )
else:
    try:
        dataset = load_dataset(
            str(DATA_PATH), DATA_PATH.stat().st_mtime
        )
    except Exception as exc:
        load_error = f"Could not read dataset: {exc}"

if load_error:
    st.error(load_error)
    st.info(
        "Required files: app.py, linear_regression_model.pkl, "
        "and Salary_Data.csv in the same directory."
    )
    st.stop()

if dataset.empty or dataset.shape[1] < 2:
    st.error("The CSV needs at least one feature and one target column.")
    st.stop()

FEATURES = dataset.columns[:-1].tolist()
TARGET = dataset.columns[-1]

if len(FEATURES) != 1:
    st.warning(
        "This dashboard is configured for your single-feature "
        "Salary vs Experience Linear Regression model."
    )
    st.stop()

FEATURE = FEATURES[0]

try:
    if not all(
        pd.api.types.is_numeric_dtype(dataset[c])
        for c in FEATURES + [TARGET]
    ):
        st.error("All model feature and target columns must be numeric.")
        st.stop()

    if dataset[FEATURES + [TARGET]].isna().any().any():
        st.error(
            "Missing values detected in the model columns. "
            "Clean the dataset before evaluation."
        )
        st.stop()

    X_train, X_test, y_train, y_test = create_evaluation_data(
        str(DATA_PATH), DATA_PATH.stat().st_mtime
    )

    y_train_pred = np.asarray(
        model.predict(X_train), dtype=float
    ).reshape(-1)

    y_test_pred = np.asarray(
        model.predict(X_test), dtype=float
    ).reshape(-1)

    if len(y_train_pred) != len(y_train):
        raise ValueError("Training prediction count does not match.")

    if len(y_test_pred) != len(y_test):
        raise ValueError("Test prediction count does not match.")

except Exception as exc:
    st.error(f"Model evaluation failed: {exc}")
    st.stop()


# ============================================================
# 4. CALCULATE MODEL METRICS
# ============================================================

train_r2 = r2_score(y_train, y_train_pred)
test_r2 = r2_score(y_test, y_test_pred)

mae = mean_absolute_error(y_test, y_test_pred)
mse = mean_squared_error(y_test, y_test_pred)
rmse = np.sqrt(mse)

# Sum of Squares calculated on the held-out test set.
test_mean = float(np.mean(y_test))

ssr = float(np.sum((y_test_pred - test_mean) ** 2))
sse = float(np.sum((y_test - y_test_pred) ** 2))
sst = float(np.sum((y_test - test_mean) ** 2))

# R-squared is based on residual sum of squares.
r_squared = 1 - (sse / sst) if sst != 0 else np.nan

slope = float(np.asarray(model.coef_).reshape(-1)[0])
intercept = float(np.asarray(model.intercept_).reshape(-1)[0])

equation_sign = "+" if intercept >= 0 else "-"
equation = (
    f"Salary = {slope:,.2f} × Experience "
    f"{equation_sign} {abs(intercept):,.2f}"
)

actual_vs_predicted = pd.DataFrame({
    "Actual Salary": np.asarray(y_test),
    "Predicted Salary": y_test_pred,
    "Residual": np.asarray(y_test) - y_test_pred
})

actual_vs_predicted["Absolute Error"] = (
    actual_vs_predicted["Residual"].abs()
)


# ============================================================
# 5. SIDEBAR NAVIGATION
# ============================================================

with st.sidebar:
    st.markdown("## 📈 SalaryWise")
    st.caption("MACHINE LEARNING ANALYTICS")
    st.divider()

    page = st.radio(
        "WORKSPACE",
        [
            "Overview",
            "Salary Predictor",
            "Model Insights",
            "Model Evaluation",
            "Statistical Analysis",
            "Data Explorer"
        ],
        key="navigation"
    )

    st.divider()

    st.markdown("**MODEL INFORMATION**")
    st.caption("Algorithm: Linear Regression")
    st.caption(f"Feature: {FEATURE}")
    st.caption(f"Target: {TARGET}")
    st.caption(f"Dataset: {len(dataset):,} records")

    st.divider()
    st.caption("SalaryWise · ML Learning Project")


# ============================================================
# 6. PAGE HEADER
# ============================================================

st.markdown(
    '<div class="section-label">MACHINE LEARNING WORKSPACE</div>',
    unsafe_allow_html=True
)

st.title(page)

st.markdown(
    '<div class="page-description">'
    'Explore salary predictions, regression mathematics, '
    'model quality and statistical properties in one workspace.'
    '</div>',
    unsafe_allow_html=True
)

st.write("")


# ============================================================
# 7. OVERVIEW
# ============================================================

if page == "Overview":

    st.subheader("Model Performance")

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Dataset Records", f"{len(dataset):,}")
    c2.metric("Test R²", f"{test_r2:.3f}")
    c3.metric("Test MAE", money(mae))
    c4.metric("Test RMSE", money(rmse))

    st.write("")

    left, right = st.columns([1.35, 1])

    with left:
        st.markdown("### Actual vs. Predicted")

        fig = px.scatter(
            actual_vs_predicted,
            x="Actual Salary",
            y="Predicted Salary",
            title="Test Set Prediction Quality",
            hover_data=["Residual", "Absolute Error"]
        )

        low = float(min(
            actual_vs_predicted["Actual Salary"].min(),
            actual_vs_predicted["Predicted Salary"].min()
        ))

        high = float(max(
            actual_vs_predicted["Actual Salary"].max(),
            actual_vs_predicted["Predicted Salary"].max()
        ))

        fig.add_trace(go.Scatter(
            x=[low, high],
            y=[low, high],
            mode="lines",
            name="Ideal prediction",
            line=dict(dash="dash")
        ))

        fig.update_layout(
            template="plotly_white",
            height=370,
            margin=dict(l=15, r=15, t=55, b=15)
        )

        st.plotly_chart(fig, use_container_width=True)

    with right:
        st.markdown("### Regression Equation")

        st.markdown(
            f'<div class="info-card">'
            f'<div class="section-label">FITTED MODEL</div>'
            f'<h3>{slope:,.2f} × X {equation_sign} '
            f'{abs(intercept):,.2f}</h3>'
            f'<p>Where X = {FEATURE} and the output is {TARGET}.</p>'
            f'</div>',
            unsafe_allow_html=True
        )

        st.write("")

        st.markdown("### Training vs. Test Score")

        score_df = pd.DataFrame({
            "Dataset": ["Training", "Test"],
            "R² Score": [train_r2, test_r2]
        })

        fig = px.bar(
            score_df,
            x="Dataset",
            y="R² Score",
            color="Dataset",
            text_auto=".3f",
            title="R² Comparison"
        )

        fig.update_layout(
            template="plotly_white",
            showlegend=False,
            height=280
        )

        st.plotly_chart(fig, use_container_width=True)

    st.markdown("### What do these metrics mean?")

    a, b, c = st.columns(3)

    a.info(
        "**R² Score**\n\n"
        "Measures how well predictions explain the variation "
        "in actual salary. A value closer to 1 is generally better."
    )

    b.info(
        "**MAE**\n\n"
        "The average absolute difference between actual and "
        "predicted salary, in the target's units."
    )

    c.info(
        "**RMSE**\n\n"
        "The square root of mean squared error. It penalizes "
        "large prediction errors more strongly than MAE."
    )


# ============================================================
# 8. SALARY PREDICTOR
# ============================================================

elif page == "Salary Predictor":

    st.subheader("Estimate an Annual Salary")

    st.markdown(
        "Enter an experience value to get a prediction from your "
        "saved Linear Regression model."
    )

    with st.container(border=True):
        experience = st.number_input(
            f"{FEATURE} (years)",
            min_value=0.0,
            max_value=50.0,
            value=float(dataset[FEATURE].median()),
            step=0.5,
            key="prediction_experience"
        )

        predict_clicked = st.button(
            "Predict Salary",
            type="primary",
            use_container_width=True
        )

    if predict_clicked or "latest_prediction" not in st.session_state:
        prediction = float(
            np.asarray(
                model.predict(
                    pd.DataFrame(
                        [[experience]],
                        columns=FEATURES
                    )
                )
            ).reshape(-1)[0]
        )

        st.session_state.latest_prediction = prediction
        st.session_state.latest_experience = experience

    prediction = st.session_state.latest_prediction
    used_experience = st.session_state.latest_experience

    st.write("")

    p1, p2, p3 = st.columns(3)

    p1.metric("Predicted Salary", money(prediction))
    p2.metric("Experience", f"{used_experience:.1f} years")
    p3.metric("Model Slope", money(slope) + " / year")

    st.caption(
        "Predictions outside the training data's experience range "
        "are extrapolations and may be unreliable. Salary is displayed "
        "in the same units and currency as the training dataset."
    )

    st.divider()

    st.markdown("### Compare Experience Scenarios")

    max_experience = float(dataset[FEATURE].max())

    scenarios = st.multiselect(
        "Choose experience values to compare",
        options=sorted(set(
            [0, 2, 5, 8, 10, 12, 15, 20] +
            [int(v) for v in dataset[FEATURE].dropna().tolist()]
        )),
        default=[2, 5, 10, 12, 20],
        help="Includes example values and experience values found in the dataset."
    )

    if scenarios:
        scenario_df = pd.DataFrame({FEATURE: scenarios})

        scenario_df["Predicted Salary"] = np.asarray(
            model.predict(scenario_df[FEATURES])
        ).reshape(-1)

        scenario_df["Within Training Range"] = scenario_df[FEATURE].between(
            dataset[FEATURE].min(), max_experience
        )

        st.dataframe(
            scenario_df,
            use_container_width=True,
            hide_index=True
        )

        fig = px.line(
            scenario_df.sort_values(FEATURE),
            x=FEATURE,
            y="Predicted Salary",
            markers=True,
            title="Predicted Salary by Experience"
        )

        fig.update_layout(template="plotly_white", height=350)
        st.plotly_chart(fig, use_container_width=True)

        st.download_button(
            "Download Scenario Results",
            data=scenario_df.to_csv(index=False).encode("utf-8"),
            file_name="salary_scenarios.csv",
            mime="text/csv"
        )


# ============================================================
# 9. MODEL INSIGHTS
# ============================================================

elif page == "Model Insights":

    st.subheader("Understand the Regression Equation")

    st.latex(r"\hat{y} = b_0 + b_1 x")

    st.markdown(
        f"""
        - **x:** {FEATURE}
        - **ŷ:** Predicted {TARGET}
        - **b₀ (intercept):** {intercept:,.4f}
        - **b₁ (slope):** {slope:,.4f}
        """
    )

    st.code(equation)

    m1, m2 = st.columns(2)

    m1.metric("Slope / Coefficient", number(slope, 4))
    m2.metric("Intercept", number(intercept, 4))

    st.markdown("### What does the slope mean?")

    st.info(
        f"For each additional year of {FEATURE}, the model's predicted "
        f"{TARGET} changes by approximately {money(slope)}, assuming "
        "the fitted linear relationship remains appropriate."
    )

    st.markdown("### Predictions at 12 and 20 years")

    requested_experience = pd.DataFrame({
        FEATURE: [12.0, 20.0]
    })

    requested_experience["Predicted Salary"] = np.asarray(
        model.predict(requested_experience[FEATURES])
    ).reshape(-1)

    st.dataframe(
        requested_experience,
        use_container_width=True,
        hide_index=True
    )

    st.markdown("### Regression Line")

    x_min = float(dataset[FEATURE].min())
    x_max = float(dataset[FEATURE].max())

    x_values = np.linspace(x_min, x_max, 150)

    line_df = pd.DataFrame({FEATURE: x_values})

    line_df["Predicted Salary"] = np.asarray(
        model.predict(line_df[FEATURES])
    ).reshape(-1)

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=dataset[FEATURE],
        y=dataset[TARGET],
        mode="markers",
        name="Actual observations",
        marker=dict(size=9, opacity=0.7)
    ))

    fig.add_trace(go.Scatter(
        x=line_df[FEATURE],
        y=line_df["Predicted Salary"],
        mode="lines",
        name="Regression line",
        line=dict(width=3)
    ))

    fig.update_layout(
        template="plotly_white",
        xaxis_title=FEATURE,
        yaxis_title=TARGET,
        height=430
    )

    st.plotly_chart(fig, use_container_width=True)

    st.caption(
        "The fitted line shows the relationship learned during training. "
        "It does not prove that experience alone causes salary changes."
    )


# ============================================================
# 10. MODEL EVALUATION
# ============================================================

elif page == "Model Evaluation":

    st.subheader("Training and Test Performance")

    st.markdown(
        "The model is evaluated on the same 80/20 split configuration "
        "used by your training script (random_state=0)."
    )

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Training R²", number(train_r2))
    c2.metric("Test R²", number(test_r2))
    c3.metric("MAE", money(mae))
    c4.metric("MSE", f"{mse:,.2f}")

    st.metric("RMSE", money(rmse))

    st.markdown("### Actual vs. Predicted Salaries")

    fig = px.scatter(
        actual_vs_predicted,
        x="Actual Salary",
        y="Predicted Salary",
        hover_data=["Residual", "Absolute Error"],
        title="Test Set"
    )

    low = float(min(
        actual_vs_predicted["Actual Salary"].min(),
        actual_vs_predicted["Predicted Salary"].min()
    ))

    high = float(max(
        actual_vs_predicted["Actual Salary"].max(),
        actual_vs_predicted["Predicted Salary"].max()
    ))

    fig.add_trace(go.Scatter(
        x=[low, high],
        y=[low, high],
        mode="lines",
        name="Perfect prediction",
        line=dict(dash="dash")
    ))

    fig.update_layout(template="plotly_white", height=430)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Residual Analysis")

    st.latex(r"e_i = y_i - \hat{y}_i")

    left, right = st.columns(2)

    with left:
        fig = px.scatter(
            actual_vs_predicted,
            x="Predicted Salary",
            y="Residual",
            title="Residuals vs. Predicted Salary"
        )

        fig.add_hline(y=0, line_dash="dash")
        fig.update_layout(template="plotly_white", height=350)
        st.plotly_chart(fig, use_container_width=True)

    with right:
        fig = px.histogram(
            actual_vs_predicted,
            x="Residual",
            nbins=15,
            title="Residual Distribution"
        )

        fig.update_layout(template="plotly_white", height=350)
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Prediction Details")

    st.dataframe(
        actual_vs_predicted,
        use_container_width=True,
        hide_index=True
    )

    st.download_button(
        "Download Test Predictions",
        data=actual_vs_predicted.to_csv(index=False).encode("utf-8"),
        file_name="model_test_predictions.csv",
        mime="text/csv"
    )

    st.markdown("### Training Score vs. Test Score")

    score_gap = train_r2 - test_r2

    st.write(f"Training R² minus test R²: **{score_gap:.4f}**")

    st.caption(
        "A large positive gap may indicate overfitting or a mismatch "
        "between training and test data. Similar scores are encouraging, "
        "but they do not by themselves prove that the model generalizes."
    )


# ============================================================
# 11. STATISTICAL ANALYSIS
# ============================================================

elif page == "Statistical Analysis":

    st.subheader("Descriptive Statistics")

    selected_column = st.selectbox(
        "Select a numeric column",
        dataset.select_dtypes(include="number").columns.tolist()
    )

    series = dataset[selected_column].dropna()

    stats = {
        "Count": series.count(),
        "Mean": series.mean(),
        "Median": series.median(),
        "Mode": (
            ", ".join(map(str, series.mode().tolist()))
            if not series.mode().empty else "N/A"
        ),
        "Variance (sample)": series.var(),
        "Standard Deviation (sample)": series.std(),
        "Skewness": series.skew(),
        "Minimum": series.min(),
        "25th Percentile": series.quantile(0.25),
        "75th Percentile": series.quantile(0.75),
        "Maximum": series.max()
    }

    stat_df = pd.DataFrame(
        [{"Statistic": key, "Value": value}
         for key, value in stats.items()]
    )

    st.dataframe(stat_df, use_container_width=True, hide_index=True)

    left, right = st.columns(2)

    with left:
        fig = px.histogram(
            dataset,
            x=selected_column,
            nbins=25,
            marginal="box",
            title=f"Distribution of {selected_column}"
        )

        fig.update_layout(template="plotly_white", height=390)
        st.plotly_chart(fig, use_container_width=True)

    with right:
        fig = px.box(
            dataset,
            y=selected_column,
            points="outliers",
            title=f"Box Plot: {selected_column}"
        )

        fig.update_layout(template="plotly_white", height=390)
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Correlation Matrix")

    numeric_df = dataset.select_dtypes(include="number")

    if numeric_df.shape[1] >= 2:
        correlation = numeric_df.corr()

        fig = px.imshow(
            correlation,
            text_auto=".2f",
            aspect="auto",
            color_continuous_scale="RdBu_r",
            zmin=-1,
            zmax=1,
            title="Pearson Correlation"
        )

        fig.update_layout(template="plotly_white", height=450)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("At least two numeric columns are needed for correlation.")

    st.markdown("### Regression Sum of Squares")

    st.caption(
        "These values are calculated using the held-out test set."
    )

    s1, s2, s3, s4 = st.columns(4)

    s1.metric("SSR", f"{ssr:,.2f}")
    s2.metric("SSE", f"{sse:,.2f}")
    s3.metric("SST", f"{sst:,.2f}")
    s4.metric("R²", number(r_squared, 4))

    st.latex(r"SST = \sum_i (y_i - \bar{y})^2")
    st.latex(r"SSE = \sum_i (y_i - \hat{y}_i)^2")
    st.latex(r"SSR = \sum_i (\hat{y}_i - \bar{y})^2")

    st.info(
        "For ordinary least-squares regression with an intercept, "
        "SST = SSR + SSE (up to numerical precision). "
        "R² = 1 − SSE/SST. SSR/SST is not the correct general "
        "formula for calculating R²."
    )


# ============================================================
# 12. DATA EXPLORER
# ============================================================

elif page == "Data Explorer":

    st.subheader("Explore Your Dataset")

    total_rows, total_columns = dataset.shape

    a, b, c, d = st.columns(4)

    a.metric("Rows", f"{total_rows:,}")
    b.metric("Columns", f"{total_columns:,}")
    c.metric("Missing Cells", f"{int(dataset.isna().sum().sum()):,}")
    d.metric("Duplicate Rows", f"{int(dataset.duplicated().sum()):,}")

    st.markdown("### Dataset Preview")

    st.dataframe(
        dataset,
        use_container_width=True,
        hide_index=True
    )

    st.download_button(
        "Download Dataset CSV",
        data=dataset.to_csv(index=False).encode("utf-8"),
        file_name="salary_dataset.csv",
        mime="text/csv"
    )

    st.markdown("### Column Information")

    column_info = pd.DataFrame({
        "Column": dataset.columns,
        "Data Type": [str(t) for t in dataset.dtypes],
        "Non-null Count": dataset.notna().sum().values,
        "Missing Count": dataset.isna().sum().values,
        "Unique Values": dataset.nunique().values
    })

    st.dataframe(
        column_info,
        use_container_width=True,
        hide_index=True
    )

    st.markdown("### Filter Dataset")

    numeric_columns = dataset.select_dtypes(
        include="number"
    ).columns.tolist()

    if numeric_columns:
        filter_column = st.selectbox(
            "Choose numeric column",
            numeric_columns
        )

        col_min = float(dataset[filter_column].min())
        col_max = float(dataset[filter_column].max())

        if col_min < col_max:
            selected_range = st.slider(
                "Select value range",
                min_value=col_min,
                max_value=col_max,
                value=(col_min, col_max)
            )

            filtered_df = dataset[
                dataset[filter_column].between(
                    selected_range[0], selected_range[1]
                )
            ]
        else:
            filtered_df = dataset.copy()

        st.write(f"Matching records: **{len(filtered_df):,}**")

        st.dataframe(
            filtered_df,
            use_container_width=True,
            hide_index=True
        )

        st.download_button(
            "Download Filtered Data",
            data=filtered_df.to_csv(index=False).encode("utf-8"),
            file_name="filtered_salary_data.csv",
            mime="text/csv"
        )


# ============================================================
# 13. FOOTER
# ============================================================

st.divider()

st.caption(
    "SalaryWise ML Analytics · Linear Regression · "
    "Predictions are estimates, not guaranteed compensation."
)
