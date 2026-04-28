from __future__ import annotations

from io import BytesIO

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


st.set_page_config(
    page_title="CRISP-DM Linear Regression Demo",
    page_icon="📈",
    layout="wide",
)


DEFAULT_STATE = {
    "n_samples": 300,
    "noise_variance": 120.0,
    "seed": 42,
}


@st.cache_data(show_spinner=False)
def generate_synthetic_data(
    n_samples: int,
    noise_variance: float,
    seed: int,
) -> tuple[pd.DataFrame, dict[str, float]]:
    rng = np.random.default_rng(seed)
    x = rng.uniform(-100, 100, size=n_samples)
    a = rng.uniform(-10, 10)
    b = rng.uniform(-50, 50)
    noise_mean = rng.uniform(-10, 10)
    noise = rng.normal(noise_mean, np.sqrt(noise_variance), size=n_samples)
    y = a * x + b + noise

    data = pd.DataFrame(
        {
            "x": x.astype(float),
            "y": y.astype(float),
            "noise": noise.astype(float),
        }
    )
    params = {
        "a": float(a),
        "b": float(b),
        "noise_mean": float(noise_mean),
        "noise_variance": float(noise_variance),
    }
    return data, params


@st.cache_resource(show_spinner=False)
def train_model(data: pd.DataFrame, seed: int) -> dict[str, object]:
    X = data[["x"]]
    y = data["y"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=seed,
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = LinearRegression()
    model.fit(X_train_scaled, y_train)

    y_train_pred = model.predict(X_train_scaled)
    y_test_pred = model.predict(X_test_scaled)

    return {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "X_train_scaled": X_train_scaled,
        "X_test_scaled": X_test_scaled,
        "model": model,
        "scaler": scaler,
        "y_train_pred": y_train_pred,
        "y_test_pred": y_test_pred,
    }


@st.cache_data(show_spinner=False)
def build_metrics(y_true: pd.Series, y_pred: np.ndarray) -> dict[str, float]:
    mse = mean_squared_error(y_true, y_pred)
    rmse = float(np.sqrt(mse))
    r2 = r2_score(y_true, y_pred)
    return {"MSE": float(mse), "RMSE": rmse, "R²": float(r2)}


def build_download_bytes(
    model: LinearRegression,
    scaler: StandardScaler,
    params: dict[str, float],
) -> bytes:
    buffer = BytesIO()
    joblib.dump(
        {
            "model": model,
            "scaler": scaler,
            "true_parameters": params,
        },
        buffer,
    )
    return buffer.getvalue()


def render_sidebar() -> tuple[bool, int, float, int, float]:
    with st.sidebar:
        st.header("Data Controls")
        with st.form("controls"):
            n_samples = st.slider("n", min_value=100, max_value=1000, value=DEFAULT_STATE["n_samples"], step=50)
            noise_variance = st.slider(
                "Noise Variance",
                min_value=0.0,
                max_value=1000.0,
                value=DEFAULT_STATE["noise_variance"],
                step=10.0,
            )
            seed = st.slider("Seed", min_value=0, max_value=9999, value=DEFAULT_STATE["seed"], step=1)
            submitted = st.form_submit_button("Generate Data", use_container_width=True)

        prediction_x = st.number_input(
            "Prediction Input x",
            min_value=-100.0,
            max_value=100.0,
            value=25.0,
            step=1.0,
        )
        st.caption("Use Generate Data to refresh the dataset and retrain the model.")

    return submitted, n_samples, noise_variance, seed, prediction_x


def render_scatter_and_line(
    data: pd.DataFrame,
    model: LinearRegression,
    scaler: StandardScaler,
) -> go.Figure:
    plot_df = data if len(data) <= 700 else data.sample(700, random_state=0)
    line_x = np.linspace(data["x"].min(), data["x"].max(), 200)
    line_y = model.predict(scaler.transform(line_x.reshape(-1, 1)))

    fig = go.Figure()
    fig.add_trace(
        go.Scattergl(
            x=plot_df["x"],
            y=plot_df["y"],
            mode="markers",
            name="Synthetic Data",
            marker={"color": plot_df["noise"], "colorscale": "Viridis", "showscale": True, "size": 7, "opacity": 0.7},
        )
    )
    fig.add_trace(
        go.Scatter(
            x=line_x,
            y=line_y,
            mode="lines",
            name="Regression Line",
            line={"color": "#111111", "width": 3},
        )
    )
    fig.update_layout(
        title="Scatter Plot with Regression Line",
        height=460,
        margin={"l": 10, "r": 10, "t": 50, "b": 10},
        xaxis_title="x",
        yaxis_title="y",
    )
    return fig


def main() -> None:
    st.title("CRISP-DM Linear Regression Workflow")
    st.caption("Single-file Streamlit app demonstrating linear regression with synthetic data and scikit-learn.")

    if "run_config" not in st.session_state:
        st.session_state.run_config = DEFAULT_STATE.copy()
        st.session_state.has_data = False

    submitted, n_samples, noise_variance, seed, prediction_x = render_sidebar()

    if submitted:
        st.session_state.run_config = {
            "n_samples": int(n_samples),
            "noise_variance": float(noise_variance),
            "seed": int(seed),
        }
        st.session_state.has_data = True

    st.subheader("1. Business Understanding")
    st.write(
        "The objective is to demonstrate how linear regression learns the relationship between a single feature "
        "`x` and target `y` under the CRISP-DM workflow, while comparing the true generating parameters with the learned model."
    )

    if not st.session_state.has_data:
        st.info("Set the sidebar controls and click `Generate Data` to create a dataset and run the workflow.")
        return

    config = st.session_state.run_config
    data, true_params = generate_synthetic_data(
        n_samples=config["n_samples"],
        noise_variance=config["noise_variance"],
        seed=config["seed"],
    )
    result = train_model(data, config["seed"])

    model = result["model"]
    scaler = result["scaler"]
    X_train = result["X_train"]
    X_test = result["X_test"]
    y_train = result["y_train"]
    y_test = result["y_test"]
    X_train_scaled = result["X_train_scaled"]
    y_train_pred = result["y_train_pred"]
    y_test_pred = result["y_test_pred"]

    train_metrics = build_metrics(y_train, y_train_pred)
    test_metrics = build_metrics(y_test, y_test_pred)

    metrics_df = pd.DataFrame(
        [
            {"Split": "Train", **train_metrics},
            {"Split": "Test", **test_metrics},
        ]
    )

    comparison_df = pd.DataFrame(
        [
            {"Parameter": "Slope (a)", "True": true_params["a"], "Learned": float(model.coef_[0])},
            {"Parameter": "Intercept (b)", "True": true_params["b"], "Learned": float(model.intercept_)},
            {"Parameter": "Noise Mean", "True": true_params["noise_mean"], "Learned": float(data["noise"].mean())},
            {
                "Parameter": "Noise Variance",
                "True": true_params["noise_variance"],
                "Learned": float(data["noise"].var(ddof=0)),
            },
        ]
    )

    prediction_y = float(model.predict(scaler.transform([[prediction_x]]))[0])

    st.subheader("2. Data Understanding")
    data_col, chart_col = st.columns([1, 1.3])
    with data_col:
        st.dataframe(data.head(10), use_container_width=True)
        st.dataframe(data.describe().T, use_container_width=True)
    with chart_col:
        st.plotly_chart(render_scatter_and_line(data, model, scaler), use_container_width=True)

    st.subheader("3. Data Preparation")
    prep_col1, prep_col2, prep_col3 = st.columns(3)
    prep_col1.metric("Train Rows", len(X_train))
    prep_col2.metric("Test Rows", len(X_test))
    prep_col3.metric("Scaled Train Mean", f"{float(X_train_scaled.mean()):.4f}")
    st.write(
        "The app first applies `train_test_split`, then fits `StandardScaler` on the training set only, "
        "and finally transforms both training and test sets."
    )

    st.subheader("4. Modeling")
    model_col1, model_col2 = st.columns([0.9, 1.1])
    with model_col1:
        st.code(
            "model = LinearRegression()\n"
            "model.fit(X_train_scaled, y_train)",
            language="python",
        )
        st.metric("Learned Slope", f"{float(model.coef_[0]):.4f}")
        st.metric("Learned Intercept", f"{float(model.intercept_):.4f}")
    with model_col2:
        st.dataframe(comparison_df, use_container_width=True)

    st.subheader("5. Evaluation")
    eval_col1, eval_col2 = st.columns([1, 1])
    with eval_col1:
        st.dataframe(metrics_df, use_container_width=True)
    with eval_col2:
        residual_df = pd.DataFrame(
            {
                "Actual": y_test.to_numpy(),
                "Predicted": y_test_pred,
            }
        )
        fig = go.Figure()
        fig.add_trace(
            go.Scattergl(
                x=residual_df["Actual"],
                y=residual_df["Predicted"],
                mode="markers",
                name="Predictions",
                marker={"color": "#1f77b4", "opacity": 0.75, "size": 8},
            )
        )
        diag_min = float(min(residual_df["Actual"].min(), residual_df["Predicted"].min()))
        diag_max = float(max(residual_df["Actual"].max(), residual_df["Predicted"].max()))
        fig.add_trace(
            go.Scatter(
                x=[diag_min, diag_max],
                y=[diag_min, diag_max],
                mode="lines",
                name="Ideal Fit",
                line={"dash": "dash", "color": "#111111"},
            )
        )
        fig.update_layout(
            title="Actual vs Predicted",
            height=420,
            margin={"l": 10, "r": 10, "t": 50, "b": 10},
            xaxis_title="Actual y",
            yaxis_title="Predicted y",
        )
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("6. Deployment")
    deploy_col1, deploy_col2 = st.columns([1, 1])
    with deploy_col1:
        st.write(f"For input `x = {prediction_x:.2f}`, the model predicts `y = {prediction_y:.2f}`.")
        st.write("The downloadable artifact contains the trained `LinearRegression` model and fitted `StandardScaler`.")
    with deploy_col2:
        st.download_button(
            label="Download joblib model",
            data=build_download_bytes(model, scaler, true_params),
            file_name="linear_regression_model.joblib",
            mime="application/octet-stream",
        )


if __name__ == "__main__":
    main()
