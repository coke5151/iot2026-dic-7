from __future__ import annotations

from io import BytesIO
from typing import Any

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


st.set_page_config(
    page_title="CRISP-DM Linear Regression Demo",
    page_icon="📈",
    layout="wide",
)


DEFAULT_CONFIG = {
    "seed": 42,
    "n_samples": 300,
    "a": 3.5,
    "b": 12.0,
    "noise_mean": 0.0,
    "noise_variance": 120.0,
    "test_size": 0.2,
}


@st.cache_data(show_spinner=False)
def generate_synthetic_data(
    n_samples: int,
    a: float,
    b: float,
    noise_mean: float,
    noise_variance: float,
    seed: int,
) -> tuple[pd.DataFrame, np.ndarray]:
    rng = np.random.default_rng(seed)
    x = rng.uniform(-100, 100, size=n_samples)
    noise = rng.normal(
        loc=noise_mean,
        scale=np.sqrt(noise_variance),
        size=n_samples,
    )
    y = a * x + b + noise
    data = pd.DataFrame({"x": x, "y": y, "noise": noise})
    return data, y


@st.cache_data(show_spinner=False)
def summarize_dataset(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    return data.head(10), data.describe().T


@st.cache_data(show_spinner=False)
def build_plot_frames(
    data: pd.DataFrame,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    y_test_pred: np.ndarray,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    scatter_df = data if len(data) <= 600 else data.sample(600, random_state=0)
    train_plot_df = pd.DataFrame({"x": X_train["x"].to_numpy(), "y": y_train.to_numpy()})
    test_plot_df = pd.DataFrame({"x": X_test["x"].to_numpy(), "y": y_test.to_numpy()})
    eval_df = pd.DataFrame(
        {
            "Actual": y_test.to_numpy(),
            "Predicted": y_test_pred,
            "Residual": y_test.to_numpy() - y_test_pred,
        }
    )
    return scatter_df, train_plot_df, eval_df


@st.cache_resource(show_spinner=False)
def train_pipeline(
    data: pd.DataFrame,
    test_size: float,
    seed: int,
) -> dict[str, Any]:
    X = data[["x"]]
    y = data["y"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
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


def build_model_artifact(
    model: LinearRegression,
    scaler: StandardScaler,
    config: dict[str, float | int],
) -> bytes:
    buffer = BytesIO()
    joblib.dump(
        {
            "model": model,
            "scaler": scaler,
            "config": config,
        },
        buffer,
    )
    return buffer.getvalue()


def main() -> None:
    st.title("CRISP-DM 線性迴歸示範")
    st.caption("以合成資料展示從業務理解到部署的完整機器學習流程。")

    if "active_config" not in st.session_state:
        st.session_state.active_config = DEFAULT_CONFIG.copy()
        st.session_state.has_generated = False

    with st.sidebar:
        st.header("參數設定")
        with st.form("config_form"):
            seed = st.number_input(
                "Random seed",
                min_value=0,
                max_value=9999,
                value=st.session_state.active_config["seed"],
            )
            n_samples = st.slider(
                "樣本數 n",
                min_value=100,
                max_value=1000,
                value=st.session_state.active_config["n_samples"],
                step=50,
            )
            true_a = st.slider(
                "真實斜率 a",
                min_value=-10.0,
                max_value=10.0,
                value=st.session_state.active_config["a"],
                step=0.5,
            )
            true_b = st.slider(
                "真實截距 b",
                min_value=-50.0,
                max_value=50.0,
                value=st.session_state.active_config["b"],
                step=1.0,
            )
            noise_mean = st.slider(
                "雜訊平均",
                min_value=-10.0,
                max_value=10.0,
                value=st.session_state.active_config["noise_mean"],
                step=0.5,
            )
            noise_variance = st.slider(
                "雜訊變異",
                min_value=0.0,
                max_value=1000.0,
                value=st.session_state.active_config["noise_variance"],
                step=10.0,
            )
            test_size = st.slider(
                "測試集比例",
                min_value=0.1,
                max_value=0.4,
                value=st.session_state.active_config["test_size"],
                step=0.05,
            )
            submitted = st.form_submit_button("Generate synthetic data", use_container_width=True)

        prediction_input = st.number_input(
            "部署測試 x 值",
            min_value=-100.0,
            max_value=100.0,
            value=25.0,
            step=1.0,
        )
        st.caption("調整參數後按下 Generate，避免每次拖拉滑桿都重跑整個流程。")

    if submitted:
        st.session_state.active_config = {
            "seed": int(seed),
            "n_samples": int(n_samples),
            "a": float(true_a),
            "b": float(true_b),
            "noise_mean": float(noise_mean),
            "noise_variance": float(noise_variance),
            "test_size": float(test_size),
        }
        st.session_state.has_generated = True

    st.header("1. Business Understanding 業務理解")
    st.write(
        "目標是用線性迴歸學習單一特徵 `x` 與目標 `y` 的線性關係，"
        "理解模型是否能在含噪聲資料下有效還原真實斜率與截距。"
    )

    if not st.session_state.has_generated:
        st.info("請先在左側設定參數並按下 `Generate synthetic data` 產生資料，之後才會進行建模、評估與部署。")
        return

    config = st.session_state.active_config
    data, _ = generate_synthetic_data(
        n_samples=config["n_samples"],
        a=config["a"],
        b=config["b"],
        noise_mean=config["noise_mean"],
        noise_variance=config["noise_variance"],
        seed=config["seed"],
    )
    summary_head, summary_stats = summarize_dataset(data)
    training_result = train_pipeline(
        data=data,
        test_size=config["test_size"],
        seed=config["seed"],
    )

    X_train = training_result["X_train"]
    X_test = training_result["X_test"]
    y_train = training_result["y_train"]
    y_test = training_result["y_test"]
    X_train_scaled = training_result["X_train_scaled"]
    model = training_result["model"]
    scaler = training_result["scaler"]
    y_train_pred = training_result["y_train_pred"]
    y_test_pred = training_result["y_test_pred"]

    metrics_df = pd.DataFrame(
        [
            {
                "Dataset": "Train",
                "MAE": mean_absolute_error(y_train, y_train_pred),
                "RMSE": root_mean_squared_error(y_train, y_train_pred),
                "R²": r2_score(y_train, y_train_pred),
            },
            {
                "Dataset": "Test",
                "MAE": mean_absolute_error(y_test, y_test_pred),
                "RMSE": root_mean_squared_error(y_test, y_test_pred),
                "R²": r2_score(y_test, y_test_pred),
            },
        ]
    )

    comparison_df = pd.DataFrame(
        [
            {"Parameter": "Slope (a)", "Ground Truth": config["a"], "Learned": model.coef_[0]},
            {"Parameter": "Intercept (b)", "Ground Truth": config["b"], "Learned": model.intercept_},
            {"Parameter": "Noise Mean", "Ground Truth": config["noise_mean"], "Learned": data["noise"].mean()},
            {
                "Parameter": "Noise Variance",
                "Ground Truth": config["noise_variance"],
                "Learned": data["noise"].var(ddof=0),
            },
        ]
    )

    predicted_y = model.predict(scaler.transform([[prediction_input]]))[0]
    scatter_df, train_plot_df, eval_df = build_plot_frames(
        data=data,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        y_test_pred=y_test_pred,
    )
    test_plot_df = pd.DataFrame({"x": X_test["x"].to_numpy(), "y": y_test.to_numpy()})

    info_col1, info_col2, info_col3 = st.columns(3)
    info_col1.metric("目前樣本數", config["n_samples"])
    info_col2.metric("測試比例", f'{config["test_size"]:.2f}')
    info_col3.metric("目前 Random Seed", config["seed"])

    st.header("2. Data Understanding 數據理解")
    left_col, right_col = st.columns([1, 1.2])
    with left_col:
        st.subheader("資料樣本")
        st.dataframe(summary_head, use_container_width=True)
        st.subheader("統計描述")
        st.dataframe(summary_stats, use_container_width=True)
    with right_col:
        scatter_fig = px.scatter(
            scatter_df,
            x="x",
            y="y",
            color="noise",
            color_continuous_scale="Viridis",
            title="Synthetic Data Distribution",
            render_mode="webgl",
        )
        scatter_fig.update_layout(height=480)
        st.plotly_chart(scatter_fig, use_container_width=True)

    st.header("3. Data Preparation 數據準備")
    prep_col1, prep_col2, prep_col3 = st.columns(3)
    prep_col1.metric("Train Size", len(X_train))
    prep_col2.metric("Test Size", len(X_test))
    prep_col3.metric("Scaled Train Mean", f"{X_train_scaled.mean():.4f}")

    prep_preview = pd.DataFrame(
        {
            "x_original": X_train["x"].head(10).to_numpy(),
            "x_scaled": X_train_scaled[:10, 0],
            "y_train": y_train.head(10).to_numpy(),
        }
    )
    st.write("先做 `train_test_split`，再只用訓練集擬合 `StandardScaler`，最後將相同轉換套用到測試集。")
    st.dataframe(prep_preview, use_container_width=True)

    st.header("4. Modeling 模型構建")
    model_col1, model_col2 = st.columns([1, 1.3])
    with model_col1:
        st.metric("Learned Slope", f"{model.coef_[0]:.4f}")
        st.metric("Learned Intercept", f"{model.intercept_:.4f}")
        st.code(
            "model = LinearRegression()\n"
            "model.fit(X_train_scaled, y_train)",
            language="python",
        )
    with model_col2:
        line_x = np.linspace(data["x"].min(), data["x"].max(), 200)
        line_y = model.predict(scaler.transform(line_x.reshape(-1, 1)))
        line_fig = go.Figure()
        line_fig.add_trace(
            go.Scatter(
                x=train_plot_df["x"],
                y=train_plot_df["y"],
                mode="markers",
                name="Train",
                marker={"color": "#1f77b4", "opacity": 0.65},
            )
        )
        line_fig.add_trace(
            go.Scatter(
                x=test_plot_df["x"],
                y=test_plot_df["y"],
                mode="markers",
                name="Test",
                marker={"color": "#ff7f0e", "opacity": 0.75},
            )
        )
        line_fig.add_trace(
            go.Scatter(
                x=line_x,
                y=line_y,
                mode="lines",
                name="Prediction Line",
                line={"color": "#111111", "width": 3},
            )
        )
        line_fig.update_layout(title="Train/Test Split with Learned Regression Line", height=480)
        st.plotly_chart(line_fig, use_container_width=True)

    st.header("5. Evaluation 評估")
    eval_col1, eval_col2 = st.columns([1, 1.1])
    with eval_col1:
        st.subheader("評估指標")
        st.dataframe(metrics_df, use_container_width=True)
        st.subheader("參數對比")
        st.dataframe(comparison_df, use_container_width=True)
    with eval_col2:
        eval_df = pd.DataFrame(
            {
                "Actual": y_test.to_numpy(),
                "Predicted": y_test_pred,
                "Residual": y_test.to_numpy() - y_test_pred,
            }
        )
        residual_fig = px.scatter(
            eval_df,
            x="Actual",
            y="Predicted",
            color="Residual",
            color_continuous_scale="RdBu",
            title="Actual vs Predicted on Test Set",
            render_mode="webgl",
        )
        residual_fig.add_shape(
            type="line",
            x0=eval_df["Actual"].min(),
            y0=eval_df["Actual"].min(),
            x1=eval_df["Actual"].max(),
            y1=eval_df["Actual"].max(),
            line={"dash": "dash", "color": "black"},
        )
        residual_fig.update_layout(height=480)
        st.plotly_chart(residual_fig, use_container_width=True)

    st.header("6. Deployment 部署")
    deploy_col1, deploy_col2 = st.columns([1, 1])
    with deploy_col1:
        st.subheader("即時預測")
        st.write(f"輸入 `x = {prediction_input:.2f}` 時，模型預測 `y = {predicted_y:.2f}`。")
        st.info("下載檔案包含訓練好的 LinearRegression、StandardScaler 與本次參數設定。")
    with deploy_col2:
        artifact = build_model_artifact(
            model=model,
            scaler=scaler,
            config={
                "seed": config["seed"],
                "n_samples": config["n_samples"],
                "a": config["a"],
                "b": config["b"],
                "noise_mean": config["noise_mean"],
                "noise_variance": config["noise_variance"],
                "test_size": config["test_size"],
            },
        )
        st.download_button(
            label="下載模型 artifact",
            data=artifact,
            file_name="linear_regression_artifact.joblib",
            mime="application/octet-stream",
        )


if __name__ == "__main__":
    main()
