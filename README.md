# iot2026-dic-7

這是一個使用 `Streamlit` 與 `scikit-learn` 製作的單檔機器學習展示專案，主題是以 `CRISP-DM` 流程說明線性迴歸（Linear Regression）如何從資料生成、資料準備、模型訓練、模型評估一路走到簡單部署。

線上展示網址：

[https://iot2026-dic-7-pytree.streamlit.app/](https://iot2026-dic-7-pytree.streamlit.app/)

## 專案在做什麼

這個專案的核心目標，是用一個可互動的網頁介面，示範機器學習中最基礎的線性迴歸流程。使用者可以在側邊欄調整資料筆數、雜訊變異與隨機種子，然後按下 `Generate Data` 生成一組合成資料，觀察模型如何從這些資料中學習出 `x` 與 `y` 的線性關係。

資料並不是從外部檔案讀取，而是依照題目要求動態生成：

- `n ∈ [100, 1000]`
- `x ~ Uniform(-100, 100)`
- `a ~ Uniform(-10, 10)`
- `b ~ Uniform(-50, 50)`
- `noise ~ Normal(mean ∈ [-10, 10], var ∈ [0, 1000])`
- `y = ax + b + noise`

這讓整個 app 可以完整展示「資料是怎麼來的」以及「模型是否學回接近真實參數」。

## CRISP-DM 六個階段

本專案在介面中依照 `CRISP-DM` 分成六個區塊：

1. `Business Understanding`
說明這個 app 的目的，是理解線性迴歸如何在含雜訊的資料下學習真實關係。

2. `Data Understanding`
顯示合成資料樣本、統計描述，以及散點圖與回歸線。

3. `Data Preparation`
先做 `train_test_split`，再對訓練資料做 `StandardScaler`，並將相同轉換套用到測試資料。

4. `Modeling`
使用 `LinearRegression` 進行模型訓練。

5. `Evaluation`
顯示 `MSE`、`RMSE`、`R²` 等評估指標，並比較真實參數與模型學到的參數。

6. `Deployment`
提供輸入 `x` 值的預測功能，並可下載訓練好的 `joblib` 模型檔。

## 主要功能

- 側邊欄互動參數控制
- 合成線性資料生成
- `train_test_split` + `StandardScaler` 標準流程
- `LinearRegression` 模型訓練
- `MSE`、`RMSE`、`R²` 模型評估
- 散點圖與回歸線視覺化
- 真實參數與學習參數對比
- 預測輸入與模型下載
- 使用 Streamlit 快取與按鈕觸發來減少不必要重算

## 技術使用

- `Streamlit`
- `NumPy`
- `Pandas`
- `scikit-learn`
- `Plotly`
- `joblib`

## 本機執行

如果要在本機啟動：

```bash
uv run streamlit run app.py
```

## 專案結構

```text
app.py         # 主程式，包含整個 Streamlit CRISP-DM 線性迴歸介面
chats.md       # 本次開發對話紀錄，含對話對象與可確認的時間資訊
pyproject.toml # 專案依賴設定
uv.lock        # uv 鎖定檔
README.md      # 專案說明
```

## 開發紀錄

本專案另附 [chats.md](C:\Users\User\Documents\Repos\iot2026-dic-7\chats.md)，整理本次與 Codex 協作的聊天記錄，包含發話對象、對話順序，以及此 session 可確認的日期與時區資訊。
