# ******************************************************************************
# Title       : BSU Learning Lab
# Theme       : Production Data Science Pipelines with Targets
# Chapter     : LAB 54: AUTOREGRESSIVE FORECASTING
# Module      : 01_recursive_basics.R
# Update Date : 2021/7/26
# URL         : https://university.business-science.io/courses/enrolled/
# ******************************************************************************


# ＜目的＞
# - 再帰的回帰の考え方を学ぶ


# ＜目次＞
# 0 準備
# 1 予測期間の日付作成
# 2 ラグ系列の作成
# 3 データ分割
# 4 モデル定義と予測
# 5 予測結果の確認


# 0 準備 -------------------------------------------------------------------------

# ライブラリ
library(modeltime)
library(tidymodels)
library(tidyverse)
library(lubridate)
library(timetk)


# データ確認
# --- 4系列
m4_monthly %>% print()
m4_monthly %>% group_by(id) %>% tally()

# プロット作成
# --- 系列ごとにデータ開始日が異なる
m4_monthly %>%
  group_by(id) %>%
  plot_time_series(date, value)


# 1 予測期間の日付作成 ----------------------------------------------------------------

# 予測ホライズン
FORECAST_HORIZON <- 24

# 将来日付の作成
m4_extended <-
  m4_monthly %>%
    group_by(id) %>%
    future_frame(.date_var = date,
                 .length_out = FORECAST_HORIZON,
                 .bind_data  = TRUE) %>%
    ungroup()

# 確認
m4_extended %>% filter(id == "M1") %>% tail(50) %>% print(n = nrow(.))
m4_extended %>% mutate(flg = is.na(value)) %>% group_by(id, flg) %>% tally()


# 2 ラグ系列の作成 -------------------------------------------------------------------

# 関数定義
# --- 原系列にラグ系列を追加する（予測ホライズンと同じ数にする）
# --- パネルデータを扱うのでグループ化を適用
lag_transformer_grouped <- function(data){
    data %>%
      group_by(id) %>%
      tk_augment_lags(value, .lags = 1:FORECAST_HORIZON) %>%
      ungroup()
}

# ラグ系列の追加
m4_lags <-
  m4_extended %>%
    lag_transformer_grouped()


# 3 データ分割 ----------------------------------------------------------------------

# 訓練データの作成
# --- NAを含む列を削除
train_data <- m4_lags %>% drop_na()

# 将来データの作成
future_data <- m4_lags %>% filter(is.na(value))

# データ確認
train_data %>% group_by(id) %>% tally()
future_data %>% group_by(id) %>% tally()


# 4 モデル定義と予測 ------------------------------------------------------------------

# ＜再帰的プロセス＞
# 1 新しいデータの最初の行の予測を計算する（最初の行には、必要な列にNAを含めることはできない）
# 2 value列のi番目の場所に計算済みの予測を入力
# 3 すでに計算された予測(value)に基づいて、特徴量となるラグ系列を作成
# 4 1-3のプロセスを予測ホライズン全ての日付に対してループ適用

# ＜参考＞
# panel_tail()
# --- 各パネル(グループ)の末尾データを返す
train_data %>% panel_tail(id, FORECAST_HORIZON) %>% group_split(id) %>% .[[1]]
train_data %>% group_split(id) %>% map(tail, FORECAST_HORIZON) %>% .[[1]]


# 線形回帰モデル
# --- 全体で回帰（グループ情報なし）
model_fit_lm <-
  linear_reg() %>%
    set_engine("lm") %>%
    fit(value ~ ., data = train_data)

# 再帰的回帰
# --- グループ情報なし
# * Recursive Linear Regression ----
model_fit_lm_recursive <-
  model_fit_lm %>%
    recursive(id         = "id",
              transform  = lag_transformer_grouped,
              train_tail = panel_tail(train_data, id, FORECAST_HORIZON))


# 5 予測結果の確認 ----------------------------------------------------------------------

# プロット作成
# --- new_dataのvalueにはNAが含まれている
modeltime_table(model_fit_lm,
                model_fit_lm_recursive) %>%
  modeltime_forecast(new_data    = future_data,
                     actual_data = m4_monthly,
                     keep_data   = TRUE) %>%
  group_by(id) %>%
  plot_modeltime_forecast(.interactive = TRUE,
                          .conf_interval_show = FALSE)

# Notice LM gives us an error while the recursive makes the forecast.
# This happens because recursive() tells the NA values to be filled
#   in use the lag transformer function.
