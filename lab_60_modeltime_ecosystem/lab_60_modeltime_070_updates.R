# ******************************************************************************
# Title       : BSU Learning Lab
# Theme       : SPECIAL: Forecasting with Modeltime!
# Chapter     : LAB 60: MODELTIME ECOSYSTEM
# Module      : lab_60_modeltime_070_updates.R
# Update Date : 2021/8/4
# URL         : https://university.business-science.io/courses/enrolled/
# ******************************************************************************


# ＜準備＞
# 0 準備
# 1 データ加工
# 2 学習用データの作成
# 3 モデル構築＆学習
# 4 モデル精度の検証
# 5 予測
# 6 XGBoostモデルの準備
# 7 並列処理による学習
# 8 モデル比較
# 9 ベースラインモデルの構築
# 10 ベースラインモデルのチューニング
# 11 ベースラインモデルを含めたモデル精度の検証
# 12 最終予測
# 13 最良モデルの決定
# 14 アンサンブルの実施


# 0 準備 ------------------------------------------------------------------------

# ライブラリ
library(tidymodels)
library(workflowsets)

library(modeltime)
library(modeltime.gluonts)
library(modeltime.ensemble)
library(timetk)

library(tidyverse)
library(lubridate)
library(janitor)


# Python Config
reticulate::py_discover_config()


# データロード
# --- requires readr >= 2.0.0
file_paths = fs::dir_ls("lab_60_modeltime_ecosystem/data", glob = "*.csv")
airline_data_raw_tbl <- read_csv(file_paths, id = "path",  name_repair = janitor::make_clean_names)

# データ確認
airline_data_raw_tbl


# 1 データ加工 ----------------------------------------------------------------

# データ整理
airline_data_tbl <-
  airline_data_raw_tbl %>%
    filter(!month %in% "TOTAL") %>%
    mutate(date = make_date(year, month)) %>%
    mutate(
        carrier = path %>%
            str_remove("lab_60_modeltime_ecosystem/data/") %>%
            str_remove(".csv") %>%
            str_replace_all("_", " ") %>%
            str_to_title()
    ) %>%
    select(carrier, date, domestic:total) %>%
    pivot_longer(cols = domestic:total, names_to = "travel_type", values_to = "value") %>%
    drop_na() %>%
    mutate(id = str_c(carrier, "_", travel_type)) %>%
    select(id, date, value)

# プロット作成
airline_data_tbl %>%
  group_by(id) %>%
  plot_time_series(date, value, .facet_ncol = 3)

# データチェック
# --- 全ての系列が同じデータ数ではない
airline_data_tbl %>%
  group_by(id) %>%
  tally()

# データチェック
# --- データ終了日が異なる
airline_data_tbl %>%
  group_by(id) %>%
  tk_summary_diagnostics()

# データ抽出
# --- 国内線のみ抽出
domestic_airline_tbl <-
  airline_data_tbl %>%
    filter(id %>% str_detect("domestic"))

# プロット作成
domestic_airline_tbl %>%
  group_by(id) %>%
  plot_time_series(date, value, .facet_ncol = 1)


# 2 学習用データの作成 ----------------------------------------------------------

# データ分割
# --- 時系列方向に日付でデータ分割
# --- 検証データ：4*6=24
splits <-
  domestic_airline_tbl %>%
    time_series_split(assess = 6, cumulative = TRUE)

# プロット確認
# --- レコードにTraining/Testingのラベルが付く
# --- 系列に関係なくプロット作成
splits %>%
  tk_time_series_cv_plan() %>%
  plot_time_series_cv_plan(date, value)


# 3 モデル構築＆学習 ------------------------------------------------------------

# DeepAR GluonTS ****************************

# モデル構築＆学習
fit_deepar_gluonts <-
  deep_ar(id = "id",
          freq = "M",
          prediction_length = 6,
          lookback_length   = 6*3,
          epochs = 10) %>%
    set_engine("gluonts_deepar") %>%
    fit(value ~ date + id, data = training(splits))

# 確認
fit_deepar_gluonts %>% print()


# NEW: DeepAR Torch *************************

#   Requires:
#   - modeltime.gluonts >= 0.3.0 (R)
#   - gluonts >= 0.8.0 (python)
#   - torch and pytorch-lightning (python)

# モデル構築＆学習
fit_deepar_torch <-
  deep_ar(id = "id",
          freq = "M",
          prediction_length = 6,
          lookback_length   = 6*3,
          epochs = 10*2) %>%
    set_engine("torch") %>%
    fit(value ~ date + id, data = training(splits))

# 確認
fit_deepar_torch %>% print()


# * NEW: GP Forecaster *********************

# モデル構築＆学習
fit_gp_forecaster <-
  gp_forecaster(id = "id",
                freq = "M",
                prediction_length = 6,
                # lookback_length   = 6*3,
                epochs = 30) %>%
    set_engine("gluonts_gp_forecaster") %>%
    fit(value ~ date + id, data = training(splits))

# 確認
fit_gp_forecaster %>% print()


# * NEW: Deep State **************************************

# モデル構築＆学習
fit_deep_state <-
  deep_state(id = "id",
             freq = "M",
             prediction_length = 6,
             lookback_length   = 6*3,
             epochs = 20) %>%
    set_engine("gluonts_deepstate") %>%
    fit(value ~ date + id, data = training(splits))

# 確認
fit_deep_state %>% print()


# 4 モデル精度の検証 ------------------------------------------------------------

# ＜ポイント＞
# - アルゴリズムごとに複数モデルを用いて予測
# - データセットに対しては1モデルしか作っていない
# - IDごとにモデル精度を検証することで、ローカルへの当てはまり具合を確認


# 検証用データの作成
# --- ID列が検証データに追加される（IDごとの精度検証が可能となる）
calib_gluonts_tbl <-
  modeltime_table(fit_deepar_gluonts,
                  fit_deepar_torch,
                  fit_gp_forecaster,
                  fit_deep_state) %>%
  modeltime_calibrate(testing(splits), id = "id")


# 予測精度の検証
# --- 全体（グローバルモデル）
calib_gluonts_tbl %>% modeltime_accuracy()


# 予測精度の計算
# --- IDごと（ローカルモデル）
accuracy_id_tbl <-
  calib_gluonts_tbl %>%
    modeltime_accuracy(acc_by_id = TRUE)

# 予測精度のテーブル表示
# --- IDごと
accuracy_id_tbl %>%
  group_by(id) %>%
  table_modeltime_accuracy()

# 予測精度
# --- IDごとの最良RMSE
accuracy_id_tbl %>%
  group_by(id) %>%
  slice_min(rmse)



# 5 予測 -------------------------------------------------------------

# * Global Confidence Intervals ----

# プロット作成
# --- グローバル予測とローカル予測が出ている
# --- グローバル予測
calib_gluonts_tbl %>%
  modeltime_forecast(new_data    = testing(splits),　
                     actual_data = domestic_airline_tbl, 
                     keep_data   = TRUE) %>% 
  group_by(id) %>%
  plot_modeltime_forecast(.facet_ncol = 1, .plotly_slider = TRUE)


# * Local Confidence Intervals ----
#   - Become more narrow depending on time series

# プロット作成
# --- IDごとに信頼区間を設定
calib_gluonts_tbl %>%
  modeltime_forecast(new_data    = testing(splits), 
                     actual_data = domestic_airline_tbl, 
                     conf_by_id  = TRUE) %>%
  group_by(id) %>%
  plot_modeltime_forecast(.facet_ncol = 1, .plotly_slider = TRUE)



# 6 XGBoostモデルの準備 -----------------------------------------------

# ＜ポイント＞
# - チューニングとワークフローセットを用いてスケーラブルな学習を行う
# - 並列処理により高速化を行う


# * 特徴量エンジニアリング ----------------

# レシピ作成
recipe_spec <-
  recipe(value ~ ., training(splits)) %>%
    step_timeseries_signature(date) %>%
    step_rm(date) %>%
    step_rm(matches("(xts$)|(iso$)")) %>%
    step_zv(all_predictors()) %>%
    step_dummy(all_nominal_predictors(), one_hot = TRUE)

# データ確認
recipe_spec %>% prep() %>% juice() %>% glimpse()



# * チューニング・グリッドの作成 ------------

# グリッド作成
# --- modeltime::create_model_grid()
grid_tbl <-
  tibble(learn_rate = c(0.001, 0.010, 0.100, 0.350, 0.500, 0.650, 0.75, 0.9, 1.0)) %>%
    create_model_grid(f_model_spec = boost_tree,
                      engine_name  = "xgboost",
                      mode         = "regression")

# # グリッド作成
# # --- スケーラブルに耐えられるかテスト
# grid_tbl <-
#   tibble(learn_rate = seq(0, 1, 0.001)) %>%
#   create_model_grid(f_model_spec = boost_tree,
#                     engine_name  = "xgboost",
#                     mode         = "regression")

# 確認
grid_tbl
grid_tbl$.models


# * ワークフロー・セットの設定 ----

# ワークフローセットの作成
# --- 前処理
wfset <-
  workflow_set(preproc = list(recipe_spec),
               models = grid_tbl$.models,
               cross  = TRUE)


# 7 並列処理による学習 -----------------------------------------------

# ＜ポイント＞
# - {modeltime}で定義したモデルテーブルに対して並列処理を適用


# 学習設定
ctrl_par <- 
  control_fit_workflowset(verbose = TRUE, 
                          allow_par = TRUE)

# 並列処理の開始
parallel_start(6)

# 学習
modeltime_xgboost_fit <- 
  wfset %>%
    modeltime_fit_workflowset(data    = training(splits), 
                              control = ctrl_par)

# 並列処理の終了
parallel_stop()


# 確認
modeltime_xgboost_fit %>% print()


# 8 モデル比較 -------------------------------------------------------

# 検証用データの作成
# --- {modeltime.gluonts}と{workfowset}のモデルテーブルを結合
calib_tbl <- 
  combine_modeltime_tables(calib_gluonts_tbl, 
                           modeltime_xgboost_fit) %>%
    modeltime_calibrate(testing(splits), id = "id")

# モデル精度の検証
calib_tbl %>% modeltime_accuracy()

# 最良モデルの抽出
# --- IDごとにRMSEが最小なモデルを探す
calib_tbl %>%
  modeltime_accuracy(acc_by_id = TRUE) %>%
  group_by(id) %>%
  slice_min(rmse)

# 予測
calib_tbl %>%
  modeltime_forecast(new_data    = testing(splits), 
                     actual_data = domestic_airline_tbl, 
                     conf_by_id  = TRUE) %>%
  group_by(id) %>%
  plot_modeltime_forecast(.facet_ncol = 1, .plotly_slider = TRUE)


# 9 ベースラインモデルの構築 ---------------------------------------------------

# ベースラインモデル
# --- IDごとの中央値
model_median_fit <- 
  window_reg(id = "id", window_size = 6) %>%
    set_engine("window_function", window_function = median) %>%
    fit(value ~ ., data = training(splits))

model_median_fit


# ベースラインモデル
# --- IDごとの平均値
model_mean_fit <- 
  window_reg(id = "id", window_size = 6) %>%
    set_engine("window_function", window_function = mean) %>%
    fit(value ~ ., data = training(splits))

model_mean_fit

# ベースラインモデル
# --- Seasonal NAIVE by ID ----
model_snaive_fit = 
  naive_reg(seasonal_period = 12, id = "id") %>%
    set_engine("snaive") %>%
    fit(value ~ ., data = training(splits))

# モデルテーブルに登録
baseline_models <- 
  modeltime_table(model_median_fit, 
                  model_mean_fit, 
                  model_snaive_fit)

# 確認
baseline_models %>% print()


# 10 ベースラインモデルのチューニング ---------------------------------------------------

# チューニング・グリッドの作成
window_grid_tbl <-
  tibble(window_size = 1:12) %>%
    create_model_grid(f_model_spec  = window_reg, 
                      id            = "id", 
                      engine_name   = "window_function", 
                      engine_params = list(window_function = ~ median(.))
                      )

# 一括学習
# --- 結果は各リストに格納される
window_model_list <- 
  window_grid_tbl %>%
    pull(.models) %>%
    map(~ fit(., value ~ ., training(splits)))


# モデルテーブルに変換
# --- リストからテーブルに変換
baseline_window_tuned <- window_model_list %>%as_modeltime_table()


# 11 ベースラインモデルを含めたモデル精度の検証 ----------------------------------------------

# 検証データの作成
calib_baseline_tbl <- 
  calib_tbl %>%
    filter(.model_id %in% c(1, 12, 13, 10)) %>%
    combine_modeltime_tables(baseline_models,　
                             baseline_window_tuned) %>%
    modeltime_calibrate(testing(splits), id = "id")

# モデル精度の検証
# --- 全体表示
calib_baseline_tbl %>% modeltime_accuracy()

# モデル抽出
# --- 最良モデル or ベースラインモデル
best_vs_baseline_tbl <- 
  calib_baseline_tbl %>%
    modeltime_accuracy(acc_by_id = TRUE) %>%
    group_by(id) %>%
    filter(.model_id == 8 | rmse == min(rmse))

# 結果のテーブル表示
best_vs_baseline_tbl %>%
    table_modeltime_accuracy()

# 最良モデルの定義
best_model_accuracy_by_id <- best_vs_baseline_tbl



# 12 最終予測 ----------------------------------------------------------

# リフィット
# --- モデルテーブルを再学習（テーブルに抽出や変更を加えた場合に実施）
refit_tbl <- 
  calib_baseline_tbl %>%
    filter(.model_id <= 4 | .model_id == 8) %>%
    modeltime_refit(data = domestic_airline_tbl)

# 予測用データ
new_data <- 
  domestic_airline_tbl %>%
    group_by(id) %>%
    future_frame(date, .length_out = 6)

# 将来データの作成
forecast_tbl <- 
  refit_tbl %>%
    modeltime_forecast(new_data = new_data, 
                       actual_data = domestic_airline_tbl,
                       conf_by_id = TRUE)

# プロット作成
forecast_tbl %>%
  group_by(id) %>%
  plot_modeltime_forecast(.plotly_slider = TRUE)


# 13 最良モデルの決定 -------------------------------------------------------

# Select Best Forecasts ----
forecast_selection_tbl <- 
  forecast_tbl %>%
    right_join(best_model_accuracy_by_id %>% select(.model_id, id))

actual_tbl <- 
  forecast_tbl %>%
    filter(is.na(.model_id))

bind_rows(actual_tbl,　
          forecast_selection_tbl) %>%
  group_by(id) %>%
  plot_modeltime_forecast(.plotly_slider = TRUE)


# 14 アンサンブルの実施 -------------------------------------------------------

# 予測用データ
new_data <- 
  domestic_airline_tbl %>%
  group_by(id) %>%
  future_frame(date, .length_out = 6)

# アンサンブル
refit_tbl %>%
  ensemble_average() %>%
  modeltime_table() %>%
  modeltime_forecast(new_data = new_data,　
                     actual_data = domestic_airline_tbl,　
                     conf_by_id = TRUE) %>%
  group_by(id) %>%
  plot_modeltime_forecast(.plotly_slider = TRUE)

