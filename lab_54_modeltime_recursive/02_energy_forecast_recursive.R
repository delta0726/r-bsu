# ******************************************************************************
# Title       : BSU Learning Lab
# Theme       : Production Data Science Pipelines with Targets
# Chapter     : LAB 54: AUTOREGRESSIVE FORECASTING
# Module      : 02_energy_forecast_recursive.R
# Update Date : 2021/7/26
# URL         : https://university.business-science.io/courses/enrolled/
# ******************************************************************************


# ＜目的＞
# - Recursiveを使って将来のエネルギー消費量の予測を行う
#   --- 以下の検証ではRecursiveは予測精度は劣っている（考え方を学ぶ）


# ＜目次＞
# 0 準備
# 1 データ準備
# 2 重複データへの対処
# 3 自己相関の確認
# 4 時系列のデータの準備
# 5 ラグ系列の追加
# 6 モデル構築
# 7 特徴量エンジニアリング
# 8 モデリング（GLMNET）
# 9 モデリング（XGBOOST）
# 10 モデリング（SVM）
# 11 最良モデルの比較
# 12 系列ごとの予測精度
# 13 リフィットして予測
# 14 最良モデルのみで予測


# 0 準備 -------------------------------------------------------------------------

# ライブラリ
library(modeltime)
library(tidymodels)
library(timetk)
library(lubridate)
library(tidyverse)
library(skimr)
library(janitor)


# データロード
data <- read_csv("lab_54_modeltime_recursive/data/Net_generation_for_all_sectors.csv")
data


# 1 データ準備 ---------------------------------------------------------------------

# カテゴリ列
data_description_tbl <-
  data %>%
    clean_names() %>%
    select(description:source_key) %>%
    slice(-(1:2)) %>%
    separate(description, into = c("location", "fuel"), sep = " : ")

# 確認
data_description_tbl %>% print()

# データ整理
data_pivoted_tbl <-
  data %>%
    select(-description, -units) %>%
    mutate(across(.cols = -(`source key`), as.numeric)) %>%
    pivot_longer(cols = -`source key`, names_to = "date") %>%
    drop_na() %>%
    mutate(date = my(date)) %>%
    clean_names()

# 日付情報の確認
data_pivoted_tbl %>%
  group_by(source_key) %>%
  tk_summary_diagnostics() %>%
  ungroup() %>%
  slice(10) %>%
  glimpse()


# 2 重複データへの対処 ------------------------------------------------------------

# プロット作成
# --- 重複データがあるので異常なプロット
data_pivoted_tbl %>%
  filter(source_key == "ELEC.GEN.SUN-US-99.M") %>%
  plot_time_series(date, value)

# 重複データを確認
# --- ELEC.GEN.SUN-US-99.M
data_pivoted_tbl %>%
  group_by(source_key, date) %>%
  tally() %>%
  filter(n > 1)

# 重複データへの対応
data_pivoted_tbl <- data_pivoted_tbl %>% distinct()

# Now we're good
data_pivoted_tbl %>%
  group_by(source_key) %>%
  tk_summary_diagnostics()

# プロット確認
data_pivoted_tbl %>%
  left_join(data_description_tbl %>% select(source_key, fuel)) %>%
  group_by(fuel) %>%
  plot_time_series(.date_var = date,
                     .value = value,
                   .facet_ncol = 3,
                   .smooth = F,
                   .title = "US Power Generation (Thousand MegaWatt-Hours)")


# 3 自己相関の確認 -------------------------------------------------------------------

# ACFプロット
data_pivoted_tbl %>%
  left_join(data_description_tbl %>% select(source_key, fuel)) %>%
  filter(fuel %in% unique(fuel)[1:3]) %>%
  group_by(fuel) %>%
  plot_acf_diagnostics(.date_var = date,
                       .value = value)


# 4 時系列のデータの準備 ------------------------------------------------------------

# 予測ホライズンの設定
FORECAST_HORIZON <- 24

# データ加工
# --- 将来日付の作成
# --- ラグ系列の追加
# --- ローリング平均系列の追加
data_extended_tbl <-
  data_pivoted_tbl %>%
    group_by(source_key) %>%
    future_frame(date, .length_out = FORECAST_HORIZON, .bind_data = TRUE) %>%
    tk_augment_lags(value, .lags = FORECAST_HORIZON, .names = "long_lag") %>%
    tk_augment_slidify(.value   = long_lag,
                       .f       = ~mean(.x, na.rm=TRUE),
                       .period  = c(0.5*FORECAST_HORIZON, FORECAST_HORIZON, FORECAST_HORIZON*2),
                       .align   = "center",
                       .partial = TRUE) %>%
    ungroup()

# データ確認
data_extended_tbl


# 5 ラグ系列の追加 ---------------------------------------------------------------

# 関数定義
# --- ラグ系列の追加
transformer_function <- function(data) {
    data %>%
        group_by(source_key) %>%
        tk_augment_lags(value, .lags = 1:FORECAST_HORIZON) %>%
        ungroup()
}

# ラグ系列の追加
data_lagged_tbl <- data_extended_tbl %>% transformer_function()

# 確認
data_lagged_tbl %>% glimpse()

# データ格納
# --- 学習用
# --- 予測用（将来日付）
data_prepared_tbl <- data_lagged_tbl %>% drop_na()
data_future_tbl <- data_lagged_tbl %>% filter(is.na(value))


# 6 モデル構築 -----------------------------------------------------------------

# リサンプリング
# --- クロスバリデーション用
resamples <-
  data_prepared_tbl %>%
    time_series_cv(cumulative  = TRUE,
                   assess      = FORECAST_HORIZON,
                   slice_limit = 1)

# 訓練データ
# --- 外れ値の処理（ts_clean_vec）
train_tbl <-
  resamples$splits[[1]] %>%
    training() %>%
    group_by(source_key) %>%
    mutate(value = ts_clean_vec(value, period = 12)) %>%
    ungroup()

# テストデータ
test_tbl  <-
  resamples$splits[[1]] %>%
    testing()


# データ確認
train_tbl %>% glimpse()


# 7 特徴量エンジニアリング ----------------------------------------------------

# * レシピ1：カテゴリをダミー変数化 ------------------------------

# レシピ定義
recipe_spec_lag <-
  recipe(value ~ ., data = train_tbl) %>%
    step_dummy(all_nominal()) %>%
    step_rm(date) %>%
    step_zv(all_predictors())

# データ確認
recipe_spec_lag %>% prep() %>% juice() %>% glimpse()


# * レシピ2：ラグ系列除外 + カレンダー追加 + カテゴリをダミー変数化 ----

# レシピ定義
recipe_spec_calendar <-
  recipe(value ~ .,
        data = train_tbl %>% select(-contains("lag"), contains("roll"))) %>%
    step_timeseries_signature(date) %>%
    step_dummy(all_nominal(), one_hot = TRUE) %>%
    step_normalize(date_index.num, starts_with("date_year")) %>%
    step_rm(date) %>%
    step_zv(all_predictors())

# データ確認
recipe_spec_calendar %>% prep() %>% juice() %>% glimpse()


# * レシピ3：カレンダー追加 + カテゴリをダミー変数化 -------------------

# レシピ定義
recipe_spec_hybrid <-
  recipe(value ~ ., data = train_tbl) %>%
    step_timeseries_signature(date) %>%
    step_dummy(all_nominal(), one_hot = TRUE) %>%
    step_normalize(date_index.num, starts_with("date_year")) %>%
    step_rm(date) %>%
    step_zv(all_predictors())

# データ確認
recipe_spec_hybrid %>% prep() %>% juice() %>% glimpse()


# 8 モデリング（GLMNET） ------------------------------------------------------------

# モデル定義
# --- Elastic Net
model_spec_glmnet <-
  linear_reg(penalty = 200,
             mixture = 0.99) %>%
    set_engine("glmnet")

# ワークフロー設定＆学習
# --- レシピ1：カテゴリをダミー変数化
# --- 再帰的回帰
wflw_fit_glmnet_lag <-
  workflow() %>%
    add_model(model_spec_glmnet) %>%
    add_recipe(recipe_spec_lag) %>%
    fit(train_tbl) %>%
    recursive(id         = "source_key",
              transform  = transformer_function,
              train_tail = panel_tail(train_tbl, source_key, FORECAST_HORIZON))

# ワークフロー設定＆学習
# --- レシピ2：ラグ系列除外 + カレンダー追加 + カテゴリをダミー変数化
wflw_fit_glmnet_calendar <-
  workflow() %>%
    add_model(model_spec_glmnet) %>%
    add_recipe(recipe_spec_calendar) %>%
    fit(train_tbl)

# ワークフロー設定＆学習
# --- レシピ3：カレンダー追加 + カテゴリをダミー変数化
# --- 再帰的回帰
wflw_fit_glmnet_hybrid <-
  workflow() %>%
    add_model(model_spec_glmnet) %>%
    add_recipe(recipe_spec_hybrid) %>%
    fit(train_tbl) %>%
    recursive(id = "source_key",
              transform = transformer_function,
              train_tail = panel_tail(train_tbl, source_key, FORECAST_HORIZON))

# モデルテーブル登録＆検証準備
# --- 学習済モデル
calibration_glmnet_tbl <-
  modeltime_table(wflw_fit_glmnet_lag,
                  wflw_fit_glmnet_calendar,
                  wflw_fit_glmnet_hybrid) %>%
    modeltime_calibrate(test_tbl)

# モデル精度の検証
calibration_glmnet_tbl %>% modeltime_accuracy()

# 予測データの作成
test_forecast_glmnet_tbl <-
  calibration_glmnet_tbl %>%
    modeltime_forecast(new_data = test_tbl,
                       actual_data = data_prepared_tbl,
                       keep_data = TRUE)

# プロット作成
# --- 明らかにおかしな予測が含まれている
test_forecast_glmnet_tbl %>%
  group_by(source_key) %>%
  plot_modeltime_forecast(.facet_ncol = 3,
                          .conf_interval_show = FALSE)


# 9 モデリング（XGBOOST） -----------------------------------------------------

# モデル定義
# --- XGBoost Model
model_spec_xgboost <-
  boost_tree(mode       = "regression",
             learn_rate = 0.75,
             min_n      = 1,
             tree_depth = 12,
             loss_reduction = 0.001) %>%
    set_engine("xgboost")

# ワークフロー設定＆学習
# --- レシピ1：カテゴリをダミー変数化
# --- 再帰的回帰
wflw_fit_xgboost_lag <-
  workflow() %>%
    add_model(model_spec_xgboost) %>%
    add_recipe(recipe_spec_lag) %>%
    fit(train_tbl) %>%
    recursive(id         = "source_key",
              transform  = transformer_function,
              train_tail = panel_tail(train_tbl, source_key, FORECAST_HORIZON))

# ワークフロー設定＆学習
# --- レシピ2：ラグ系列除外 + カレンダー追加 + カテゴリをダミー変数化
wflw_fit_xgboost_calendar <-
  workflow() %>%
    add_model(model_spec_xgboost) %>%
    add_recipe(recipe_spec_calendar) %>%
    fit(train_tbl)

# ワークフロー設定＆学習
# --- レシピ3：カレンダー追加 + カテゴリをダミー変数化
# --- 再帰的回帰
wflw_fit_xgboost_hybrid <-
  workflow() %>%
    add_model(model_spec_xgboost) %>%
    add_recipe(recipe_spec_hybrid) %>%
    fit(train_tbl) %>%
    recursive(id = "source_key",
              transform = transformer_function,
              train_tail = panel_tail(train_tbl, source_key, FORECAST_HORIZON))

# モデルテーブル登録＆検証準備
# --- 学習済モデル
calibration_xgboost_tbl <-
  modeltime_table(wflw_fit_xgboost_lag,
                  wflw_fit_xgboost_calendar,
                  wflw_fit_xgboost_hybrid) %>%
    modeltime_calibrate(test_tbl)

# モデル精度の検証
calibration_xgboost_tbl %>% modeltime_accuracy()

# 予測データの作成
test_forecast_xgboost_tbl <-
  calibration_xgboost_tbl %>%
    modeltime_forecast(new_data    = test_tbl,
                       actual_data = data_prepared_tbl,
                       keep_data   = TRUE)

# プロット作成
# --- まだ異常値を含んでいる
# --- GLMNET単独は少しマシか？
test_forecast_xgboost_tbl %>%
    group_by(source_key) %>%
    plot_modeltime_forecast(.facet_ncol = 3, .conf_interval_show = FALSE)


# 10 モデリング（SVM） ----------------------------------------------------------

# モデル定義
# --- SVM
model_spec_svm <-
  svm_rbf(mode = "regression",
          margin = 0.001) %>%
    set_engine("kernlab")

# ワークフロー設定＆学習
# --- レシピ1：カテゴリをダミー変数化
# --- 再帰的回帰
wflw_fit_svm_lag <-
  workflow() %>%
    add_model(model_spec_svm) %>%
    add_recipe(recipe_spec_lag) %>%
    fit(train_tbl) %>%
    recursive(id         = "source_key",
              transform  = transformer_function,
              train_tail = panel_tail(train_tbl, source_key, FORECAST_HORIZON))

# ワークフロー設定＆学習
# --- レシピ2：ラグ系列除外 + カレンダー追加 + カテゴリをダミー変数化
wflw_fit_svm_calendar <-
  workflow() %>%
    add_model(model_spec_svm) %>%
    add_recipe(recipe_spec_calendar) %>%
    fit(train_tbl)

# ワークフロー設定＆学習
# --- レシピ3：カレンダー追加 + カテゴリをダミー変数化
# --- 再帰的回帰
wflw_fit_svm_hybrid <-
  workflow() %>%
    add_model(model_spec_svm) %>%
    add_recipe(recipe_spec_hybrid) %>%
    fit(train_tbl) %>%
    recursive(id = "source_key",
              transform = transformer_function,
              train_tail = panel_tail(train_tbl, source_key, FORECAST_HORIZON))

# モデルテーブル登録＆検証準備
# --- 学習済モデル
calibration_svm_tbl <-
  modeltime_table(wflw_fit_svm_lag,
                  wflw_fit_svm_calendar,
                  wflw_fit_svm_hybrid) %>%
    modeltime_calibrate(test_tbl)

# モデル精度の検証
calibration_svm_tbl %>% modeltime_accuracy()

# 予測データの作成
test_forecast_svm_tbl <-
  calibration_svm_tbl %>%
    modeltime_forecast(new_data    = test_tbl,
                       actual_data = data_prepared_tbl,
                       keep_data   = TRUE)

# プロット作成
# --- まだ異常値を含んでいる
# --- KERNLAB単独は少しマシか？
test_forecast_svm_tbl %>%
  group_by(source_key) %>%
  plot_modeltime_forecast(.facet_ncol = 3, .conf_interval_show = FALSE)


# 11 最良モデルの比較 ----------------------------------------------------

# モデルテーブル登録＆検証準備
# --- 学習済モデル
calibration_tbl <-
  modeltime_table(wflw_fit_xgboost_calendar,
                  wflw_fit_svm_calendar,
                  wflw_fit_glmnet_lag) %>%
    modeltime_calibrate(test_tbl)

# モデル精度の検証
calibration_tbl %>% modeltime_accuracy()

# 予測データの作成
test_forecast_tbl <-
  calibration_tbl %>%
    modeltime_forecast(test_tbl, keep_data = TRUE)


# 12 系列ごとの予測精度 ----------------------------------------------------

# モデル精度の検証
# --- 系列ごと
accuracy_by_identifier_tbl <-
  test_forecast_tbl %>%
    select(source_key, .model_id, .model_desc, .index, .value,  value) %>%
    group_by(source_key, .model_id, .model_desc) %>%
    summarize_accuracy_metrics(truth      = value,
                               estimate   = .value,
                               metric_set = default_forecast_accuracy_metric_set()) %>%
    ungroup()

# 最良モデルの抽出
# --- RMSEで評価
best_rmse_by_indentifier_tbl <-
  accuracy_by_identifier_tbl %>%
    group_by(source_key) %>%
    slice_min(rmse, n = 1) %>%
    ungroup()

# データ確認
best_rmse_by_indentifier_tbl %>% print()


# 13 リフィットして予測 ----------------------------------------------------

# データ作成
# --- リフィット用の全期間データ
refit_data <-
  data_prepared_tbl %>%
    group_by(source_key) %>%
    mutate(value = ts_clean_vec(value, period = 12)) %>%
    ungroup()

# リフィット
refitted_tbl <-
  calibration_tbl %>%
    modeltime_refit(data = refit_data)

# アウトサンプルの予測
future_forecast_tbl <-
  refitted_tbl %>%
    modeltime_forecast(new_data    = data_future_tbl,
                       actual_data = data_prepared_tbl,
                       keep_data   = TRUE)

# プロット作成
future_forecast_tbl %>%
    group_by(source_key) %>%
    plot_modeltime_forecast(.facet_ncol = 3, .conf_interval_show = FALSE)


# 14 最良モデルのみで予測 ----------------------------------------------------

# データ作成
# --- Actualデータ
actual_tbl <- future_forecast_tbl %>% filter(.model_desc == "ACTUAL")

# データ作成
# --- Forecastデータ
future_forecast_best_tbl <-
  future_forecast_tbl %>%
    right_join(
        best_rmse_by_indentifier_tbl %>% select(source_key, .model_id, .model_desc),
        by = c(".model_id", ".model_desc", "source_key")
    )

# プロット作成
# --- 全てを最良モデルにすると、それなりに当てはまりがよい
actual_tbl %>%
  bind_rows(future_forecast_best_tbl) %>%
  left_join(data_description_tbl %>% select(source_key, fuel)) %>%
  group_by(fuel) %>%
  plot_modeltime_forecast(.facet_ncol = 3, .conf_interval_show = FALSE)
