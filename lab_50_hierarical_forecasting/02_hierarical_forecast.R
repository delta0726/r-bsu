# ******************************************************************************
# Title       : BSU Learning Lab
# Chapter     : LAB 50: LIGHTGBM
# Theme       : HIERARCHICAL FORECASTING WITH LIGHTGBM & FRIENDS
# Module      : 02_modeltime_forecast.R
# Update Date : 2021/8/18
# URL         : https://university.business-science.io/courses/enrolled/
# ******************************************************************************


# ＜概要＞
# - 機械学習による時系列階層予測モデリングのフローを確認する
#   ---- 全体で学習してIDごとにモデル評価する


# ＜インストール＞
# - TREESNIP: remotes::install_github("curso-r/treesnip")
# - CATBOOST: devtools::install_github('catboost/catboost', subdir = 'catboost/R-package')


# ＜目次＞
# 0 準備
# 1 データ加工
# 2 訓練用データの作成
# 3 時系列データ分割
# 4 特徴量エンジニアリング
# 5 個別モデル定義＆学習
# 6 モデルテーブル登録
# 7 モデル精度の検証
# 8 IDごとのモデル評価3
# 9 アンサンブルモデルの作成
# 10 アンサンブルモデルによる将来データの予測
# 11 変数重要度の算出
# 12 変数重要度の算出


# 0 準備 ----------------------------------------------------------------------

# ライブラリ
library(lightgbm)
library(catboost)
library(xgboost)
library(treesnip)
library(modeltime)
library(modeltime.ensemble)
library(tidymodels)
library(skimr)
library(timetk)
library(tidyverse)


# ローケール設定変更
Sys.setlocale("LC_TIME", "English")
Sys.getlocale("LC_TIME")

# データロード
# --- カレンダー
calendar_tbl <- read_csv("lab_50_hierarical_forecasting/m5-forecasting-accuracy/calendar.csv")
calendar_tbl

# データロード
# --- セールスデータ
sales_sample_tbl <- read_rds("lab_50_hierarical_forecasting/m5-forecasting-accuracy/sales_sample_tbl.rds")
sales_sample_tbl

# データ抽出
# --- 階層キー
hierarchy_tbl <- sales_sample_tbl %>% select(contains("id"))
hierarchy_tbl


# 1 データ加工 ------------------------------------------------------------------

# データ加工
# --- ロングフォーマットに変換
# --- カレンダーを結合
sales_sample_long_tbl <-
  sales_sample_tbl %>%
    pivot_longer(cols      = starts_with("d_"),
                 names_to  = "day",
                 values_to = "value") %>%
    left_join(calendar_tbl, by = c("day" = "d")) %>%
    select(contains("_id"), date, value)

# データ確認
sales_sample_long_tbl %>% print()
sales_sample_long_tbl %>% skim()

# IDをランダム抽出
# --- プロット作成用
set.seed(123)
item_id_sample <- sales_sample_long_tbl$item_id %>% unique() %>% sample(size = 6)

# プロット作成
sales_sample_long_tbl %>%
  filter(item_id %in% item_id_sample) %>%
  group_by(item_id) %>%
  plot_time_series(.date_var = date, .value = value, .smooth = TRUE,
                   .smooth_period = 28, .facet_ncol = 2)


# 2 訓練用データの作成 --------------------------------------------------------------

# モデル用データの作成
full_data_tbl <-
  sales_sample_long_tbl %>%
    add_column(all_stores_id = "all_stores", .before = 1) %>%
    pivot_longer(cols      = ends_with("_id"),
                 names_to  = "category",
                 values_to = "identifier") %>%
    group_by(category, identifier, date) %>%
    summarise(value = sum(value, na.rm = TRUE)) %>%
    ungroup() %>%
    group_by(category, identifier) %>%
    pad_by_time(date, .by = "day", .pad_value = 0) %>%
    future_frame(date, .length_out = 28, .bind_data = TRUE) %>%
    tk_augment_lags(value, .lags = 28) %>%
    tk_augment_slidify(value_lag28,
                       .f       = ~ mean(., na.rm = TRUE),
                       .period  = c(7, 14, 28, 28*2),
                       .align   = "center",
                       .partial = TRUE) %>%
    ungroup() %>%
    rowid_to_column(var = "row_id")

# データ確認
full_data_tbl %>% print()
full_data_tbl %>% glimpse()
full_data_tbl %>% skim()


# 学習用データの作成
# --- 訓練データ/検証用データとして使用
data_prepared_tbl <-
  full_data_tbl %>%
    filter(!is.na(value)) %>%
    filter(!is.na(value_lag28))

# データ確認
data_prepared_tbl %>% print()
data_prepared_tbl %>% glimpse()
data_prepared_tbl %>% skim()


# 将来データの作成
# --- アウトサンプルの予測に使用
future_data_tbl <-
  full_data_tbl %>%
    filter(is.na(value))

# データ確認
future_data_tbl %>% print()
future_data_tbl %>% glimpse()
future_data_tbl %>% skim()


# 3 時系列データ分割 -------------------------------------------------------------

# ＜ポイント＞
# - 時間で分割するのでIDごとのデータの入り方によっては、訓練データと検証データの数は均質ではない


# データ分割
splits <-
  data_prepared_tbl %>%
    time_series_split(date, assess = 28, cumulative = TRUE)

# プロット作成
splits %>%
  tk_time_series_cv_plan() %>%
  plot_time_series_cv_plan(date, value)


# 4 特徴量エンジニアリング --------------------------------------------------------

# レシピ定義
recipe_spec <-
  recipe(value ~ ., data = training(splits)) %>%
    update_role(row_id, date, new_role = "id") %>%
    step_timeseries_signature(date) %>%
    step_rm(matches("(.xts$)|(.iso$)|(hour)|(minute)|(second)|(am.pm)")) %>%
    step_dummy(all_nominal(), one_hot = TRUE)

# データ確認
recipe_spec %>% prep() %>% juice() %>% glimpse()

# レシピサマリー
recipe_spec %>% summary()
recipe_spec %>% prep() %>% summary()


# 5 個別モデル定義＆学習 ----------------------------------------------------------

# * LIGHTGBM ----------------------------------------

# モデル定義
model_lightgbm_defaults <-
  boost_tree(mode = "regression") %>%
    set_engine("lightgbm")

model_lightgbm_tweedie <-
  boost_tree(mode = "regression") %>%
    set_engine("lightgbm", objective = "tweedie")

# ワークフロー設定＆学習
wflw_lightgbm_defaults <-
  workflow() %>%
    add_model(model_lightgbm_defaults) %>%
    add_recipe(recipe_spec) %>%
    fit(training(splits))

wflw_lightgbm_tweedie <-
  workflow() %>%
    add_model(model_lightgbm_tweedie) %>%
    add_recipe(recipe_spec) %>%
    fit(training(splits))


# * XGBOOST ----------------------------------------

# モデル定義
model_xgboost_defaults <-
  boost_tree(mode = "regression") %>%
    set_engine("xgboost")

model_xgboost_tweedie <-
  boost_tree(mode = "regression") %>%
    set_engine("xgboost", objective = "reg:tweedie")

# ワークフロー設定＆学習
wflw_xgboost_defaults <-
  workflow() %>%
    add_model(model_xgboost_defaults) %>%
    add_recipe(recipe_spec) %>%
    fit(training(splits))

wflw_xgboost_tweedie <-
  workflow() %>%
    add_model(model_xgboost_tweedie) %>%
    add_recipe(recipe_spec) %>%
    fit(training(splits))


# * CATBOOST ----------------------------------------

# モデル定義
model_catboost_defaults <-
  boost_tree(mode = "regression") %>%
    set_engine("catboost")

model_catboost_tweedie <-
  boost_tree(mode = "regression") %>%
    set_engine("catboost", loss_function = "Tweedie:variance_power=1.5")

# ワークフロー設定＆学習
wflw_catboost_defaults <-
  workflow() %>%
    add_model(model_catboost_defaults) %>%
    add_recipe(recipe_spec) %>%
    fit(training(splits))

wflw_catboost_tweedie <-
  workflow() %>%
    add_model(model_catboost_tweedie) %>%
    add_recipe(recipe_spec) %>%
    fit(training(splits))


# 6 モデルテーブル登録 ---------------------------------------------------------------

# テーブル登録
# --- 学習済モデル
model_tbl <-
  modeltime_table(wflw_xgboost_defaults,
                  wflw_xgboost_tweedie)


# 7 モデル精度の検証 -----------------------------------------------------------------

# 検証用データの作成
calibration_tbl <-
  model_tbl%>%
    modeltime_calibrate(testing(splits)) %>%
    mutate(.model_desc = ifelse(.model_id > 3, str_c(.model_desc, " - Tweedie"), .model_desc))

# メトリックの出力
calibration_tbl %>% modeltime_accuracy()

# 予測データの作成
# --- 検証データ
test_forecast_tbl <-
  calibration_tbl %>%
    modeltime_forecast(new_data    = testing(splits),
                       actual_data = data_prepared_tbl,
                       keep_data   = TRUE)

# 抽出用のキー設定
# - All Stores Aggregated
# - Item-Level Forecasts (Sample of 6)
filter_identfiers_all <- "all_stores"
filter_identfiers_items <- item_id_sample

# プロット作成
test_forecast_tbl %>%
  filter(identifier %in% filter_identfiers_items) %>%
  group_by(identifier) %>%
  filter_by_time(.start_date = last(date) %-time% "3 month",
                 .end_date = "end") %>%
    plot_modeltime_forecast(.facet_ncol         = 2,
                            .conf_interval_show = FALSE,
                            .interactive        = TRUE)


# 8 IDごとのモデル評価 -----------------------------------------------------------

# 予測精度の評価
# --- IDごと
accuracy_by_identifier_tbl <-
  test_forecast_tbl %>%
    select(category, identifier, .model_desc, .index, .value) %>%
    pivot_wider(names_from   = .model_desc,
                values_from  = .value) %>%
    filter(!is.na(LIGHTGBM)) %>%
    pivot_longer(cols = LIGHTGBM:`CATBOOST - Tweedie`) %>%
    group_by(category, identifier, name) %>%
    summarize_accuracy_metrics(truth      = ACTUAL,
                               estimate   = value,
                               metric_set = default_forecast_accuracy_metric_set())

# 最良モデルの抽出
# --- IDごと
best_rmse_by_indentifier_tbl <-
  accuracy_by_identifier_tbl %>%
    group_by(identifier) %>%
    slice_min(rmse, n = 1) %>%
    ungroup()


best_rmse_by_indentifier_tbl %>%
    group_by(category) %>%
    summarise(median_rmse = median(rmse))

best_rmse_by_indentifier_tbl %>% count(name, sort = TRUE)


# 9 アンサンブルモデルの作成 -------------------------------------------------------

# アンサンブルモデルの作成
# --- モデルテーブル形式
ensemble_tbl <-
  calibration_tbl %>%
    filter(.model_id %in% c(2, 5)) %>%
    ensemble_weighted(loadings = c(2, 3)) %>%
    modeltime_table()

# 予測
test_forecast_ensemble_tbl <-
  ensemble_tbl %>%
    modeltime_calibrate(testing(splits)) %>%
    modeltime_forecast(new_data    = testing(splits),
                       actual_data = data_prepared_tbl,
                       keep_data   = TRUE)


# 10 アンサンブルモデルによる将来データの予測 -----------------------------------------

ensemble_refit_tbl <- ensemble_tbl %>%
    modeltime_refit(data_prepared_tbl)

future_forecast_ensemble_tbl <- ensemble_refit_tbl %>%
    modeltime_forecast(
        new_data    = future_data_tbl,
        actual_data = data_prepared_tbl,
        keep_data   = TRUE
    )

# プロット作成
future_forecast_ensemble_tbl %>%
  filter(identifier %in% filter_identfiers_all) %>%
  group_by(identifier) %>%
  filter_by_time(.start_date = last(date) %-time% "6 month",
                 .end_date = "end") %>%
  plot_modeltime_forecast(.facet_ncol         = 2,
                          .conf_interval_show = TRUE,
                          .interactive        = TRUE)


# 11 変数重要度の算出 --------------------------------------------------------------------

# 変数重要度
wflw_xgboost_defaults %>%
    pull_workflow_fit() %>%
    pluck("fit") %>%
    xgboost::xgb.importance(model = .) %>%
    xgboost::xgb.plot.importance()
