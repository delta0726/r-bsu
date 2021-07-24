# ******************************************************************************
# Title       : BSU Learning Lab
# Theme       : Tidymodels Ecosystem
# Chapter     : LAB 51: Deep Learning with Torch & Tabnet
# Module      : 02_modeltime_forecasting_tabnet.R
# Update Date : 2021/7/24
# URL         : https://university.business-science.io/courses/enrolled/
# ******************************************************************************


# ＜テーマ＞
# - {tabnet}を使って{torch}のディープラーニングによる階層的時系列予測を行う


# ＜目次＞
# 0 準備
# 1 データ加工
# 2 データ確認
# 3 モデルデータの準備
# 4 予測用データの作成
# 5 データ分割
# 6 特徴量エンジニアリング
# 7 モデル構築＆実行
# 8 モデル精度の検証
# 9 プロット作成
# 10 最良モデルの選定
# 11 アンサンブル
# 12 変数重要度


# 0 準備 --------------------------------------------------------------------------

# Machine Learning
library(tabnet)
library(xgboost)
library(torch)

# Tidymodels
library(modeltime)
library(modeltime.ensemble)
library(tidymodels)
library(magrittr)

# Interpretation 
library(vip)

# Core
library(skimr)
library(timetk)
library(tidyverse)
library(tictoc)


# データロード
calendar_tbl <- read_csv("lab_51_torch_tabnet/m5-forecasting-accuracy/calendar.csv")
sales_sample_tbl <- read_rds("lab_51_torch_tabnet/m5-forecasting-accuracy/sales_sample_tbl.rds")

# データ確認
calendar_tbl %>% glimpse()
sales_sample_tbl %>% glimpse()



# 1 データ加工 ---------------------------------------------------------------------

# 階層キーの取得
hierarchy_tbl <- sales_sample_tbl %>% select(contains("id"))
hierarchy_tbl

# データ加工
# --- ロング型に変換
# --- ｢階層キー｣｢日付｣｢値｣のデータセットを作成
sales_sample_long_tbl <-
  sales_sample_tbl %>%
    pivot_longer(cols      = starts_with("d_"),
                 names_to  = "day",
                 values_to = "value") %>%
    left_join(calendar_tbl, by = c("day" = "d")) %>%
    select(contains("_id"), date, value)


# 2 データ確認 ---------------------------------------------------------------------

# データサマリー
sales_sample_long_tbl %>% skim()

# IDの抽出
# --- item_id
set.seed(123)
item_id_sample <-
  sales_sample_long_tbl$item_id %>%
    unique() %>%
    sample(size = 6)

# 時系列プロット作成
# --- IDを抽出したパネルデータのプロット
sales_sample_long_tbl %>%
  filter(item_id %in% item_id_sample) %>%
  group_by(item_id) %>%
  plot_time_series(date, value,
                   .smooth        = TRUE,
                   .smooth_period = 28,
                   .facet_ncol    = 2)


# 3 モデルデータの準備 -------------------------------------------------------------

# 階層を2つに集約
# --- ID列名とID列アイテム
# --- データを合計して扱う
model_data_tbl <-
  sales_sample_long_tbl %>%
    add_column(all_stores_id = "all_stores", .before = 1) %>%
    pivot_longer(cols      = ends_with("_id"),
                 names_to  = "category",
                 values_to = "identifier") %>%
    group_by(category, identifier, date) %>%
    summarise(value = sum(value, na.rm = TRUE)) %>%
    ungroup()

# データ確認
model_data_tbl %>%
  group_by(category, identifier) %>%
  tally() %>%
  print(n = nrow(.))

# モデル用データの作成
# --- 将来予測のレコードを作成
# --- ラグ系列の追加
# --- 移動平均の追加（value_lag28のみ）
full_data_tbl <-
  model_data_tbl %>%
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
full_data_tbl %>% glimpse()
full_data_tbl %>% skim()

# データロード
# --- 提供データ
#full_data_tbl <- read_rds("lab_51_torch_tabnet/m5-forecasting-accuracy/full_data_tbl.rds")


# 4 予測用データの作成 --------------------------------------------------------

# 訓練用データ
# --- 値が入っている
data_prepared_tbl <-
  full_data_tbl %>%
    filter(!is.na(value)) %>%
    filter(!is.na(value_lag28))

# 予測用データ
# --- 値がない
future_data_tbl <-
  full_data_tbl %>%
    filter(is.na(value))

# データ確認
# --- 訓練用データ
# --- 予測用データ
data_prepared_tbl %>% filter(identifier == "all_stores")
future_data_tbl %>% filter(identifier == "all_stores")

# データサマリー
data_prepared_tbl %>% skim()
future_data_tbl %>% skim()


# 5 データ分割 --------------------------------------------------------------

# 時系列データ分割
splits <-
  data_prepared_tbl %>%
    time_series_split(date, assess = 28, cumulative = TRUE)

# 分割イメージ
splits %>%
  tk_time_series_cv_plan() %>%
  plot_time_series_cv_plan(date, value)


# 6 特徴量エンジニアリング -----------------------------------------------------

# レシピ作成
# --- ｢モデル用データ作成｣で一定の前処理を施している
# --- step_timeseries_signature()で時系列情報を持たせている
recipe_spec <-
  recipe(value ~ ., data = training(splits)) %>%
    update_role(row_id, date, new_role = "id") %>%
    step_timeseries_signature(date) %>%
    step_rm(matches("(.xts$)|(.iso$)|(hour)|(minute)|(second)|(am.pm)")) %>%
    step_dummy(all_nominal(), one_hot = TRUE)

# サマリー
recipe_spec %>% summary()
recipe_spec %>% prep() %>% summary()

# 加工データの確認
recipe_spec %>% prep() %>% juice() %>% glimpse()


# 7 モデル構築＆実行 ------------------------------------------------------

# * TabNet ------------------------------------

# モデル定義
model_tabnet <-
  tabnet(mode               = "regression",
         batch_size         = 1024*2,
         virtual_batch_size = 1024*2,
         epochs             = 2) %>%
    set_engine("torch", verbose = TRUE)

# ワークフロー設定＆学習
tic()
wflw_tabnet_fit <- workflow() %>%
    add_model(model_tabnet) %>%
    add_recipe(recipe_spec) %>%
    fit(training(splits))
toc()


# * モデル保存/ロード ------------------------------------------------

# Serialization
# wflw_tabnet_fit %>% write_rds( "models_02_forecast/wflw_tabnet_fit.rds")
# wflw_tabnet_fit$fit$fit$fit$fit$network %>% torch_save( "models_02_forecast/torch_network")

# Loading
# wflw_tabnet_fit <- read_rds("models_02_forecast/wflw_tabnet_fit.rds")
# torch_network   <- torch::torch_load("models_02_forecast/torch_network")

# オブジェクト上書き
# wflw_tabnet_fit$fit$fit$fit$fit$network <- torch_network
# wflw_tabnet_fit


# * XGBoost ------------------------------------

# モデル定義
model_xgboost <-
  boost_tree(mode = "regression") %>%
              set_engine("xgboost")

# ワークフロー設定＆学習
tic()
wflw_xgboost_defaults <-
  workflow() %>%
    add_model(model_xgboost)%>%
    add_recipe(recipe_spec) %>%
    fit(training(splits))
toc()


# 8 モデル精度の検証 -------------------------------------------------------------------

# 検証データの作成
# --- モデルテーブルの作成
# --- キャリブレーション
calibration_tbl <-
  modeltime_table(wflw_tabnet_fit,
                  wflw_xgboost_defaults) %>%
    modeltime_calibrate(testing(splits))

# モデル精度の検証
# --- 検証用データを使用
calibration_tbl %>% modeltime_accuracy()


# 9 プロット作成 ----------------------------------------------------------------------

# 予測値の算出
test_forecast_tbl <-
  calibration_tbl %>%
    modeltime_forecast(new_data    = testing(splits),
                       actual_data = data_prepared_tbl,
                       keep_data   = TRUE)

# IDの作成
# - All Stores Aggregated（トップレベル）
filter_identfiers_all <- "all_stores"

# IDの取得
# - State-Level Forecasts（州レベル）
filter_identfiers_state <-
  full_data_tbl %>%
    filter(category == "state_id") %>%
    distinct(identifier) %>%
    pull()

# - Item-Level Forecasts (Sample of 6)
filter_identfiers_items <- item_id_sample 

# プロット作成
test_forecast_tbl %>%
  filter(identifier %in% filter_identfiers_items) %>%
  group_by(identifier) %>%
  filter_by_time(.start_date = last(date) %-time% "3 month",
                 .end_date   = "end") %>%
  plot_modeltime_forecast(.facet_ncol         = 2,
                          .conf_interval_show = FALSE,
                          .interactive        = TRUE)


# 10 最良モデルの選定 ---------------------------------------------------------------

# 予測精度の算出
# --- IDごと
accuracy_by_identifier_tbl <-
  test_forecast_tbl %>%
    select(category, identifier, .model_desc, .index, .value) %>%
    pivot_wider(names_from   = .model_desc,
                values_from  = .value) %>%
    filter(!is.na(TORCH)) %>%
    pivot_longer(cols = TORCH:XGBOOST) %>%
    group_by(category, identifier, name) %>%
    summarize_accuracy_metrics(truth      = ACTUAL,
                               estimate   = value,
                               metric_set = default_forecast_accuracy_metric_set())

# 最良モデルの抽出
# --- rmseに基づく
best_rmse_by_indentifier_tbl <-
  accuracy_by_identifier_tbl %>%
    group_by(identifier) %>%
    slice_min(rmse, n = 1) %>%
    ungroup()


# カテゴリごとの予測精度
best_rmse_by_indentifier_tbl %>%
  group_by(category) %>%
  summarise(median_rmse = median(rmse))

best_rmse_by_indentifier_tbl %>% count(name, sort = TRUE)


# 11 アンサンブル ----------------------------------------------------------------

# * Combine Models and Forecast ----

ensemble_tbl <-
  calibration_tbl %>%
    ensemble_weighted(loadings = c(0.3, 0.7)) %>%
    modeltime_table() %>%
    modeltime_calibrate(testing(splits)) 

ensemble_tbl %>% modeltime_accuracy()

test_forecast_ensemble_tbl <- ensemble_tbl %>%
    modeltime_forecast(
        new_data    = testing(splits),
        actual_data = data_prepared_tbl,
        keep_data   = TRUE 
    )

# * Visualize ----

test_forecast_ensemble_tbl %>%
    
    # FILTER IDENTIFIERS
    filter(identifier %in% filter_identfiers_items) %>%
    
    group_by(identifier) %>%
    
    # Focus on end of series
    filter_by_time(
        .start_date = last(date) %-time% "3 month",
        .end_date   = "end"
    ) %>%
    
    plot_modeltime_forecast(
        .facet_ncol         = 2, 
        .conf_interval_show = FALSE,
        .interactive        = TRUE
    )

# 12 変数重要度 ------------------------------------------------------------------

# 変数重要度
# * TabNet ----
# * XGBoost ----
wflw_tabnet_fit %>% pull_workflow_fit() %>% vip()
wflw_xgboost_defaults %>% pull_workflow_fit() %>% vip()
