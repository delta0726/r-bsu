# ******************************************************************************
# Title       : BSU Learning Lab
# Theme       : ROBYN SETUP
# Chapter     : LAB 63: MODELTIME NESTED FORECASTING
# Module      : 01_analysis.R
# Update Date : 2021/9/11
# URL         : https://university.business-science.io/courses/enrolled/
# ******************************************************************************


# ＜概要＞
# - 大量の時系列データを1系列ごとにモデルを構築して学習するためのワークフロー
# - lab50の階層時系列は全系列のデータで1モデルを作成したが、今回は1系列ずつモデルを作成する
# - 並列処理の仕組みとtibbleを使ったネストを使った学習が秀逸


# ＜ライブラリ更新＞
# 最新版のモデルタイム（2021/09/11時点で今回の機能の実装はgithubバージョンのみ）
# remotes::install_github("business-science/modeltime", dependencies = TRUE)


# ＜目次＞
# 0 準備
# 1 データ確認
# 2 データ加工
# 3 レシピ作成
# 4 モデル構築
# 5 並列処理なしの学習
# 6 並列処理ありの学習
# 7 予測精度の確認
# 8 最良モデルの抽出
# 9 リフィット
# 10 期間が短い系列の学習
# 11 終了処理
# 12 学習期間を変えて予測


# 0 準備 ----------------------------------------------------------------------

# ライブラリ
library(modeltime)
library(tidymodels)
library(tidyverse)
library(timetk)
library(magrittr)
library(tictoc)


# データロード
sales_raw_tbl <- read_rds("lab_63_nested_forecasting/data/walmart_item_sales.rds")

# 並列処理の開始
cl <- parallel::detectCores() - 1
parallel_start(cl)


# 1 データ確認 -------------------------------------------------------------------

# データ確認
# --- 全てのidで同じレコード数（全IDての同じ日付が含まれる）
sales_raw_tbl %>% print()
sales_raw_tbl$item_id %>% table()

# プロット確認
# --- as.numeric(item_id)でファクターを番号に変換
# --- 12系列のみ抽出
sales_raw_tbl %>%
  filter(as.numeric(item_id) %in% 1:12) %>%
  group_by(item_id) %>%
  plot_time_series(date, value, .facet_ncol = 3, .smooth = FALSE)


# 2 データ加工 -------------------------------------------------------------------

# ＜ポイント＞
# - IDごとにネストしたデータフレームを作成する

# データネスト化
# --- 将来データの追加
# --- ネスト化してActualとFeatureの列を分離
# --- データ分割
nested_data_tbl <-
  sales_raw_tbl %>%
    group_by(item_id) %>%
    extend_timeseries(.id_var = item_id,
                      .date_var = date,
                      .length_future = 90) %>%
    nest_timeseries(.id_var = item_id,
                    .length_future = 90) %>%
    split_nested_timeseries(.length_test = 90)

# データ確認
nested_data_tbl %>% print()

# 参考：将来データの確認
# --- valueの部分はNAとなっている
sales_raw_tbl %>%
  group_by(item_id) %>%
  extend_timeseries(.id_var = item_id,
                    .date_var = date,
                    .length_future = 90) %>%
  filter(item_id == "FOODS_3_090") %>%
  tail(100) %>%
  print(n = nrow(.))


# 3 レシピ作成 ---------------------------------------------------------------------

# ＜ポイント＞
# - XGBoostを意識したレシピを作成
# - データセットはネストから取り出した個別IDのデータに基づいて定義

# レシピ作成
# --- id1のデータを抽出してレシピを作成する
# --- 日付要素を分解＆ダミー変数化して特徴量とする
rec_xgb <-
  recipe(value ~ ., extract_nested_train_split(nested_data_tbl)) %>%
    step_timeseries_signature(date) %>%
    step_rm(date) %>%
    step_zv(all_predictors()) %>%
    step_dummy(all_nominal_predictors(), one_hot = TRUE)

# データ確認
rec_xgb %>%
  prep() %>%
  bake(extract_nested_train_split(nested_data_tbl)) %>%
  glimpse()


# 4 モデル構築 -------------------------------------------------------------------

# ＜ポイント＞
# - XGBoostで2モデルを作成してワーフクローを構築する
# - thiefの階層モデルをベンチマークとして使用（レシピは未加工のものを使用）

# モデル1
# --- 学習率：0.35
wflw_xgb_1 <-
  workflow() %>%
    add_model(boost_tree("regression", learn_rate = 0.35) %>% set_engine("xgboost")) %>%
    add_recipe(rec_xgb)

# モデル2
# --- 学習率：0.50
wflw_xgb_2 <-
  workflow() %>%
    add_model(boost_tree("regression", learn_rate = 0.50) %>% set_engine("xgboost")) %>%
    add_recipe(rec_xgb)

# モデル3（ベンチマーク）
# --- 階層時系列モデル
wflw_thief <-
  workflow() %>%
    add_model(temporal_hierarchy() %>% set_engine("thief")) %>%
    add_recipe(recipe(value ~ ., extract_nested_train_split(nested_data_tbl)))

# モデルリストの作成
model_list <-
  list(wflw_xgb_1,
       wflw_xgb_2,
       wflw_thief)


# 5 並列処理なしの学習 ---------------------------------------------------------

# ＜ポイント＞
# - テストとして1系列のみで学習
# - キャリブレーションでは｢.splits｣の訓練データのみを用いて学習する
#   --- 後のプロセスで検証データで予測精度を確認

# 学習設定
control <-
  control_nested_fit(verbose   = TRUE,
                     allow_par = FALSE)

# 学習
# --- テストとして1系列のみで学習
# --- モデルリストでモデル登録（...に個別モデルを入力して指定することも可能）
try_sample_tbl <-
  nested_data_tbl %>%
    slice(1) %>%
    modeltime_nested_fit(model_list = model_list,
                         control    = control)

# 確認
# --- .modeltime_tablesに学習結果が格納されている
try_sample_tbl %>% print()

# エラーレポート
# --- エラーなし
try_sample_tbl %>% extract_nested_error_report()


# 6 並列処理ありの学習 ---------------------------------------------------------

# ＜ポイント＞
# - ネストした全系列を並列処理を用いて学習
# - モデルタイムのワークフローにおける｢モデルテーブルの登録｣と｢キャリブレーション｣が行われる
# - キャリブレーションでは｢.splits｣の訓練データのみを用いて学習する
#   --- 後のプロセスで検証データで予測精度を確認

# 学習設定
# --- キャリブレーション学習の設定
control <-
  control_nested_fit(verbose   = TRUE,
                     allow_par = TRUE)

# 学習
# --- 約55秒
tic()
nested_modeltime_tbl <-
  nested_data_tbl %>%
    modeltime_nested_fit(model_list = model_list,
                         control    = control)
toc()

# 確認
# --- ｢.modeltime_tables｣に学習結果が格納されている
nested_modeltime_tbl %>% print()

# テーブル構造の確認
# --- ネストの中には3モデルの結果が格納されている
# --- ｢.calibration_data｣に検証データでの予測結果が格納されている
nested_modeltime_tbl$.modeltime_tables[[1]]
nested_modeltime_tbl$.modeltime_tables[[1]]$.calibration_data[[1]]

# エラーレポート
# --- ｢HOUSEHOLD_2_101｣でエラーが出ている（日付が1つだけ短い系列）
nested_modeltime_tbl %>% extract_nested_error_report()

# エラー原因
# --- データ分割が適切に行われていなかった
# --- 元データが90日分しかないため全て検証データで使用された（訓練データが確保できなかった）
nested_modeltime_tbl %>%
  filter(item_id == "HOUSEHOLD_2_101") %>%
  extract_nested_train_split()


# 7 予測精度の確認 ---------------------------------------------------------------

# ＜ポイント＞
# - 学習により作成されたキャリブレーションデータを用いて予測精度を確認する
# - 検証データにおける予測値と信頼区間の算出を行う

# 予測精度の確認
# --- ネストごとに登録された3モデルに対して検証データの予測結果を抽出
nested_modeltime_tbl %>%
  extract_nested_test_accuracy() %>%
  table_modeltime_accuracy()

# プロット作成
# --- 検証データにおける予測値を取得(.value)
# --- 予測値の信頼区間も取得(.conf_lo/.conf_hi)
# --- 学習データの実績値と検証データの予測値をプロット
nested_modeltime_tbl %>%
  extract_nested_test_forecast() %>%
  filter(item_id == "FOODS_3_090") %>%
  group_by(item_id) %>%
  plot_modeltime_forecast(.facet_ncol = 3)

# 参考：データ構造の確認
nested_modeltime_tbl %>%
  extract_nested_test_forecast() %>%
  filter(item_id == "FOODS_3_090") %>%
  group_by(item_id, .model_desc, .key, .model_id) %>%
  tally()


# 8 最良モデルの抽出 ----------------------------------------------------------------

# ＜ポイント＞
# - 複数モデルからRMSEを用いて最良モデルを選定する
#   --- 今回はアルゴリズムごとにモデルを作成したが、レシピごとに作成することも可能

# 除外系列の指定
# --- 期間が短い系列
ids_small_timeseries <- "HOUSEHOLD_2_101"

# データ抽出
# --- 除外系列以外
nested_modeltime_subset_tbl <-
  nested_modeltime_tbl %>%
    filter(!item_id %in% ids_small_timeseries)

# 最良モデルの抽出
# --- 系列ごとにRMSEでモデルを判定
# --- ｢.modeltime_tables｣の行数が3から1に更新される
nested_best_tbl <-
  nested_modeltime_subset_tbl %>%
    modeltime_nested_select_best(metric = "rmse")

# プロット作成
# --- 12系列のみ
# --- 学習データの実績値と検証データの予測値をプロット
nested_best_tbl %>%
  extract_nested_test_forecast() %>%
  filter(as.numeric(item_id) %in% 1:12) %>%
  group_by(item_id) %>%
  plot_modeltime_forecast(.facet_ncol = 3)


# 9 リフィット ----------------------------------------------------------------------

# ＜ポイント＞
# - リフィットでは｢.actual_data｣を使って学習を行う
# - ｢.future_data｣は実績値がないという想定なので予測精度の検証は行えない
#   --- バックテストの場合は手持ちのデータと比較することが可能

# 学習設定
# --- リフィット用の設定
control <-
  control_refit(verbose   = TRUE,
                allow_par = TRUE)

# 再学習
# --- 最良モデルを使用
nested_best_refit_tbl <-
  nested_best_tbl %>%
    modeltime_nested_refit(control = control)

# 確認
nested_best_refit_tbl %>% print()

# エラーレポート
# --- エラーなし
nested_best_refit_tbl %>% extract_nested_error_report()

# プロット作成
# --- 将来データの予測（将来日付の作成の際にNAとしたデータを予測）
nested_best_refit_tbl %>%
  extract_nested_future_forecast() %>%
  filter(as.numeric(item_id) %in% 1:12) %>%
  group_by(item_id) %>%
  plot_modeltime_forecast(.facet_ncol = 3)


# 10 期間が短い系列の学習 -----------------------------------------------------------------

# 除外系列の指定
# --- 期間が短い系列
ids_small_timeseries <- "HOUSEHOLD_2_101"

# ネストデータの再作成
# --- 予測期間を30日に変更(他の系列は90日)
nested_data_small_ts_tbl <-
  sales_raw_tbl %>%
    filter(item_id %in% ids_small_timeseries) %>%
    group_by(item_id) %>%
    extend_timeseries(.id_var = item_id, .date_var = date, .length_future = 90) %>%
    nest_timeseries(.id_var = item_id, .length_future = 90) %>%
    split_nested_timeseries(.length_test = 30)

# 学習設定
control <-
  control_nested_fit(verbose   = TRUE,
                     allow_par = FALSE)

# 学習プロセス
# --- 事前学習で最良モデルを判定
# --- 最良モデルで再学習
nested_best_refit_small_ts_tbl <-
  nested_data_small_ts_tbl %>%
    modeltime_nested_fit(model_list = model_list, control = control) %>%
    modeltime_nested_select_best() %>%
    modeltime_nested_refit()

# プロット作成
nested_best_refit_small_ts_tbl %>%
  extract_nested_future_forecast() %>%
  group_by(item_id) %>%
  plot_modeltime_forecast(.facet_ncol = 3)


# 11 終了処理 --------------------------------------------------------------------

# 並列処理の終了
parallel_stop()


# 12 学習期間を変えて予測 -----------------------------------------------------

# ＜ポイント＞
# - 最良モデルを用いて予測期間を変えて予測することも可能

# 結果の結合
# --- 期間が長い系列
# --- 期間が短い系列（1系列のみ）
nested_best_refit_all_tbl <-
  nested_best_refit_tbl %>%
    bind_rows(nested_best_refit_small_ts_tbl)

# 並列処理の開始
parallel_start(6)

# 予測設定
control <-
  control_nested_forecast(verbose   = TRUE,
                          allow_par = TRUE)

# 予測
# --- ホライズンを90から365に変更
new_forecast_tbl <-
  nested_best_refit_all_tbl %>%
    modeltime_nested_forecast(h = 365,
                              conf_interval = 0.99,
                              control = control)

# 確認
new_forecast_tbl %>%
  filter(item_id == "FOODS_3_090") %>%
  group_by(.model_desc, .key) %>%
  tally()

# プロット作成
# ---
new_forecast_tbl %>%
  filter(as.numeric(item_id) %in% 1:12) %>%
  group_by(item_id) %>%
  plot_modeltime_forecast(.facet_ncol = 3)

# 並列処理の終了
parallel_stop()
