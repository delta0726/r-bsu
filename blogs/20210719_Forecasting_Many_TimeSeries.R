# ******************************************************************************
# Title       : Business Science Blog
# Theme       : Forecasting Many Time Series (Using NO For-Loops)
# Update Date : 2021/7/19
# URL         : https://www.business-science.io/code-tools/2021/07/19/modeltime-panel-data.html
# ******************************************************************************


# ＜テーマ＞
# - Modeltimeは、パネルデータとグローバルモデルを使用して異なるアプローチを取るように設計
#   --- これらのアプローチを使用して、予測を行うことができる規模を劇的に増やすことが可能
#   --- クラスタリング手法といくつかのパネルモデルの作成により、それを超えることも可能


# ＜伝統的モデリングの問題点＞
# - ARIMAのような従来のモデリング手法は、一度に1つの時系列でのみ使用
#   --- 大量の系列を分析する場合はループ処理が必要となる（メモリエラーなどが発生しがち）


# ＜グローバルモデル＞
# - グローバルモデルは、すべての時系列を一度に予測する単一のモデル
#   --- 非常にスケーラブルであり、1〜10,000の時系列の問題を解決する
#   --- XGBoostモデルは、単一のモデルで1000個の時系列パネルすべての関係を判別することができる
# - 欠点は、反復アプローチよりも精度が低くなる可能性がある点が挙げられる
#   --- 機能エンジニアリングと時系列識別子によるローカライズされたモデルの選択が不可欠


# 0 準備
# 1 データ確認
# 2 データ分割
# 3 前処理
# 4 モデル構築
# 5 モデル精度の検証
# 6 最終結果

# 0 準備 -------------------------------------------------------------------------

library(tidymodels)
library(modeltime)
library(tidyverse)
library(timetk)


# データ準備
data <-
  walmart_sales_weekly %>%
    select(id, Date, Weekly_Sales) %>%
    set_names(c("ID", "date", "value"))


# 1 データ確認 ---------------------------------------------------------------------

# ＜ポイント＞
# - データセットは、特定の週に店舗と部門の組み合わせによって生成された収益の1001個の観測値で構成
# - IDは7つの系列で構成（パネルデータの形式）
# - 時系列グループは単一のグローバルモデルとして構成
# - プロットより系列のほとんどは毎年の季節性と長期的な傾向が確認される


# データ確認
data %>% print()
data %>% group_by(ID) %>% tally()

# プロット作成
data %>%
  group_by(ID) %>%
  plot_time_series(.date_var    = date,
                   .value       = value,
                   .facet_ncol  = 3,
                   .interactive = FALSE)


# 2 データ分割 ---------------------------------------------------------------------

# 時系列データ分割
# --- 日付で分割しているため、IDごとに正しく処理されている
splits <-
  data %>%
    time_series_split(assess = "3 months", cumulative = TRUE)

# データ確認
splits %>% print()
splits %>% training() %>% group_by(ID) %>% tally()
splits %>% testing() %>% group_by(ID) %>% tally()


# 3 前処理 ------------------------------------------------------------------------

# 前処理の設定
rec_obj <-
  recipe(value ~ ., training(splits)) %>%
    step_mutate(ID = droplevels(ID)) %>%
    step_timeseries_signature(date) %>%
    step_rm(date) %>%
    step_zv(all_predictors()) %>%
    step_dummy(all_nominal_predictors(), one_hot = TRUE)

# 処理確認
rec_obj %>% prep() %>% summary()
rec_obj %>% prep() %>% juice() %>% glimpse()


# 4 モデル構築 -----------------------------------------------------------------------

# モデル設定
# --- 学習済
wflw_xgb <-
  workflow() %>%
    add_model(boost_tree() %>% set_engine("xgboost")) %>%
    add_recipe(rec_obj) %>%
    fit(training(splits))

# テーブル登録
model_tbl <- modeltime_table(wflw_xgb)

# 確認
wflw_xgb %>% print()
model_tbl %>% print()


# 5 モデル精度の検証 -------------------------------------------------------------------

# 検証データの予測
# --- キャリブレーションはサンプル外の残差誤差を計算
# --- グループを判定するためIDを指定
calib_tbl <-
  model_tbl %>%
    modeltime_calibrate(new_data = testing(splits),
                        id       = "ID")

# 確認
# --- 学習器自体は1つしかできていない
calib_tbl %>% print()
calib_tbl$.model
calib_tbl$.calibration_data

# 予測データの作成
# --- グローバルモデル
calib_tbl %>%
  modeltime_accuracy(acc_by_id = FALSE) %>%
  table_modeltime_accuracy(.interactive = FALSE)

# 予測データの作成
# --- グローバルモデル
calib_tbl %>%
  modeltime_accuracy(acc_by_id = TRUE) %>%
  table_modeltime_accuracy(.interactive = FALSE)


# 6 最終結果 ---------------------------------------------------------------------

# データ作成
# --- actual_data引数でデータを指定すると、データが追加される
# --- 検証データはactualとpredictionの両方に存在する
plot_data <-
  calib_tbl %>%
    modeltime_forecast(new_data    = testing(splits),
                       actual_data = bind_rows(training(splits), testing(splits)),
                       conf_by_id  = TRUE)

# データ確認
# --- 全体のレコード数
plot_data %>%
  group_by(.key, ID) %>%
  tally()

# データ確認
# --- 検証データはactualとpredictionの両方に存在する
plot_data %>%
  filter(.index %in% lubridate::ymd("2012-10-26"))


# プロット作成
# --- 訓練データと検証データを結合して出力
calib_tbl %>%
    modeltime_forecast(new_data    = testing(splits),
                       actual_data = bind_rows(training(splits), testing(splits)),
                       conf_by_id  = TRUE) %>%
    group_by(ID) %>%
    plot_modeltime_forecast(.facet_ncol  = 3,
                            .interactive = FALSE)
