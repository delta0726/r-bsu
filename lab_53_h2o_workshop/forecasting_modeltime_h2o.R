# ******************************************************************************
# Title       : BSU Learning Lab
# Theme       : Production Data Science Pipelines with Targets
# Chapter     : Lab 53: MODELTIME H2O WORKSHOP
# Module      : forecasting_modeltime_h2o.R
# Update Date : 2021/7/26
# URL         : https://university.business-science.io/courses/enrolled/
# ******************************************************************************


# BUSINESS OBJECTIVE ----
# - Forecast intermittent demand
# - Predict next 52-WEEKS


# ＜目次＞
# 0 準備
# 1 データ準備
# 2 季節性の確認
# 3 データ分割
# 4 特徴量エンジニアリング
# 5 モデル構築
# 6 学習
# 7 モデルテーブル
# 8 モデル精度の検証
# 9 予測データの作成
# 10 リフィット


# 0 準備 ------------------------------------------------------------------------

# ライブラリ
library(tidymodels)
library(magrittr)
library(modeltime.h2o)
library(tidyverse)
library(timetk)

# ローケール変更
Sys.setlocale("LC_TIME", "English")
Sys.getlocale("LC_TIME")


# データロード
walmart_sales_weekly


# 1 データ準備 -------------------------------------------------------------------

# データ加工
# --- idで区切ったWeekly_Sales(Y)のパネルデータ
# --- 週次データ
data_tbl <-
  walmart_sales_weekly %>%
    select(id, Date, Weekly_Sales)

# データ確認
# --- 全体で143W
data_tbl %>% group_by(id) %>% tally()

# プロット作成
data_tbl %>%
  group_by(id) %>%
  plot_time_series(.date_var    = Date,
                   .value       = Weekly_Sales,
                   .facet_ncol  = 2,
                   .smooth      = TRUE,
                   .smooth_period = "2 quarters",
                   .interactive = TRUE)


# 2 季節性の確認 -----------------------------------------------------------------

# ユニークID取得
ids <- data_tbl$id %>% unique()

# プロット作成
# --- 季節性プロット
# --- 週次や月次で季節性が確認される（特徴量のヒント）
data_tbl %>%
  filter(id == ids[1]) %>%
  plot_seasonal_diagnostics(.date_var = Date,
                            .value    = log(Weekly_Sales))


# 3 データ分割 ------------------------------------------------------------------

# 予測ホライズン
# --- 全データが143Wなので、検証期間は約36％
FORECAST_HORIZON <- 52

# データ分割
# --- cumulativeは全データセットを使うことを意味する（デフォルトでは、訓練データの数を選ぶことができる）
splits <-
  data_tbl %>%
    time_series_split(date_var = Date,
                      assess = FORECAST_HORIZON,
                      cumulative = TRUE)

# データ確認
splits %>%
  tk_time_series_cv_plan() %>%
  group_by(.id, .key, id) %>%
  tally()

# データ確認
# --- 訓練データ：2010-02-05 to 2011-10-28
# --- 検証データ：2011-11-04 to 2012-10-26
data_tbl %>% filter(id == "1_1") %>% use_series(Date)
splits %>% training() %>% filter(id == "1_1") %>% use_series(Date)
splits %>% testing() %>% filter(id == "1_1") %>% use_series(Date)

# プロット確認
splits %>%
  tk_time_series_cv_plan() %>%
  plot_time_series_cv_plan(Date, Weekly_Sales)


# 4 特徴量エンジニアリング -----------------------------------------------------------

# レシピ作成
# --- 日付の詳細情報を追加（季節性をとらえるため）
recipe_spec <-
  recipe(Weekly_Sales ~ ., data = training(splits)) %>%
    step_timeseries_signature(Date) %>%
    step_normalize(Date_index.num, starts_with("Date_year")) 

# 確認
recipe_spec %>% prep() %>% juice() %>% glimpse()


# 5 モデル構築 --------------------------------------------------------------------

# H2O設定
# --- 起動
# --- プログレスバーの非表示
h2o.init(nthreads = -1, ip = 'localhost', port = 54321)
h2o.no_progress()

# モデル定義
# --- H2OのAutoML
model_spec_h2o <-
  automl_reg(mode = 'regression') %>%
    set_engine(engine                     = 'h2o',
               max_runtime_secs           = 30,
               max_runtime_secs_per_model = 10,
               max_models                 = 30,
               nfolds                     = 5,
               exclude_algos              = c("DeepLearning"),
               verbosity                  = NULL,
               seed                       = 786)

# 確認
model_spec_h2o %>% print()


# 6 学習 ---------------------------------------------------------------------------

# 学習
# --- AutoML実行
wflw_fit_h2o <-
  workflow() %>%
    add_model(model_spec_h2o) %>%
    add_recipe(recipe_spec) %>%
    fit(training(splits))

# 確認
wflw_fit_h2o %>% print()
wflw_fit_h2o %>% glimpse()

# リーダーボード確認
wflw_fit_h2o %>% automl_leaderboard()

  # 保存
# wflw_fit_h2o %>%
#     automl_update_model('XGBoost_grid__1_AutoML_20210407_141755_model_2') %>%
#     save_h2o_model(path = 'h2o_models/XGBoost_grid__1_AutoML_20210407_141755_model_2')


# ロード
# --- 学習済モデル
model_path <- "lab_53_h2o_workshop/h2o_models/XGBoost_grid__1_AutoML_20210407_141755_model_2"
modeltime.h2o::load_h2o_model(model_path)


# 7 モデルテーブル ---------------------------------------------------------------------------

# モデルテーブルへの登録
modeltime_tbl <- modeltime_table(wflw_fit_h2o)

# 確認
modeltime_tbl


# 8 モデル精度の検証 --------------------------------------------------------------------------

# キャリブレーション
# --- 検証データで予測値を取得
calibration_tbl <-
  modeltime_tbl %>%
    modeltime_calibrate(testing(splits)) 

# メトリックの確認
calibration_tbl %>% modeltime_accuracy() %>% table_modeltime_accuracy()


calibration_tbl$.calibration_data


# 9 予測データの作成 -------------------------------------------------------------------------

# プロット作成
# --- モデルテーブルをフラットデータに変換する
# --- modeltime_forecast()は信頼区間も出力される（calibrateでは予測値のみ）
calibration_tbl %>%
  modeltime_forecast(new_data    = testing(splits),
                     actual_data = data_tbl,
                     keep_data   = TRUE) %>%
  group_by(id) %>%
  plot_modeltime_forecast(.facet_ncol  = 2,
                          .interactive = TRUE)

# 10 リフィット ------------------------------------------------------------------------------

# リフィット
# --- 検証データを含めた全データで再学習
refit_tbl <-
  calibration_tbl %>%
    modeltime_refit(data_tbl)

# 将来データの作成
# --- データセットに存在しない日付を生成
future_tbl <-
  splits %>%
    testing() %>%
    group_by(id) %>%
    future_frame(Date, .length_out = 52) %>%
    ungroup()

# 予測の作成
refit_tbl %>%
  modeltime_forecast(new_data    = future_tbl,
                     actual_data = data_tbl,
                     keep_data   = TRUE) %>%
  group_by(id) %>%
  plot_modeltime_forecast(.facet_ncol  = 2,
                          .interactive = TRUE)
