# ******************************************************************************
# Title       : BSU Learning Lab
# Theme       : Production Data Science Pipelines with Targets
# Chapter     : Lab 56: TARGETS KERAS CHURN
# Module      : 02_keras_analysis.R
# Update Date : 2021/6/30
# URL         : https://university.business-science.io/courses/enrolled/
# ******************************************************************************


# ＜目標＞
# - {keras}を用いた分類問題のチュートリアルを確認する


# ＜目次＞
# 0 準備
# 1 データ
# 2 データ分割
# 3 前処理
# 4 モデリング
# 5 予測


# 0 準備 -----------------------------------------------------------------------

# ライブラリ
library(keras)
library(tidymodels)
library(tidyquant)
library(tidyverse)
library(reticulate)


# 仮想環境の設定
use_condaenv("r-tf", required = TRUE)
py_config()


# 1 データ ---------------------------------------------------------------------

# データロード
churn_data_raw <- read_csv("lab_56_targets_churn/data/churn.csv")
churn_data_raw %>% glimpse()

# データ加工
churn_data_tbl <-
  churn_data_raw %>%
    select(-customerID) %>%
    drop_na() %>%
    select(Churn, everything())

# データ確認
churn_data_tbl %>% glimpse()


# 2 データ分割 ------------------------------------------------------------------

# データ分割
set.seed(100)
train_test_split <- churn_data_tbl %>% initial_split(prop = 0.8)
train_test_split

# データ格納
train_tbl <- train_test_split %>% training()
test_tbl  <- train_test_split %>% testing()


# 3 前処理 --------------------------------------------------------------------

# ＜ポイント＞
# - ディープラーニングでは以下の処理が必要
#   --- 数値データ：基準化
#   --- カテゴリカルデータ：ダミー変換して数値化

# データ確認
train_tbl %>% glimpse()

# レシピ作成
# --- tenureは分位のカテゴリカルデータに変換
# --- カテゴリカルデータはダミー変換
# --- 数値は基準化
rec_obj <-
  recipe(Churn ~ ., data = train_tbl) %>%
    step_discretize(tenure, options = list(cuts = 6)) %>%
    step_log(TotalCharges) %>%
    step_dummy(all_nominal(), -all_outcomes()) %>%
    step_center(all_predictors(), -all_outcomes()) %>%
    step_scale(all_predictors(), -all_outcomes())

# レシピ実行
rec_prepped <- rec_obj %>% prep(data = train_tbl)

# 処理結果の確認
rec_prepped %>% juice() %>% glimpse()

# レシピ適用
x_train_tbl <- rec_prepped %>% bake(new_data = train_tbl) %>% select(-Churn)
x_test_tbl  <- rec_prepped %>% bake(new_data = test_tbl) %>% select(-Churn)

# データ確認
x_train_tbl %>% glimpse()

# ラベルデータの取得
# --- {kears}にはベクトルで渡す必要がある
y_train_vec <- ifelse(pull(train_tbl, Churn) == "Yes", 1, 0)
y_test_vec  <- ifelse(pull(test_tbl, Churn) == "Yes", 1, 0)


# 4 モデリング -----------------------------------------------------------------

# モデル準備
# --- シーケンシャルモデル
model_keras <- keras_model_sequential()

# モデル定義
model_keras %>%
    layer_dense(units              = 16,
                kernel_initializer = "uniform",
                activation         = "relu",
                input_shape        = ncol(x_train_tbl)) %>%
    layer_dropout(rate = 0.1) %>%
    layer_dense(units              = 16,
                kernel_initializer = "uniform",
                activation         = "relu") %>%
    layer_dropout(rate = 0.1) %>%
    layer_dense(units              = 1,
                kernel_initializer = "uniform",
                activation         = "sigmoid") %>%
    compile(optimizer = 'adam',
            loss      = 'binary_crossentropy',
            metrics   = c('accuracy'))

# モデル確認
model_keras %>% print()

# 学習
history <-
  model_keras %>%
    fit(x                = as.matrix(x_train_tbl),
        y                = y_train_vec,
        batch_size       = 50,
        epochs           = 35,
        validation_split = 0.30)

# 結果確認
history %>% print()

# プロット作成
history %>%
  plot() +
    theme_tq() +
    scale_color_tq() +
    scale_fill_tq()


# 5 予測 --------------------------------------------------------------------

# クラス分類の予測
yhat_keras_class_vec <-
  model_keras %>%
    predict_classes(x = as.matrix(x_test_tbl)) %>%
    as.vector()

# クラス確率の予測
yhat_keras_prob_vec <-
  model_keras %>%
    predict_proba(x = as.matrix(x_test_tbl)) %>%
    as.vector()

# テーブル作成
# --- 正解値/予測値/クラス確率
estimates_keras_tbl <-
  tibble(truth      = as.factor(y_test_vec) %>% fct_recode(yes = "1", no = "0"),
         estimate   = as.factor(yhat_keras_class_vec) %>% fct_recode(yes = "1", no = "0"),
         class_prob = yhat_keras_prob_vec)

# 確認
estimates_keras_tbl %>% print()

# 予測精度の評価
# --- 混合行列（不均衡問題になっている）
# --- AUC
# --- Accuracy
estimates_keras_tbl %>% conf_mat(truth, estimate)
estimates_keras_tbl %>% roc_auc(truth, class_prob, event_level = 'second')
estimates_keras_tbl %>% accuracy(truth, estimate)
