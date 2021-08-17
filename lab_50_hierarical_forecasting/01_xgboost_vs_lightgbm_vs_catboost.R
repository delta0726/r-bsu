# ******************************************************************************
# Title       : BSU Learning Lab
# Chapter     : LAB 50: LIGHTGBM
# Theme       : MODEL COMPARISONS（LIGHTGBM VS XGBOOST VS CATBOOST）
# Module      : 01_xgboost_vs_lightgbm_vs_catboost.R
# Update Date : 2021/8/18
# URL         : https://university.business-science.io/courses/enrolled/
# ******************************************************************************


# ＜概要＞
# - 勾配ブースティングの代表的な３アルゴリズムを確認する
# - {treesnip}を使うことでtidymodelsのフレームワークに統一して扱うことができることを確認する


# ＜インストール＞
# - TREESNIP: remotes::install_github("curso-r/treesnip")
# - CATBOOST: devtools::install_github('catboost/catboost', subdir = 'catboost/R-package')


# ＜目次＞
# 0 準備
# 1 lightGBMの基本フロー
# 2 データ加工
# 3 関数定義
# 4 LightGBM
# 5 XGBoost
# 6 CatBoost
# 7 ワークフローを用いたソリューション


# 0 準備 -----------------------------------------------------------------------

# ライブラリ
library(lightgbm)
library(catboost)
library(xgboost)
library(treesnip)
library(tidymodels)
library(tidyverse)


# 1 lightGBMの基本フロー -------------------------------------------------------

# データロード
# --- ライブラリのサンプルデータ
data(agaricus.train, package='lightgbm')
data(agaricus.test, package='lightgbm')

# データ確認
# --- 疎データ
agaricus.train %>% print()
agaricus.test %>% print()

# モデル構築
train  <- agaricus.train
dtrain <- lgb.Dataset(train$data, label = train$label)

# 学習
model <-
  lgb.train(params = list(objective = "regression" , metric = "l2"),
            data = dtrain)

# 予測
model %>% predict(agaricus.test$data)


# 2 データ加工 ----------------------------------------------------------

# 訓練データ
# --- ラベルデータをファクターに変換
agaricus_train_tbl <-
  agaricus.train$data %>%
    as.matrix() %>% 
    as_tibble() %>%
    add_column(target = agaricus.train$label, .before = 1) %>%
    mutate(target = factor(target))

# 検証データ
# --- ラベルデータをファクターに変換
agaricus_test_tbl <-
  agaricus.test$data %>%
    as.matrix() %>% 
    as_tibble() %>%
    add_column(target = agaricus.test$label, .before = 1) %>%
    mutate(target = factor(target))


# 3 関数定義 -----------------------------------------------------------

# モデル構築＆学習
train_model <- function(model_spec, train = agaricus_train_tbl) {
    workflow() %>%
      add_model(spec = model_spec) %>%
      add_recipe(recipe = recipe(target ~ ., train)) %>%
      fit(train)
}

# 予測データの作成
make_predictions <- function(model, test = agaricus_test_tbl, type = "prob") {
    predict(model, test, type = type) %>%
        bind_cols(test %>% select(target)) %>%
        mutate(target = factor(target))
}

# モデル抽出
extract_fit_parsnip <- function(model) {
    model %>%
      pull_workflow_fit() %>%
      pluck("fit")
}


# 4 LightGBM -----------------------------------------------------------

# モデル構築＆学習
agaricus_lightgbm_fit_wflw <-
  boost_tree(mode       = "classification",
             learn_rate = 2) %>%
    set_engine("lightgbm") %>%
    train_model(agaricus_train_tbl)

# モデル精度の確認
# --- 予測データの作成
# --- AUCの算出
agaricus_lightgbm_fit_wflw %>%
  make_predictions(agaricus_test_tbl, type = "prob") %>%
  yardstick::roc_auc(target, .pred_1, event_level = "second")

# 変数重要度の出力
agaricus_lightgbm_fit_wflw %>%
  extract_fit_parsnip() %>%
  lightgbm::lgb.importance() %>%
  lightgbm::lgb.plot.importance()


# 5 XGBoost -----------------------------------------------------------

# モデル構築＆学習
agaricus_xgboost_fit_wflw <- 
  boost_tree(mode = "classification") %>%
    set_engine("xgboost") %>%
    train_model(agaricus_train_tbl)

# モデル精度の確認
# --- 予測データの作成
# --- AUCの算出
agaricus_xgboost_fit_wflw %>%
  make_predictions(agaricus_test_tbl, type = "prob") %>%
  yardstick::roc_auc(target, .pred_1, event_level = "second")

# 変数重要度の出力
agaricus_xgboost_fit_wflw %>%
  extract_fit_parsnip() %>%
  xgboost::xgb.importance(model = .) %>%
  xgboost::xgb.plot.importance()


# 6 CatBoost -----------------------------------------------------------

# モデル構築＆学習
agaricus_catboost_fit_wflw <-
  boost_tree(mode = "classification") %>%
    set_engine("catboost") %>%
    train_model(agaricus_train_tbl)

# モデル精度の確認
# --- 予測データの作成
# --- AUCの算出
agaricus_catboost_fit_wflw %>%
  make_predictions(agaricus_test_tbl, type = "prob") %>%
  yardstick::roc_auc(target, .pred_1, event_level = "second")

# プロット作成
# --- 変数重要度の出力
agaricus_catboost_fit_wflw %>%
  extract_fit_parsnip() %>%
  catboost::catboost.get_feature_importance() %>%
  as_tibble(rownames = "feature") %>%
  rename(value = V1) %>%
  arrange(-value) %>%
  mutate(feature = as_factor(feature) %>% fct_rev()) %>%
  dplyr::slice(1:10) %>%
  ggplot(aes(value, feature)) +
  geom_col()


# 7 ワークフローを用いたソリューション --------------------------------------------

# * 準備 -----------------------------------------------

# データ確認
diamonds

# CVデータの作成
set.seed(123)
diamonds_splits <- vfold_cv(diamonds, v = 5)

# レシピ定義
# --- フォーミュラのみ定義
recipe_spec <- recipe(price ~ ., data = diamonds)

# XGBoost
doParallel::registerDoParallel(8)
resamples_xgboost_tbl <- 
    workflow() %>%
    add_model(boost_tree(mode = 'regression') %>% set_engine("xgboost")) %>%
    add_recipe(recipe_spec %>% step_dummy(all_nominal())) %>%
    fit_resamples(resamples = diamonds_splits, 
                  control   = control_resamples(verbose = TRUE, allow_par = FALSE))

# LightGBM
doParallel::registerDoParallel(8)
resamples_lightgbm_tbl <- 
    workflow() %>%
    add_model(boost_tree(mode = 'regression') %>% set_engine("lightgbm")) %>%
    add_recipe(recipe_spec) %>%
    fit_resamples(resamples = diamonds_splits, 
                  control   = control_resamples(verbose = TRUE, allow_par = TRUE))

# Catboost
doParallel::registerDoParallel(8)
resamples_catboost_tbl <- workflow() %>%
    add_model(boost_tree(mode = 'regression') %>% set_engine("catboost")) %>%
    add_recipe(recipe_spec) %>%
    fit_resamples(resamples = diamonds_splits, 
                  control   = control_resamples(verbose = TRUE, allow_par = TRUE))

# 結果比較
bind_rows(resamples_catboost_tbl %>% collect_metrics() %>% add_column(.model = "catboost"),
          resamples_lightgbm_tbl %>% collect_metrics() %>% add_column(.model = "lightgbm"),
          resamples_xgboost_tbl %>% collect_metrics() %>% add_column(.model = "xgboost"))

