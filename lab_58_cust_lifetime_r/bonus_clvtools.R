# ******************************************************************************
# Title       : BSU Learning Lab
# Theme       : Marketing Analytics with R & Python
# Chapter     : Lab 58: Customer Lifetime Value (CLV) with R Shiny
# Module      : bonus_clvtools.R
# Update Date : 2021/6/27
# URL         : https://university.business-science.io/courses/enrolled/
# ******************************************************************************


# ＜目次＞
# 0 準備
# 1 データ準備
# 2 コホート分析
# 3 CLVToolsモデリング
# 4 予測（顧客消費額）
# 5 参考資料


# 0 準備 ----------------------------------------------------------------------

# ライブラリ
library(CLVTools)
library(plotly)
library(tidyquant)
library(tidyverse)
library(timetk)
library(lubridate)


# 1 データ準備 ----------------------------------------------------------------

# データロード
# --- テキストファイルを読込
cdnow_raw_tbl <-
  vroom::vroom(file = "lab_58_cust_lifetime_r/data/CDNOW_master.txt",
               delim = " ",
               col_names = FALSE)

# データ加工
cdnow_tbl <-
  cdnow_raw_tbl %>%
    select(X2, X3, X5, X8) %>%
    set_names(c("customer_id", "date", "quantity", "price")) %>%
    mutate(date = ymd(as.character(date))) %>%
    drop_na()

# データ確認
cdnow_tbl %>% print()
cdnow_tbl %>% glimpse()


# 2 コホート分析 ----------------------------------------------------------------

# * 準備 ------------------------------------------------

# 初回購入日の取得
# --- カスタマーIDごと
cdnow_first_purchase_tbl <-
  cdnow_tbl %>%
    group_by(customer_id) %>%
    slice_min(date) %>%
    ungroup()

# 期間の確認
# --- 初回購入日
cdnow_first_purchase_tbl %>% pull(date) %>% range()


# * コホート期間の設定 ----------------------------------------

# ＜ポイント＞
# - 分析対象期間を1997-01-01 to 1997-03-31と設定する


# ID抽出
# --- 対象期間に購入したカスタマーID
ids_in_cohort <-
  cdnow_first_purchase_tbl %>%
    filter_by_time(.date_var = date, .start_date = "1997-01", .end_date   = "1997-03") %>%
    distinct(customer_id) %>%
    pull(customer_id)

# データ抽出
# --- 対象期間に購入したIDのみ
cdnow_cohort_tbl <-
  cdnow_tbl %>%
    filter(customer_id %in% ids_in_cohort)


# * 可視化: コホートの売上高 --------------------------------

# プロット作成
# --- 期間ごとの売上高
cdnow_cohort_tbl %>%
    summarize_by_time(total_price = sum(price, na.rm = TRUE),
                      .by   = "month") %>%
    plot_time_series(date, total_price, .y_intercept = 0)


# * 可視化: 個人の売上高 ------------------------------------


n    <- 1:10
ids  <- unique(cdnow_cohort_tbl$customer_id)[n]

# プロット作成
# --- 個別IDごとにファセットを作成
# --- なぜかgeom_point()が作動しないので削除
cdnow_cohort_tbl %>%
    filter(customer_id %in% ids) %>%
    group_by(customer_id) %>%
    plot_time_series(date, price,
                     .y_intercept = 0,
                     .smooth      = FALSE,
                     .facet_ncol  = 2)


# 3 CLVToolsモデリング --------------------------------------------------------

# * CLV Data ----
cdnow_cohort_clv <-
  cdnow_cohort_tbl %>%
      clvdata(date.format       = "ymd",
              time.unit         = "day",
              estimation.split  = "1998-01-01",
              name.id           = "customer_id",
              name.date         = "date",
              name.price        = "price")

# データ確認
cdnow_cohort_clv %>% print()
cdnow_cohort_clv %>% summary()


# * PNBD METHOD ----

?CLVTools::pnbd

# モデル構築
# --- Pareto/NBD models
model_pnbd <- cdnow_cohort_clv %>% pnbd()

# サマリー
model_pnbd %>% summary()
model_pnbd %>% coef()


# 4 予測 ----------------------------------------------------------------

# 予測
model_pnbd %>%
  predict(cdnow_cohort_clv) %>%
  as_tibble()

g <- model_pnbd %>% plot()


gg <- g$data %>%
    ggplot(aes(period.until, value, color = variable)) +
    geom_line() +
    scale_color_tq() +
    theme_minimal()

ggplotly(gg)


model_gg <-
  cdnow_cohort_clv %>%
    gg(remove.first.transaction = FALSE)

model_gg

# 予測
model_gg %>% predict()

model_gg %>% plot()


# 5 参考資料 ----------------------------------------------------------------

# R Examples:
#   https://bookdown.org/mike/marketing_research/customer-lifetime-value-clv.html
#   https://www.clvtools.com/index.html
#   https://github.com/bachmannpatrick/CLVTools

# Python Examples:
#    https://github.com/CamDavidsonPilon/lifetimes
#    https://lifetimes.readthedocs.io/en/latest/

# Case Studies:
#   https://hbr.org/2007/10/how-valuable-is-word-of-mouth


# Marketing Mix Modeling
#    Python Tutorial: https://towardsdatascience.com/marketing-channel-attribution-with-markov-chains-in-python-part-2-the-complete-walkthrough-733c65b23323
#    Basic Marketing Mix Model in Sklearn: https://practicaldatascience.co.uk/machine-learning/how-to-create-a-basic-marketing-mix-model-in-scikit-learn


