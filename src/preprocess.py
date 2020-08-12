import pandas as pd
import numpy as np
from datetime import datetime, timedelta


def financial_ratios(raw_data):
    """
    Compute multiple financial ratios
    Require quarterly or yearly financial data matrix as input
    :type raw_data: data frame
    """

    # Profitability ratios
    operating_margin = raw_data.operating_Income / raw_data.revenues
    operating_profit_margin = operating_margin * 100
    net_income = raw_data.earnings_Per_Share * raw_data.number_Of_Shares
    return_on_total_equity = net_income / raw_data.total_Equity
    return_on_total_liabilites_and_equity = net_income / raw_data.total_Liabilities_And_Equity
    return_on_total_assets = net_income / raw_data.total_Assets
    roa_dp = (net_income / raw_data.revenues) * (raw_data.revenues / raw_data.total_Assets)
    gross_profit_margin = raw_data.gross_Income / raw_data.revenues
    cash_flow = raw_data.cash_Flow_From_Operating_Activities + \
                raw_data.cash_Flow_From_Investing_Activities + raw_data.cash_Flow_From_Financing_Activities

    free_cash_flow = raw_data.free_Cash_Flow
    capital_employed = raw_data.total_Assets - raw_data.current_Liabilities
    cash_flow_return_on_investment = raw_data.cash_Flow_From_Operating_Activities / capital_employed
    Return_on_Capital_Employed = raw_data.profit_Before_Tax / capital_employed
    net_gearing = raw_data.net_Debt / raw_data.total_Equity
    basic_earning_power_ratio = raw_data.profit_Before_Tax / raw_data.total_Assets
    return_on_net_assets = raw_data.profit_Before_Tax / (
            raw_data.tangible_Assets + raw_data.current_Assets - raw_data.current_Liabilities)

    # Liquidity ratios
    current_ratio = raw_data.current_Assets / raw_data.current_Liabilities
    quick_ratio = raw_data.cash_And_Equivalents / raw_data.current_Liabilities
    operating_cash_flow_ratio = raw_data.cash_Flow_From_Operating_Activities / raw_data.net_Debt

    # Activity ratios / efficiency ratios
    asset_turnover = raw_data.revenues / raw_data.total_Assets

    # Debt ratios / leveraging ratios
    debt_ratio = (raw_data.current_Liabilities + raw_data.non_Current_Liabilities) / raw_data.total_Assets
    debt_to_equity_ratio = raw_data.net_Debt / raw_data.total_Equity
    total_liability_to_equity_ratio = (raw_data.current_Liabilities + raw_data.non_Current_Liabilities) / \
                                      raw_data.total_Equity

    # Market ratios
    payout_ratio = raw_data.dividend / raw_data.earnings_Per_Share
    dividend_cover = raw_data.earnings_Per_Share / raw_data.dividend
    PE_ratio = raw_data.stock_Price_Average / raw_data.earnings_Per_Share
    total_shareholders_equity = raw_data.total_Assets - (
                raw_data.non_Current_Liabilities + raw_data.current_Liabilities)
    book_value_per_share = total_shareholders_equity / raw_data.number_Of_Shares
    PB_ratio = raw_data.stock_Price_Average / book_value_per_share
    PS_ratio = raw_data.stock_Price_Average / raw_data.revenues
    market_capitalization = raw_data.stock_Price_Average * raw_data.number_Of_Shares
    enterprise_value = market_capitalization + raw_data.net_Debt - raw_data.cash_And_Equivalents
    EV_EBITDA_ratio = enterprise_value / raw_data.profit_Before_Tax
    dividend_yield = raw_data.dividend / raw_data.stock_Price_Average

    # Not computed
    # risk_adjusted_return_on_capital = expectedReturn / economicCapital
    # average colletion period = accounts recievable / (annual credit sales / 365 days)
    # degree of operating leverage = percent change in net operating income / percent change in sales
    # DSO ratio = accounts reciavable / (total annual sales / 365 days )
    # average payment period = accounts payable /(annual credit purchase / 365 days)
    # stock turnover ratio = cost of goods sold /average inventory
    # recievables turnover ratio = net credit sales / average net recievables
    # inventory conversion ratio = 365 days / inventory turnover
    # Receivables conversion period = recievables / net salwes   * 365 days
    # Payables conversion period = accounts payables / purchases
    # time_interest_earned_ratio = net income / annual interest expense
    # debt_service_coverage_ratio = net operating income / total debt service
    # roa_dp = (net_income / net_sales)*(net_sales / total_assets)
    # ROC = EBIT*(1-tax_rate)/invested_Capital
    # roe_DP = (net_income/rawStockQuarterD.revenues)*(rawStockQuarterD.revenues/average_assets)*
    # (average_assets/average_equity)
    # efficiency_ratio=non_intereset_expense/revenue
    # PEG_ratio = PE_ratio / annaul eps growth
    # cash_flow_ratio = raw_data.stock_Price_Average / present value of cash flow per share

    ratiolist = pd.concat([operating_margin, operating_profit_margin, net_income, return_on_total_equity,
                           return_on_total_liabilites_and_equity, return_on_total_assets, roa_dp, gross_profit_margin,
                           cash_flow, free_cash_flow, capital_employed, cash_flow_return_on_investment,
                           Return_on_Capital_Employed, net_gearing, basic_earning_power_ratio, return_on_net_assets,
                           current_ratio, quick_ratio, operating_cash_flow_ratio, asset_turnover, debt_ratio,
                           debt_to_equity_ratio, total_liability_to_equity_ratio, payout_ratio, dividend_cover,
                           PE_ratio, total_shareholders_equity, book_value_per_share, PB_ratio, PS_ratio,
                           market_capitalization, enterprise_value, EV_EBITDA_ratio, dividend_yield],
                          axis=1,
                          keys=["operating_margin", "operating_profit_margin", "net_income", "return_on_total_equity",
                                "return_on_total_liabilites_and_equity", "return_on_total_assets", "roa_dp",
                                "gross_profit_margin", "cash_flow", "free_cash_flow", "capital_employed",
                                "cash_flow_return_on_investment", "Return_on_Capital_Employed", "net_gearing",
                                "basic_earning_power_ratio", "return_on_net_assets", "current_ratio", "quick_ratio",
                                "operating_cash_flow_ratio", "asset_turnover", "debt_ratio", "debt_to_equity_ratio",
                                "total_liability_to_equity_ratio", "payout_ratio", "dividend_cover", "PE_ratio",
                                "total_shareholders_equity", "book_value_per_share", "PB_ratio", "PS_ratio",
                                "market_capitalization", "enterprise_value", "EV_EBITDA_ratio", "dividend_yield"])

    return ratiolist


def fill_quarter_data(all_data, shift_size):
    today = datetime.now()
    i = 0
    missing_count = 0
    num_stocks = len(all_data)
    for stock in range(0, num_stocks):

        if len(all_data[stock]["Quarterly data"]) == 0:
            print(all_data[stock]["Ticker"] + " has no quarterly data")
            i += 1
            missing_count += 1
            continue

        if len(all_data[stock]["Daily data"]) == 0:
            print(all_data[stock]["Ticker"] + " has no Daily data")
            i += 1
            missing_count += 1
            continue

        quarter_df = pd.DataFrame.from_dict(all_data[stock]["Quarterly data"])
        price_df = pd.DataFrame.from_dict(all_data[stock]["Daily data"])

        price_df = price_df.drop(price_df[price_df.volume == 0].index)  # Drop rows with 0 volume aka no trade

        quarter_df["report_End_Date"] = pd.to_datetime(quarter_df["report_End_Date"])
        price_df["Time"] = pd.to_datetime(price_df["Time"])
        quarter_df["reported_quarter"] = quarter_df["report_End_Date"] + timedelta(days=shift_size)

        # This removes bruten räkneskapsår
        # TODO filter out quarter instead of removing, check broken_fiscal_year column
        if quarter_df["report_End_Date"][0] > today:
            print(all_data[stock]["Ticker"] + " has brutet räkneskapsår")
            i += 1
            missing_count += 1
            continue

        # This removes those with not enough price data
        # TODO filter out rows instead of removing
        if quarter_df["report_End_Date"].min() < price_df["Time"].min():
            print(all_data[stock]["Ticker"] + " has missing price data")
            i += 1
            missing_count += 1
            continue

        if quarter_df["report_End_Date"].max() > price_df["Time"].max():
            print(all_data[stock]["Ticker"] + " has missing price data")
            i += 1
            missing_count += 1
            continue

        # Compute all financial ratios
        quarter_ratios = financial_ratios(quarter_df)
        quarter_values = quarter_df.drop(["stock_Price_Average", "stock_Price_High", "stock_Price_Low"], axis=1)
        quarter_df = quarter_df.sort_values(by="report_End_Date", ascending=False)

        quarter_ratios.reset_index(drop=True, inplace=True)
        quarter_values.reset_index(drop=True, inplace=True)
        full_quarter_matrix1 = pd.concat([quarter_ratios,
                                          quarter_values], axis=1)

        full_quarter_matrix1["Ticker"] = all_data[stock]["Ticker"]
        full_quarter_matrix1["Sector"] = all_data[stock]["Sector"]
        full_quarter_matrix1["Market"] = all_data[stock]["Market"]
        full_quarter_matrix1["Country"] = all_data[stock]["Country"]

        # For each column add daily pricing/volume data, for both fiscal quarters and actual observed reports
        num_quarters = len(quarter_df["reported_quarter"])
        k = 0
        try:
            for stamp in range(num_quarters):
                if k == num_quarters - 1:
                    reported_subset = price_df[
                        (price_df['Time'] > quarter_df["reported_quarter"][stamp] - timedelta(days=90)) &
                        (price_df['Time'] <= quarter_df["reported_quarter"][stamp])]

                    fiscal_subset = price_df[
                        (price_df['Time'] > quarter_df["reported_quarter"][stamp] - timedelta(days=90)) &
                        (price_df['Time'] <= quarter_df["report_End_Date"][stamp])]

                else:
                    reported_subset = price_df[(price_df['Time'] > quarter_df["reported_quarter"][(stamp + 1)]) &
                                               (price_df['Time'] <= quarter_df["reported_quarter"][stamp])]
                    fiscal_subset = price_df[(price_df['Time'] > quarter_df["report_End_Date"][(stamp + 1)]) &
                                             (price_df['Time'] <= quarter_df["report_End_Date"][stamp])]

                # print(reported_subset.mean().values.reshape(1, 5))

                quarter_aggregates = pd.concat([pd.DataFrame(reported_subset.mean().values.reshape(1, 5),
                                                             columns=("reported_mean_Close", "reported_mean_High",
                                                                      "reported_mean_Low",
                                                                      "reported_mean_Open", "reported_mean_volume")),
                                                pd.DataFrame(reported_subset.median().values.reshape(1, 5),
                                                             columns=("reported_median_Close", "reported_median_High",
                                                                      "reported_median_Low",
                                                                      "reported_median_Open",
                                                                      "reported_median_volume")),
                                                pd.DataFrame(reported_subset.std().values.reshape(1, 5),
                                                             columns=(
                                                                 "reported_std_Close", "reported_std_High",
                                                                 "reported_std_Low",
                                                                 "reported_std_Open", "reported_std_volume")),
                                                pd.DataFrame(fiscal_subset.mean().values.reshape(1, 5),
                                                             columns=(
                                                                 "fiscal_mean_Close", "fiscal_mean_High",
                                                                 "fiscal_mean_Low",
                                                                 "fiscal_mean_Open", "fiscal_mean_volume")),
                                                pd.DataFrame(fiscal_subset.median().values.reshape(1, 5),
                                                             columns=("fiscal_median_Close", "fiscal_median_High",
                                                                      "fiscal_median_Low",
                                                                      "fiscal_median_Open", "fiscal_median_volume")),
                                                pd.DataFrame(fiscal_subset.std().values.reshape(1, 5),
                                                             columns=(
                                                                 "fiscal_std_Close", "fiscal_std_High",
                                                                 "fiscal_std_Low",
                                                                 "fiscal_std_Open", "fiscal_std_volume"))], axis=1)
                if k == 0:
                    all_quarter_aggregates = quarter_aggregates
                else:
                    all_quarter_aggregates = pd.concat([all_quarter_aggregates, quarter_aggregates], axis=0)
                k += 1
        except Exception:
            print(all_data[stock]["Ticker"] + " has missing price data")
            missing_count += 1

        # Combine price aggregation data with quarterly data
        full_quarter_matrix1.reset_index(drop=True, inplace=True)
        all_quarter_aggregates.reset_index(drop=True, inplace=True)
        full_quarter_matrix = pd.concat([full_quarter_matrix1, all_quarter_aggregates], axis=1)

        if i == 0:
            all_stocks_matrix = full_quarter_matrix
        else:
            all_stocks_matrix = pd.concat([all_stocks_matrix, full_quarter_matrix], axis=0)
        i += 1

        print("Ticker " + str(i) + " of " + str(num_stocks) + " ,  " + all_data[stock]["Ticker"] + " Updated")

    return all_stocks_matrix, missing_count


class QuarterDataMLReady:
    def __init__(self, full_quarter_data, target_name, date_variables, seq_length=4, holdout_date="2020-03-31"):
        # Handle NA's
        full_quarter_data.replace([np.inf, -np.inf], np.nan, inplace=True)
        full_quarter_data = full_quarter_data.fillna(0)

        self.raw_features = full_quarter_data.drop([target_name], axis=1)
        self.target = full_quarter_data[target_name]

        # self.one_hot_encode_variables = pd.get_dummies(self.raw_features.select_dtypes(include=['object']))
        self.one_hot_encode_variables = pd.get_dummies(
            self.raw_features[
                self.raw_features.columns.difference(self.raw_features.filter(regex="Ticker").columns)
            ].select_dtypes(include=['object']))

        self.normalized_features = self.normalize(self.raw_features)
        self.processed_features = pd.concat([self.one_hot_encode_variables,
                                             self.normalized_features,
                                             self.raw_features[date_variables],
                                             self.raw_features["Ticker"]], axis=1)

        self.all_data = pd.concat([self.processed_features, self.target], axis=1)

        self.train_feature_array, self.train_target_array, self.holdout_feature_array, self.holdout_target_array = \
            self.rnn_ready(self.processed_features, self.target, seq_length=seq_length, holdout_date=holdout_date)

    def normalize(self, raw_features, norm_type="maxmin"):
        data1 = raw_features.select_dtypes(include=['float64'])
        # data3 = data1[data1.columns.difference( data1.filter(regex=("reported")).columns)]

        if norm_type == "maxmin":
            normalized_features = (data1 - data1.min()) / (data1.max() - data1.min())
        if norm_type == "stdmean":
            normalized_features = (data1 - data1.mean()) / (data1.std())

        return normalized_features

    def rnn_ready(self, processed_features, target, seq_length, holdout_date):

        i = 0
        for ticker in processed_features.Ticker.unique():
            # print(ticker)
            feature_array = processed_features[processed_features['Ticker'] == ticker]
            feature_array = np.array(feature_array[feature_array.columns.difference(["Ticker"])])
            target_array = target[processed_features['Ticker'] == ticker]
            # target_array = np.array(data_iteration[target_name])

            num_samples = target_array.shape[0] - seq_length
            if (num_samples <= 0):
                print("Not enough quarter data for stock : " + ticker)
                continue

            # feature_array2 = np.nan_to_num(feature_array, nan=0, posinf=0, neginf=0)
            feature_array_reshaped = np.array(
                [np.stack(feature_array[iter:iter + seq_length]) for iter in range(1, num_samples + 1)])
            target_array_reshaped = target_array[0:num_samples]

            if i == 0:
                full_feature_array = feature_array_reshaped
                full_target_array = target_array_reshaped
            else:
                full_feature_array = np.concatenate((full_feature_array, feature_array_reshaped))
                full_target_array = np.concatenate((full_target_array, target_array_reshaped))
            i = +1

        # Split holdout and train set
        # Timestamps full_feature_array[0][0][[93,103]]
        holdout_index = np.where(full_feature_array[:, 0, 93] >= datetime.strptime(holdout_date, '%Y-%m-%d'))
        train_index = np.where(full_feature_array[:, 0, 93] < datetime.strptime(holdout_date, '%Y-%m-%d'))

        holdout_feature_array = full_feature_array[holdout_index, :, ][0]
        holdout_feature_array = np.delete(holdout_feature_array, 93, axis=2)

        train_feature_array = full_feature_array[train_index, :, :][0]
        train_feature_array = np.delete(train_feature_array, 93, axis=2)

        holdout_target_array = full_target_array[holdout_index]
        train_target_array = full_target_array[train_index]

        return train_feature_array, train_target_array, holdout_feature_array, holdout_target_array
