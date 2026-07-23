import streamlit as st
import pandas as pd
import pickle
import joblib
import re
from datetime import datetime
import numpy as np
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import xgboost as xgb

def preprocess_raw_data(df):
    """
    Очищает сырые данные и заполняет пропуски по правилам, определённым в 01_data_preparation.
    """
    df = df.copy()
    df = df.drop_duplicates()
    df = df.dropna(subset=['client_nm'])
    df['order_dttm'] = pd.to_datetime(df['order_dttm'])

    # Заполнение client_type
    df['client_type'] = df['client_type'].fillna(
        df['okved'].notna().map({True: 'Юр. лицо', False: 'Физ. лицо'})
    )

    # Заполнение страны
    most_frequent_country = df['registration_country_nm'].mode()[0]
    df['registration_country_nm'] = df['registration_country_nm'].fillna(most_frequent_country)

    # Заполнение категорий товаров
    known_categories = (
        df[df['product_category'].notna()]
        .drop_duplicates('product_nm')
        .set_index('product_nm')['product_category']
        .to_dict()
    )
    df['product_category'] = df['product_category'].fillna(df['product_nm'].map(known_categories))
    df['product_category'] = df['product_category'].fillna('Другое')

    # Заполнение валюты
    df['currency_nm'] = df['currency_nm'].fillna(df['currency_nm'].mode()[0])

    # Заполнение okved
    mask_no_okved = df['client_type'].isin(['Физ. лицо', 'Индивидуальный предприниматель'])
    df.loc[mask_no_okved & df['okved'].isnull(), 'okved'] = 'Нет ОКВЭД'

    client_okved_mode = (
        df[df['client_type'] == 'Юр. лицо']
        .groupby('client_uuid')['okved']
        .agg(lambda x: x.mode()[0] if not x.mode().empty else None)
    )
    mask = (df['client_type'] == 'Юр. лицо') & (df['okved'].isnull())
    df.loc[mask, 'okved'] = df.loc[mask, 'client_uuid'].map(client_okved_mode)
    df['okved'] = df['okved'].fillna('Не указан')

    return df

def create_features(df, cutoff_date, usd_rate_series, compute_churn=True, churn_threshold_days=None):
    """
    Создаёт признаки для модели на основе истории до cutoff_date.

    compute_churn=True - режим обучения/оценки (нужны и история, и "будущее" для таргета).
    compute_churn=False - режим скоринга текущих клиентов, колонка 'churn' не создаётся.

    churn_threshold_days - если задан, используется как есть (нужно для согласованности,
    когда сравниваются два разных cutoff - порог должен быть одним и тем же).
    Если None и compute_churn=True - вычисляется по 95-му перцентилю интервалов
    (по истории этого среза) и возвращается наружу.

    Возвращает: (model_data, churn_threshold_days_used_or_None)
    """
    df = df.copy()

    df['order_date'] = df['order_dttm'].dt.date
    df['order_date_dt'] = pd.to_datetime(df['order_date'])

    df['usd_rate'] = df['order_date_dt'].map(usd_rate_series)
    df['usd_rate'] = df['usd_rate'].ffill()

    df['amount_usd'] = df['turnover_amount_amt']
    mask_rub = df['sales_currency'] == 'RUB'
    df.loc[mask_rub, 'amount_usd'] = df.loc[mask_rub, 'turnover_amount_amt'] / df.loc[mask_rub, 'usd_rate']

    history_df = df[df['order_dttm'] < cutoff_date].copy()
    all_clients = history_df['client_uuid'].unique()
    base_df = pd.DataFrame({'client_uuid': all_clients})

    used_churn_threshold = None

    if compute_churn:
        future_df = df[df['order_dttm'] >= cutoff_date].copy()

        future_client_orders = future_df.groupby('client_uuid')['order_dttm'].min().reset_index()
        future_client_orders.columns = ['client_uuid', 'first_future_order']
        base_df = base_df.merge(future_client_orders, on='client_uuid', how='left')

        client_orders = history_df.groupby('client_uuid')['order_dttm'].apply(list).reset_index()
        client_orders['intervals'] = client_orders['order_dttm'].apply(
            lambda dates: [(dates[i + 1] - dates[i]).days for i in range(len(dates) - 1)] if len(dates) > 1 else []
        )
        all_intervals = [item for sublist in client_orders['intervals'] for item in sublist]

        if churn_threshold_days is None:
            used_churn_threshold = int(np.percentile(all_intervals, 95)) if all_intervals else 30
        else:
            used_churn_threshold = churn_threshold_days

        base_df['churn'] = (
            (base_df['first_future_order'].isna()) |
            (base_df['first_future_order'] > cutoff_date + pd.Timedelta(days=used_churn_threshold))
        ).astype(int)
        base_df = base_df.drop(columns=['first_future_order'])

    # ---- базовые RFM ----
    rfm = history_df.groupby('client_uuid').agg(
        recency_days=('order_dttm', lambda x: (cutoff_date - x.max()).days),
        frequency=('order_uuid', 'nunique'),
        monetary_sum=('amount_usd', 'sum'),
        monetary_mean=('amount_usd', 'mean'),
        monetary_std=('amount_usd', 'std'),
        days_active=('order_dttm', lambda x: (x.max() - x.min()).days if len(x) > 1 else 0)
    ).reset_index()
    rfm.fillna({'monetary_std': 0}, inplace=True)

    cat_count = history_df.groupby('client_uuid')['product_category'].nunique().reset_index()
    cat_count.columns = ['client_uuid', 'unique_categories']
    prod_count = history_df.groupby('client_uuid')['product_uuid'].nunique().reset_index()
    prod_count.columns = ['client_uuid', 'unique_products']

    rfm = rfm.merge(cat_count, on='client_uuid', how='left')
    rfm = rfm.merge(prod_count, on='client_uuid', how='left')

    model_data = base_df.merge(rfm, on='client_uuid', how='inner')

    # ==========================================================================
    # ДОБАВЛЕНО: клиентские атрибуты и агрегаты по заказам (были только в 02_rfm,
    # в проде отсутствовали). Строим orders_hist САМОСТОЯТЕЛЬНО из history_df,
    # чтобы пайплайн не зависел от отдельно подготовленного orders.parquet -
    # при реальном скоринге новых данных такого файла может не быть.
    # Винзоризация считается только по истории до cutoff (не по всем заказам).
    # ==========================================================================
    orders_hist = history_df.groupby('order_uuid').agg(
        client_uuid=('client_uuid', 'first'),
        total_amount=('amount_usd', 'sum'),
        ecom_flg=('ecom_flg', 'first'),
        client_type=('client_type', 'first'),
        country=('registration_country_nm', 'first'),
        okved=('okved', 'first')
    ).reset_index()

    if len(orders_hist) > 0:
        p99_order = orders_hist['total_amount'].quantile(0.99)
        orders_hist['amount_wins'] = orders_hist['total_amount'].clip(upper=p99_order)
        orders_hist['is_large_order'] = (orders_hist['total_amount'] > p99_order).astype(int)

        client_attrs = orders_hist[['client_uuid', 'client_type', 'country', 'okved', 'ecom_flg']].drop_duplicates('client_uuid')
        agg_orders = orders_hist.groupby('client_uuid').agg(
            avg_amount_wins=('amount_wins', 'mean'),
            max_amount_wins=('amount_wins', 'max'),
            large_order_share=('is_large_order', 'mean')
        ).reset_index()
        client_features = client_attrs.merge(agg_orders, on='client_uuid', how='left')
        model_data = model_data.merge(client_features, on='client_uuid', how='left')

    numeric_client_cols = ['avg_amount_wins', 'max_amount_wins', 'large_order_share']
    for col in numeric_client_cols:
        if col in model_data.columns:
            model_data[col] = model_data[col].fillna(0)
        else:
            model_data[col] = 0
    for col in ['client_type', 'country', 'okved', 'ecom_flg']:
        if col not in model_data.columns:
            model_data[col] = 'Неизвестно'

    # ---- оконные статистики (без изменений) ----
    windows = [30, 60, 90, 180, 365]

    def calc_window_stats(df_window, window_days, cutoff_date):
        start_date = cutoff_date - pd.Timedelta(days=window_days)
        mask = df_window['order_dttm'] >= start_date
        recent = df_window[mask]
        stats_ = recent.groupby('client_uuid').agg(
            freq=('order_uuid', 'nunique'),
            sum_amount=('amount_usd', 'sum'),
            mean_amount=('amount_usd', 'mean'),
            std_amount=('amount_usd', 'std'),
            min_amount=('amount_usd', 'min'),
            max_amount=('amount_usd', 'max'),
            active_days=('order_dttm', lambda x: x.dt.date.nunique())
        ).reset_index()
        rename = {col: f'{col}_{window_days}d' for col in stats_.columns if col != 'client_uuid'}
        stats_.rename(columns=rename, inplace=True)
        return stats_

    def calc_recency_for_window(df_window, cutoff_date, window_days):
        start_date = cutoff_date - pd.Timedelta(days=window_days)
        recent = df_window[df_window['order_dttm'] >= start_date].groupby('client_uuid')['order_dttm'].max().reset_index()
        recent.columns = ['client_uuid', f'last_order_{window_days}d']
        recent[f'recency_{window_days}d'] = (cutoff_date - recent[f'last_order_{window_days}d']).dt.days
        return recent[['client_uuid', f'recency_{window_days}d']]

    all_stats = None
    for w in windows:
        stats_w = calc_window_stats(history_df, w, cutoff_date)
        all_stats = stats_w if all_stats is None else all_stats.merge(stats_w, on='client_uuid', how='outer')
        rec = calc_recency_for_window(history_df, cutoff_date, w)
        all_stats = all_stats.merge(rec, on='client_uuid', how='left')

    for col in all_stats.columns:
        if col != 'client_uuid':
            all_stats[col] = all_stats[col].fillna(0)

    model_data = model_data.merge(all_stats, on='client_uuid', how='left')

    rename_dict = {}
    for w in windows:
        for stat in ['sum', 'mean', 'std', 'min', 'max']:
            old, new = f'{stat}_amount_{w}d', f'{stat}_{w}d'
            if old in model_data.columns:
                rename_dict[old] = new
    if rename_dict:
        model_data.rename(columns=rename_dict, inplace=True)

    for w in windows:
        model_data[f'freq_share_{w}d'] = model_data[f'freq_{w}d'] / (model_data['frequency'] + 1e-6)

    model_data['avg_interval_days'] = np.where(
        model_data['frequency'] > 1, model_data['days_active'] / (model_data['frequency'] - 1), 0)

    model_data['monetary_cv'] = np.where(
        model_data['monetary_mean'] > 0, model_data['monetary_std'] / model_data['monetary_mean'], 0)
    model_data['monetary_cv'] = model_data['monetary_cv'].replace([np.inf, -np.inf], 0)

    model_data['recent_order_share'] = np.where(
        model_data['frequency'] > 0, model_data['freq_90d'] / model_data['frequency'], 0)

    model_data['monetary_trend'] = model_data['mean_90d'] - model_data['monetary_mean']

    monetary_cols = [
        'monetary_sum', 'monetary_mean', 'monetary_std',
        'sum_30d', 'sum_60d', 'sum_90d', 'sum_180d', 'sum_365d',
        'mean_30d', 'mean_60d', 'mean_90d', 'mean_180d', 'mean_365d',
        'monetary_cv'
    ]
    for col in monetary_cols:
        if col in model_data.columns:
            model_data[f'log_{col}'] = np.log1p(model_data[col])

    for w in windows:
        col = f'freq_{w}d'
        if col in model_data.columns:
            model_data[f'log_{col}'] = np.log1p(model_data[col])

    freq_cols = [f'freq_{w}d' for w in windows]
    sum_cols = [f'sum_{w}d' for w in windows]
    log_freq_cols = [f'log_{c}' for c in freq_cols]
    log_sum_cols = [f'log_{c}' for c in sum_cols]

    for col in log_freq_cols + log_sum_cols:
        if col not in model_data.columns:
            model_data[col] = 0

    x = np.log(windows)

    def slope_per_client(data, cols):
        slopes = []
        values = data[cols].values.astype(float)
        for row in values:
            y = np.nan_to_num(row, nan=0.0, posinf=0.0, neginf=0.0)
            if np.all(y == 0):
                slopes.append(0.0)
            else:
                try:
                    slope, _, _, _, _ = stats.linregress(x, y)
                    slopes.append(slope)
                except Exception:
                    slopes.append(0.0)
        return slopes

    model_data['freq_trend_slope'] = slope_per_client(model_data, log_freq_cols)
    model_data['sum_trend_slope'] = slope_per_client(model_data, log_sum_cols)

    for col in freq_cols + sum_cols:
        if col in model_data.columns:
            model_data[col] = pd.to_numeric(model_data[col], errors='coerce').fillna(0)

    model_data['freq_cv_windows'] = model_data[freq_cols].std(axis=1) / (model_data[freq_cols].mean(axis=1) + 1e-6)
    model_data['sum_cv_windows'] = model_data[sum_cols].std(axis=1) / (model_data[sum_cols].mean(axis=1) + 1e-6)
    model_data[['freq_cv_windows', 'sum_cv_windows']] = model_data[['freq_cv_windows', 'sum_cv_windows']].replace([np.inf, -np.inf], 0)

    def interval_quantiles(dates):
        if len(dates) < 2:
            return [np.nan, np.nan, np.nan]
        diffs_days = np.diff(sorted(dates)) / np.timedelta64(1, "D")
        return np.percentile(diffs_days, [25, 50, 75])

    client_intervals = history_df.groupby('client_uuid')['order_dttm'].apply(
        lambda x: interval_quantiles(x)
    ).reset_index()
    client_intervals[['p25_interval', 'median_interval', 'p75_interval']] = pd.DataFrame(
        client_intervals['order_dttm'].tolist(), index=client_intervals.index
    )
    client_intervals.drop(columns=['order_dttm'], inplace=True)

    model_data = model_data.merge(client_intervals, on='client_uuid', how='left')
    for col in ['p25_interval', 'median_interval', 'p75_interval']:
        model_data[col] = model_data[col].fillna(model_data[col].median())

    category_dummies = history_df.groupby('client_uuid')['product_category'].value_counts(normalize=True).unstack(fill_value=0)
    category_dummies.columns = [f'cat_share_{col}' for col in category_dummies.columns]
    category_dummies = category_dummies.reset_index()
    model_data = model_data.merge(category_dummies, on='client_uuid', how='left')

    # ---- сезонные доли: ФИКС - гарантируем все 12 месяцев / 4 квартала ----
    month_dummies = history_df.groupby('client_uuid')['order_dttm'].apply(
        lambda x: x.dt.month.value_counts(normalize=True)
    ).unstack(fill_value=0)
    month_dummies.columns = [f'month_share_{int(c)}' for c in month_dummies.columns]
    for m in range(1, 13):
        col = f'month_share_{m}'
        if col not in month_dummies.columns:
            month_dummies[col] = 0.0
    month_dummies = month_dummies.reset_index()

    quarter_dummies = history_df.groupby('client_uuid')['order_dttm'].apply(
        lambda x: x.dt.quarter.value_counts(normalize=True)
    ).unstack(fill_value=0)
    quarter_dummies.columns = [f'quarter_share_{int(c)}' for c in quarter_dummies.columns]
    for q in range(1, 5):
        col = f'quarter_share_{q}'
        if col not in quarter_dummies.columns:
            quarter_dummies[col] = 0.0
    quarter_dummies = quarter_dummies.reset_index()

    model_data = model_data.merge(month_dummies, on='client_uuid', how='left')
    model_data = model_data.merge(quarter_dummies, on='client_uuid', how='left')

    numeric_cols = model_data.select_dtypes(include=['number']).columns
    model_data[numeric_cols] = model_data[numeric_cols].fillna(0)

    cols_to_drop = [c for c in model_data.columns if c.startswith('last_order_')]
    model_data.drop(columns=[c for c in cols_to_drop if c in model_data.columns], errors='ignore', inplace=True)

    return model_data, used_churn_threshold

def sanitize_columns(df):
    """LightGBM (и на всякий случай другие модели) не разрешает символы
    '"', ',', ':', '{', '}', '[', ']' в именах признаков - они попадают в дамми-колонки
    из названий okved. Заменяем на '_', схлопываем повторы, разруливаем коллизии."""
    new_cols = []
    seen = {}
    for col in df.columns:
        clean = re.sub(r'[",:\{\}\[\]]', '_', str(col))
        clean = re.sub(r'\s+', '_', clean)
        clean = re.sub(r'_+', '_', clean).strip('_')
        if clean in seen:
            seen[clean] += 1
            clean = f"{clean}_{seen[clean]}"
        else:
            seen[clean] = 0
        new_cols.append(clean)
    df.columns = new_cols
    return df


class ChurnPipeline:
    DEFAULT_PARAMS = {
        'max_depth': 4,
        'learning_rate': 0.05,
        'n_estimators': 100,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'random_state': 42,
        'eval_metric': 'aucpr',
        'verbosity': 0
    }
    DEFAULT_THRESHOLD = 0.5
    CATEGORICAL_COLS = ['client_type', 'country', 'okved', 'ecom_flg']

    def __init__(self, model_type='xgb', params=None, threshold=None):
        self.model_type = model_type
        self.params = dict(params) if params else dict(self.DEFAULT_PARAMS)
        self.threshold = threshold if threshold is not None else self.DEFAULT_THRESHOLD
        self.model = None
        self.scaler = None
        self.feature_names = None
        self.usd_rate_series = None
        self.cutoff_date = None
        self.churn_threshold_days = None

    @staticmethod
    def _encode_categorical(df):
        """one-hot + фикс bool->int + санитизация имён колонок - три шага одним вызовом."""
        encoded = pd.get_dummies(df, columns=ChurnPipeline.CATEGORICAL_COLS, drop_first=True)
        bool_cols = encoded.select_dtypes(include='bool').columns
        encoded[bool_cols] = encoded[bool_cols].astype(int)
        encoded = sanitize_columns(encoded)
        return encoded

    @staticmethod
    def _select_features(X):
        variances = X.var()
        keep_cols = variances[variances > 0.01].index.tolist()
        corr_matrix = X[keep_cols].corr().abs()
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        to_drop = [c for c in upper.columns if any(upper[c] > 0.9)]
        return [c for c in keep_cols if c not in to_drop]

    def fit(self, df, cutoff_date, usd_rate_series, gap_days=150):
        self.usd_rate_series = usd_rate_series
        self.cutoff_date = cutoff_date
        cutoff_eval_train = cutoff_date - pd.Timedelta(days=gap_days)

        print(f"cutoff прод-модели (обучаем на всей истории до этой даты): {cutoff_date.date()}")
        print(f"cutoff внутренней оценки (train для честной проверки):     {cutoff_eval_train.date()}")

        # ---- честная time-based оценка методологии (в прод не идёт) ----
        eval_train_data, churn_threshold_days = create_features(
            df, cutoff_eval_train, usd_rate_series, compute_churn=True)
        eval_test_data, _ = create_features(
            df, cutoff_date, usd_rate_series, compute_churn=True,
            churn_threshold_days=churn_threshold_days)

        # one-hot вместе (чтобы у train/test совпал набор колонок ДО split),
        # затем сплит по _part, ЗАТЕМ санитизация каждой части по отдельности -
        # санитизация до split ломает имя служебной колонки _part (как было с _split в 03-м)
        eval_train_data['_part'] = 'train'
        eval_test_data['_part'] = 'test'
        eval_combined = pd.concat([eval_train_data, eval_test_data], ignore_index=True)

        eval_combined_encoded = pd.get_dummies(eval_combined, columns=self.CATEGORICAL_COLS, drop_first=True)
        bool_cols = eval_combined_encoded.select_dtypes(include='bool').columns
        eval_combined_encoded[bool_cols] = eval_combined_encoded[bool_cols].astype(int)

        eval_train_encoded = eval_combined_encoded[eval_combined_encoded['_part'] == 'train'].drop(columns=['_part'])
        eval_test_encoded = eval_combined_encoded[eval_combined_encoded['_part'] == 'test'].drop(columns=['_part'])

        eval_train_encoded = sanitize_columns(eval_train_encoded)
        eval_test_encoded = sanitize_columns(eval_test_encoded)

        eval_X_train = eval_train_encoded.drop(columns=['client_uuid', 'churn']).select_dtypes(include=[np.number])
        eval_y_train = eval_train_encoded['churn']
        eval_feature_names = self._select_features(eval_X_train)

        eval_X_train = eval_X_train[eval_feature_names]
        eval_X_test = eval_test_encoded.reindex(columns=eval_feature_names, fill_value=0)
        eval_y_test = eval_test_encoded['churn']

        eval_scaler = StandardScaler()
        eval_X_train_scaled = eval_scaler.fit_transform(eval_X_train)
        eval_X_test_scaled = eval_scaler.transform(eval_X_test)

        neg, pos = np.bincount(eval_y_train)
        eval_params = dict(self.params)
        eval_params['scale_pos_weight'] = neg / pos if pos > 0 else 1.0
        eval_model = xgb.XGBClassifier(**eval_params)
        eval_model.fit(eval_X_train_scaled, eval_y_train)

        from sklearn.metrics import roc_auc_score, average_precision_score
        eval_proba = eval_model.predict_proba(eval_X_test_scaled)[:, 1]
        print(f"[Time-based проверка] ROC-AUC: {roc_auc_score(eval_y_test, eval_proba):.4f}")
        print(f"[Time-based проверка] PR-AUC:  {average_precision_score(eval_y_test, eval_proba):.4f}")
        print("(эта оценка не используется для прод-модели ниже - только честная проверка методологии)\n")

        # ---- прод-модель: обучается на ВСЕХ данных до cutoff_date ----
        model_data, _ = create_features(
            df, cutoff_date, usd_rate_series, compute_churn=True,
            churn_threshold_days=churn_threshold_days)

        # здесь сплита нет (одна выборка) - можно кодировать и санитизировать одним вызовом
        model_data_encoded = self._encode_categorical(model_data)

        X = model_data_encoded.drop(columns=['client_uuid', 'churn'])
        y = model_data_encoded['churn']

        non_numeric = X.select_dtypes(exclude=[np.number]).columns.tolist()
        if non_numeric:
            print(f"Внимание: нечисловые колонки исключены из обучения: {non_numeric}")
            X = X.drop(columns=non_numeric)

        self.feature_names = self._select_features(X)
        X = X[self.feature_names]

        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)

        neg, pos = np.bincount(y)
        self.params['scale_pos_weight'] = neg / pos if pos > 0 else 1.0

        self.model = xgb.XGBClassifier(**self.params)
        self.model.fit(X_scaled, y)
        self.churn_threshold_days = churn_threshold_days

        print(f"Прод-модель обучена на {len(X)} клиентах (вся история до {cutoff_date.date()}), "
              f"{len(self.feature_names)} признаков.")

        return self

    def predict(self, df, cutoff_date=None):
        if cutoff_date is None:
            cutoff_date = self.cutoff_date

        model_data_new, _ = create_features(
            df, cutoff_date, self.usd_rate_series, compute_churn=False)

        # одна выборка, сплита нет - кодируем и санитизируем одним вызовом
        model_data_new_encoded = self._encode_categorical(model_data_new)

        for col in self.feature_names:
            if col not in model_data_new_encoded.columns:
                model_data_new_encoded[col] = 0

        X_new = model_data_new_encoded[self.feature_names]
        non_numeric_new = X_new.select_dtypes(exclude=[np.number]).columns.tolist()
        if non_numeric_new:
            raise ValueError(
                f"Нечисловые признаки при скоринге: {non_numeric_new}. "
                "Проверьте, что create_features возвращает те же типы, что и при обучении."
            )
        X_new_scaled = self.scaler.transform(X_new)
        proba = self.model.predict_proba(X_new_scaled)[:, 1]

        result = pd.DataFrame({
            'client_uuid': model_data_new_encoded['client_uuid'],
            'probability': proba
        })
        result['at_risk'] = (result['probability'] >= self.threshold).astype(int)

        # Обогащаем читаемыми полями для отображения: имя клиента и дата
        # последнего заказа - берём из той же истории (до cutoff_date),
        # которую видела модель при построении признаков.
        history_for_display = df[df['order_dttm'] < cutoff_date]
        client_info = history_for_display.groupby('client_uuid').agg(
            client_nm=('client_nm', 'first'),
            last_order_date=('order_dttm', 'max')
        ).reset_index()
        result = result.merge(client_info, on='client_uuid', how='left')
        result['client_nm'] = result['client_nm'].fillna('Без названия')

        return result.sort_values('probability', ascending=False).reset_index(drop=True)

    def save(self, path):
        joblib.dump({
            'model': self.model,
            'scaler': self.scaler,
            'feature_names': self.feature_names,
            'params': self.params,
            'threshold': self.threshold,
            'model_type': self.model_type,
            'usd_rate_series': self.usd_rate_series,
            'cutoff_date': self.cutoff_date,
            'churn_threshold_days': self.churn_threshold_days
        }, path)

    def load(self, path):
        data = joblib.load(path)
        self.model = data['model']
        self.scaler = data['scaler']
        self.feature_names = data['feature_names']
        self.params = data['params']
        self.threshold = data.get('threshold', self.DEFAULT_THRESHOLD)
        self.model_type = data['model_type']
        self.usd_rate_series = data['usd_rate_series']
        self.cutoff_date = data['cutoff_date']
        self.churn_threshold_days = data.get('churn_threshold_days')
        return self

# ------------------------------------------------------------------
# 2. Настройка страницы
# ------------------------------------------------------------------
st.set_page_config(page_title="Система оценки оттока клиентов B2B", layout="wide")
st.title("📊 Скоринг оттока клиентов (B2B)")
st.markdown("Загрузите данные клиентов и получите вероятность ухода каждого из них.")

# ------------------------------------------------------------------
# 3. Загрузка модели и курса валют (кэшируем, чтобы не перезагружать)
# ------------------------------------------------------------------
@st.cache_resource
def load_pipeline():
    # Курс USD уже сохранён ВНУТРИ churn_pipeline.pkl (см. ChurnPipeline.save() -
    # там есть ключ 'usd_rate_series'), поэтому отдельный файл usd_rate_series.pkl
    # загружать не нужно - pipeline.usd_rate_series используется автоматически
    # внутри pipeline.predict().
    pipeline = ChurnPipeline()  # Ваш класс
    pipeline.load('churn_pipeline.pkl')
    return pipeline

try:
    pipeline = load_pipeline()
    st.success("✅ Модель успешно загружена!")
except FileNotFoundError:
    st.error(
        "Файл churn_pipeline.pkl не найден рядом с app.py. "
        "Сначала обучите и сохраните пайплайн (pipeline.save('churn_pipeline.pkl')), "
        "затем положите файл в ту же папку, откуда запускаете streamlit."
    )
    st.stop()
except Exception as e:
    st.error(f"Ошибка загрузки модели: {e}")
    st.stop()

# ------------------------------------------------------------------
# 4. Боковая панель с настройками
# ------------------------------------------------------------------
st.sidebar.header("Настройки скоринга")

# Опция: использовать данные по умолчанию (изначальный файл) или загрузить свой
DEMO_OPTION = "Использовать демо-данные (dataset_anonymized_meaningful.csv)"
UPLOAD_OPTION = "Загрузить свой CSV-файл"

data_source = st.sidebar.radio(
    "Источник данных",
    (DEMO_OPTION, UPLOAD_OPTION)
)

uploaded_file = None
if data_source == UPLOAD_OPTION:
    uploaded_file = st.sidebar.file_uploader("Выберите CSV файл", type=["csv"])

# Кнопка запуска
run_button = st.sidebar.button("🚀 Запустить скоринг", type="primary")

st.sidebar.markdown("---")
st.sidebar.subheader("Фильтр актуальности")
active_window_days = st.sidebar.slider(
    "Показывать клиентов с заказом за последние N дней",
    min_value=30, max_value=730, value=365, step=30,
    help=(
        "Клиент без единого заказа дольше этого срока, скорее всего, уже потерян, "
        "а не 'находится в зоне риска' - предупреждать о нём поздно. Кроме того, для "
        "таких клиентов модель работает на грани экстраполяции (таких больших значений "
        "давности заказа она почти не видела на обучении) и часто выдаёт всем одинаковую "
        "максимальную вероятность. Уменьшите окно, чтобы видеть только тех, кого ещё "
        "реально можно удержать."
    )
)

# ------------------------------------------------------------------
# 5. Основная логика приложения
# ------------------------------------------------------------------
def run_scoring(df_raw):
    """Применяет пайплайн к сырым данным и возвращает результат."""
    # Очистка
    df_clean = preprocess_raw_data(df_raw)  # Ваша функция
    
    # Определяем cutoff (последняя дата в данных + 1 день, как у вас в ноутбуке)
    scoring_cutoff = df_clean['order_dttm'].max() + pd.Timedelta(days=1)
    
    # Предсказание
    predictions = pipeline.predict(df_clean, cutoff_date=scoring_cutoff)
    return predictions

# ------------------------------------------------------------------
# 6. Отображение результатов
# ------------------------------------------------------------------
if run_button:
    # Загружаем данные
    if data_source == DEMO_OPTION:
        try:
            df_raw = pd.read_csv('dataset_anonymized_meaningful.csv')
            st.info("📁 Используются демо-данные")
        except FileNotFoundError:
            st.error("Файл dataset_anonymized_meaningful.csv не найден в папке с приложением.")
            st.stop()
    else:
        if uploaded_file is not None:
            df_raw = pd.read_csv(uploaded_file)
            st.info("📁 Загружен пользовательский файл")
        else:
            st.warning("Пожалуйста, загрузите CSV-файл.")
            st.stop()

    # Показываем прогресс
    with st.spinner("Выполняется скоринг клиентов..."):
        try:
            result_df_all = run_scoring(df_raw)
        except Exception as e:
            st.error(f"Ошибка во время скоринга: {e}")
            st.stop()

    # Отсекаем клиентов, которые не заказывали дольше выбранного окна -
    # они уже потеряны, показывать их как "риск" неактуально и вводит в заблуждение
    scoring_cutoff = pd.to_datetime(result_df_all['last_order_date']).max() + pd.Timedelta(days=1)
    is_active = pd.to_datetime(result_df_all['last_order_date']) >= scoring_cutoff - pd.Timedelta(days=active_window_days)
    result_df = result_df_all[is_active].reset_index(drop=True)
    lost_clients = len(result_df_all) - len(result_df)

    # --- ВЫВОД ДАННЫХ  ---

    if lost_clients > 0:
        st.caption(
            f"ℹ️ {lost_clients:,} клиентов без заказов дольше {active_window_days} дней скрыты из списка "
            "как уже потерянные (не 'в зоне риска', а фактический отток в прошлом). "
            "Измените окно в боковой панели, чтобы включить их обратно."
        )

    # 6.1. Метрики
    col1, col2, col3 = st.columns(3)
    total_clients = len(result_df)
    risk_clients = result_df['at_risk'].sum()
    risk_rate = risk_clients / total_clients * 100
    
    col1.metric("Активных клиентов", f"{total_clients:,}")
    col2.metric("Клиенты в зоне риска", f"{risk_clients:,}", delta=f"{risk_rate:.1f}%")
    col3.metric("Средняя вероятность оттока", f"{result_df['probability'].mean():.2%}")

    # 6.2. Гистограмма распределения вероятностей (настоящая гистограмма по бинам
    # 0-5%, 5-10% ... а не просто отсортированные значения по номеру строки)
    st.subheader("📈 Распределение вероятностей оттока")
    bins = np.arange(0, 1.05, 0.05)
    hist_counts, bin_edges = np.histogram(result_df['probability'], bins=bins)
    hist_df = pd.DataFrame({
        'Вероятность оттока': [f"{bin_edges[i]:.0%}–{bin_edges[i+1]:.0%}" for i in range(len(bin_edges) - 1)],
        'Число клиентов': hist_counts
    }).set_index('Вероятность оттока')
    st.bar_chart(hist_df, height=300)

    # 6.2b. Риск оттока в разрезе даты последнего заказа - показывает, что чем
    # дольше клиент не покупал, тем выше вероятность оттока (проверка здравого смысла модели)
    st.subheader("🗓️ Риск оттока по дате последнего заказа")
    result_df['last_order_date'] = pd.to_datetime(result_df['last_order_date'])
    result_df['last_order_month'] = result_df['last_order_date'].dt.to_period('M').dt.to_timestamp()
    monthly = result_df.groupby('last_order_month').agg(
        avg_probability=('probability', 'mean'),
        at_risk_count=('at_risk', 'sum')
    )
    col_a, col_b = st.columns(2)
    with col_a:
        st.caption("Средняя вероятность оттока по месяцу последнего заказа")
        st.line_chart(monthly['avg_probability'], height=280)
    with col_b:
        st.caption("Число клиентов в зоне риска по месяцу последнего заказа")
        st.bar_chart(monthly['at_risk_count'], height=280)

    # 6.3. Таблица с результатами (ТОП-20 самых рискованных) - имя клиента и дата
    # последнего заказа вместо голого UUID, короткий ID оставлен для сверки
    st.subheader("🔴 Топ-20 клиентов с наибольшим риском")
    display_df = result_df.head(20).copy()
    display_df['client_id_short'] = display_df['client_uuid'].str[:8]
    # NumberColumn не умножает значение сам - без явного *100 доля 0.95 показывалась
    # бы как "0.95%" вместо "95.00%"
    display_df['probability_pct'] = display_df['probability'] * 100
    st.dataframe(
        display_df,
        column_config={
            "client_nm": "Клиент",
            "client_id_short": "ID (кратко)",
            "last_order_date": st.column_config.DateColumn("Последний заказ", format="DD.MM.YYYY"),
            "probability_pct": st.column_config.NumberColumn("Вероятность оттока", format="%.2f%%"),
            "at_risk": st.column_config.CheckboxColumn("В зоне риска"),
        },
        column_order=["client_nm", "client_id_short", "last_order_date", "probability_pct", "at_risk"],
        hide_index=True,
        use_container_width=True
    )

    # 6.4. Полная таблица (можно скрыть за спойлером для ВКР)
    with st.expander("📋 Показать полный список всех клиентов"):
        full_df = result_df.copy()
        full_df['client_id_short'] = full_df['client_uuid'].str[:8]
        full_df['probability_pct'] = full_df['probability'] * 100
        st.dataframe(
            full_df,
            column_config={
                "client_nm": "Клиент",
                "client_id_short": "ID (кратко)",
                "last_order_date": st.column_config.DateColumn("Последний заказ", format="DD.MM.YYYY"),
                "probability_pct": st.column_config.NumberColumn("Вероятность оттока", format="%.2f%%"),
                "at_risk": st.column_config.CheckboxColumn("В зоне риска"),
            },
            column_order=["client_nm", "client_id_short", "last_order_date", "probability_pct", "at_risk"],
            hide_index=True,
            use_container_width=True
        )

    # 6.5. Кнопка скачать результат (CSV)
    csv = result_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="💾 Скачать результаты скоринга (CSV)",
        data=csv,
        file_name=f"scoring_results_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
        mime="text/csv"
    )