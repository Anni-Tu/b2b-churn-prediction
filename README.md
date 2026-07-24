# Прогнозирование оттока клиентов в B2B-сегменте методами машинного обучения

ВКР по курсу «Наука о данных: профессиональный уровень (Data Science PRO)».

## Структура репозитория

```
notebooks/
├── 01_data_preparation.ipynb          — очистка данных, обработка пропусков и выбросов
├── 02_rfm.ipynb                       — построение RFM-признаков и целевой переменной churn
├── 03_modeling_churn_prediction.ipynb — обучение и сравнение моделей (LogReg, RandomForest, XGBoost, LightGBM)
├── 04_pipeline_1.ipynb                — production-класс ChurnPipeline (fit/predict/save/load)
└── 05_neural_network_comparison.ipynb — нейросеть (MLP) и сравнение с остальными моделями

app/
├── app.py                              — Streamlit-приложение для скоринга оттока
├── churn_pipeline.pkl                  — обученный пайплайн
├── dataset_anonymized_demo_sample.csv  — демо-выборка для приложения (1200 клиентов)
└── requirements.txt                    — зависимости приложения
```

## Задача

Прогнозирование вероятности оттока B2B-клиентов на основе истории заказов:
очистка данных → построение RFM/оконных признаков → обучение и сравнение
моделей (включая нейросеть) → production-пайплайн → веб-приложение для скоринга.

## Данные

Полные датасеты, использованные в ноутбуках, доступны на Яндекс.Диске:

[Датасеты (Яндекс.Диск)](https://disk.yandex.ru/d/zEnL6erWd6mlUg)
[Датасеты (Яндекс.Диск)](https://disk.yandex.ru/d/ry6hLMe93jOgmQ)

## Запуск приложения

```
cd app
pip install -r requirements.txt
streamlit run app.py
```

Приложение поддерживает два режима: загрузка собственного CSV или демо-данные
(`dataset_anonymized_demo_sample.csv`) — случайная выборка из 1200 клиентов
с полной историей заказов.
