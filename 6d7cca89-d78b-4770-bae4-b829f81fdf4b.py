#!/usr/bin/env python
# coding: utf-8

# ## Подключитесь к базе. Загрузите таблицы sql

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import sklearn.metrics

from sqlalchemy import create_engine
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split, RandomizedSearchCV, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import confusion_matrix, f1_score, roc_auc_score, roc_curve, precision_score, accuracy_score, recall_score, precision_recall_curve
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.dummy import DummyClassifier
from sklearn.feature_selection import SelectFromModel
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier


# In[2]:


db_config = {
'user': 'praktikum_student', # имя пользователя,
'pwd': 'Sdf4$2;d-d30pp', # пароль,
'host': 'rc1b-wcoijxj3yxfsf3fs.mdb.yandexcloud.net',
'port': 6432, # порт подключения,
'db': 'data-science-vehicle-db' # название базы данных,
} 


connection_string = 'postgresql://{}:{}@{}:{}/{}'.format(
    db_config['user'],
    db_config['pwd'],
    db_config['host'],
    db_config['port'],
    db_config['db'],
)


# In[3]:


engine = create_engine(connection_string)


# ## Проведите первичное исследование таблиц

# ## Таблица case_ids

# <div class="alert alert-info">
# <font size="4">🍕<b> Комментарий ревьюера</b></font>
#     <br /> 
#     <font size="3", color = "black">
# <br />
# Перед тем как непосредственно исследовать таблицы можно проверить схему БД. Это можно сделать примерно таким запросом:
# 
# `SELECT table_name
# FROM information_schema.tables
# WHERE table_type = 'BASE TABLE'`

# In[4]:


query = '''

SELECT *
FROM case_ids
LIMIT 5

'''

case_id_df = pd.read_sql_query(query, con=engine) 
case_id_df


# <div class="alert alert-success">
# <font size="4"><b>✔️ Комментарий ревьюера</b></font>
#     <br /> 
#     <font size="3", color = "black">
# <br />    
# Здорово, что выгружаешь табличку по лимиту 5, а не целиком. Для предварительного анализа таблицы полностью нам не требуются - только оперативку бы зря потратили )

# Таблица case_ids содержит уникальные case_id. Информации о db_year нет, но скорее всего, это год создания базы данных.

# ## Таблица vehicles

# In[5]:


query = '''

SELECT *
FROM vehicles
LIMIT 5

'''

vehicles_df = pd.read_sql_query(query, con=engine) 
vehicles_df


# Имеет неуникальные case_id и неуникальные party_number, которые сопоставляются с таблицей collisions и таблицей parties. Если нужен уникальный идентификатор, это case_id and party_number.

# ## Таблица parties

# In[6]:


query = '''

SELECT *
FROM parties
LIMIT 5

'''

parties_df = pd.read_sql_query(query, con=engine) 
parties_df


# Имеет неуникальный case_id, который сопоставляется с соответствующим ДТП в таблице collisions. Каждая строка здесь описывает одну из сторон, участвующих в ДТП. Если столкнулись две машины, в этой таблице должно быть две строки с совпадением case_id. Если нужен уникальный идентификатор, это case_id and party_number

# ## Таблица collisions

# In[7]:


query = '''

SELECT *
FROM collisions
LIMIT 5
'''

collisions_df = pd.read_sql_query(query, con=engine) 
collisions_df


# Таблица collisions описывает общую информацию о ДТП. Например, где оно произошло и когда. Имеет уникальный case_id.

# Все таблицы имеют общию ключ case_id. Могут отличаться названия столбцов, но в целом все данные на месте и соответствуют условию задачи.

# <div class="alert alert-success">
# <font size="4"><b>✔️ Комментарий ревьюера</b></font>
#     <br /> 
#     <font size="3", color = "black">
# <br />    
# 👌

# ##  Проведите статистический анализ факторов ДТП

# ## Выявление наиболее аварийных месяцев

# Выясним, в какие месяцы происходит наибольшее количество аварий. Для этого выгрузим нужные столбцы из таблицы collisions, посчитаем количество case_id и сгруппируем по месяцам.

# In[8]:


query = '''

WITH help AS
(SELECT DATE_TRUNC('MONTH', collision_date)::date AS collisions_month, 
       count(case_id) AS collisions_count
FROM collisions
WHERE collision_date BETWEEN '2009-01-01' AND '2012-06-01'
GROUP BY collisions_month)

SELECT EXTRACT(MONTH FROM collisions_month) AS month,
    AVG(collisions_count) AS collisions_avg
FROM help
GROUP BY month
'''
coll_count_df = pd.read_sql_query(query, con=engine) 
coll_count_df


# Построим график по полученной таблице:

# In[9]:




coll_count_df.sort_values(by='collisions_avg', ascending=False).plot(x='month',
                                                                          y='collisions_avg',
                                                                          kind='bar',
                                                                          grid=True,
                                                                          legend=False,
                                                                          rot=0,
                                                                          figsize=(12,6));
plt.title('Количество аварий по месяцам');
plt.xlabel('Месяц');
plt.ylabel('Количество аварий');


# <div class="alert alert-warning">
# <font size="4"><b>⚠️ Комментарий ревьюера V2</b></font>
#     <br /> 
#     <font size="3", color = "black">
# <br />
# Принято. Но для понимания есть ли сезонность всё-таки стоило не сортировать по числу аварий, а оставить в привычном порядке следования месяцев: от января к декабрю.

# Наибольшее количество ДТП наблюдается в период с января по май, дальше идет резкое снижение, пик приходится на март. Такой всплеск можно обьяснить неблагоприятными погодными условиями и состоянием дорог в зимне-весенний период, либо же неполнотой данных за период с июня по декабрь.

# 

# In[10]:


query = '''

SELECT DISTINCT date_trunc('month', collision_date)::date AS collision_month,
                count(case_id) AS case_count
FROM collisions
GROUP BY 1
HAVING count(case_id) < 30000
ORDER by 2 

'''
missing_count_df = pd.read_sql_query(query, con=engine) 
missing_count_df


# ## Постановка задач для рабочей группы

# 
# Выяснить, в какие месяцы за все время наблюдений чаще всего наблюдаются неблагоприятное состояние дороги по естественным причинам - мокрая либо заснеженная.
# 
# Выявить пять округов, в которых наиболее часто виновниками аварии являются пьяные водители в ночное время (с 23:00 до 05:00), а также средний возраст автомобилей по этим округам
# 
# Проанализировать сумму страховых выплат по месяцам, оценить среднюю страховку. влияет ли сезонность на сумму?
# 
# Влияет ли тип дорожного покрытия и степень естественного освещения на степень вины водителя в ДТП?
# 
# Выяснить, автомобиль с каким типом кузова чаще всего становится виновником аварии.
# 
# Зависит ли количество аварий от типа кузова автомобиля? столько участников вовлечено в такие ДТП?

# <div class="alert alert-success">
# <font size="4"><b>✔️ Комментарий ревьюера</b></font>
#     <br /> 
#     <font size="3", color = "black">
# <br />    
# Задач ровно 6, как и требует ТЗ 👌

# ### Задача 1

# Выясним, в какие месяцы за все время наблюдений чаще всего наблюдаются неблагоприятное состояние дороги по естественным причинам - мокрая либо заснеженная.

# In[11]:


query = '''

SELECT DISTINCT extract(MONTH FROM cast(collision_date AS date))::int AS collision_month,
                count(road_surface) AS bad_surface_count
FROM collisions
WHERE road_surface = 'wet'
  OR road_surface = 'snowy'
GROUP BY 1
ORDER BY 2 DESC

'''
bad_surface_df = pd.read_sql_query(query, con=engine) 
bad_surface_df


# In[12]:


plt.figure(figsize=[10,6])
sns.lineplot(data=bad_surface_df, x='collision_month', y='bad_surface_count', label='Число ДТП', color='purple')

plt.legend()
plt.title('Случаи ДТП на мокрой/снежной дороге по месяцам за 2011 год')
plt.xlabel('Месяц')
plt.ylabel('Количество ДТП')
plt.grid()
plt.show()


# Как и ожидалось чаще всего неблагоприятные погодные условия наблюдаются с октября по март, пиковый месяц - февраль

# ### Задача 2

# Выявить пять округов, в которых наиболее часто виновниками аварии являются пьяные водители в ночное время (с 23:00 до 05:00), а также средний возраст автомобилей по этим округам.

# In[13]:


query = '''

SELECT county_location,
       count(c.case_id) AS case_count,
       round(avg(v.vehicle_age), 1) AS avg_vehicle_age
FROM parties p
JOIN collisions c ON c.case_id = p.case_id
JOIN vehicles v ON c.case_id = v.case_id
WHERE at_fault = 1
  AND party_sobriety like '%%had been%%'
  AND (extract(HOUR FROM cast(collision_time AS TIME))::int <= 5
       OR extract(HOUR FROM cast(collision_time AS TIME))::int >= 23)
GROUP BY county_location
ORDER BY 2 DESC
LIMIT 5

'''
drunk_df = pd.read_sql_query(query, con=engine) 
drunk_df


# Лос-Анджелес занимает первое место, а возраст автомобиля примерно одинаковый

# ### Задача 3 

# Влияние возраста автомобился на тяжесть аварии

# In[14]:


query = '''
SELECT c.collision_damage,
       AVG(v.vehicle_age) AS avg_vehicle_age
FROM collisions AS c 
JOIN vehicles AS v ON c.case_id=v.case_id
GROUP BY c.collision_damage
'''

avg_vehicle_age = pd.read_sql_query(query, con=engine).sort_values(by='avg_vehicle_age', ascending=False)

sns.barplot(data=avg_vehicle_age, 
                 x='avg_vehicle_age', 
                 y='collision_damage');
plt.gcf().set_size_inches(12,6);
plt.subplots_adjust(top=.95);


# Чем выше средний возвраст автомобился, тем серьезнее повредения и также чаще допускаются царапины водителями

# <div class="alert alert-success">
# <font size="4"><b>✔️ Комментарий ревьюера V2</b></font>
#     <br /> 
#     <font size="3", color = "black">
# <br />    
# 👌

# ## Создайте модель для оценки водительского риска

# ### Подготовка набора данных

# Прежде чем создать модель, выявим факторы, влияющие на целевой признак at_fault.
# 
# Таблица vehicles:
# 
# -vehicle_transmission - авто с МКПП может быть сложнее для неопытных водителей, использующих каршеринг.
# 
# Таблица parties:
# 
# -at_fault - целевой признак;
# -cellphone_in_use - использование телефона снижает внимательность;
# -party_sobriety - трезвость водителя;
# 
# Таблица collisions:
# 
# -weather_1 - погода;
# -intersection - чаще всего, по статистике, ДТП происходят именно на перекрестках;
# -pcf_violation_category- основная причина аварии, выявление причины позволит предотвращать аварии в будущем;
# -motor_vehicle_involved_with - дополнительные участники ДТП, выявление наиболее частых участников позволит уменьшить число дтп;
# -road_surface - состояние поверхности дороги - гололед, мокрая/скользкая/сухая дорога;
# -control_device - наличие контролирующего утройства - неизвестный фактор, который модет оказать влияние
# -lighting - освещенность дороги.
# 
# 
# С помощью запроса выгрузим необходимые данные в таблицу. По условию задачи, нужны только данные за 2012 год, где участник аварии - машина и с повреждениями больше чем царапина.
# 
# 
# 

# In[15]:


query = '''

SELECT DISTINCT c.case_id,
       weather_1,
       pcf_violation_category,
       vehicle_age,
       motor_vehicle_involved_with,
       road_surface,
       control_device,
       lighting,
       vehicle_transmission,
       cellphone_in_use, 
       party_sobriety,
       at_fault
       
FROM collisions c
inner JOIN parties p ON c.case_id = p.case_id inner JOIN vehicles v ON c.case_id = v.case_id 
WHERE (extract(YEAR FROM cast(collision_date AS date))::int = 2012
  AND party_type = 'car'
  AND collision_damage != 'scratch')


'''
df = pd.read_sql_query(query, con=engine) 

df.head()


# In[16]:


df.info()


# In[17]:


df.isna().sum()/len(df)


# In[18]:


df = df.dropna() 


# Удаляем пропущенные значения , так как их доля очент мала

# In[19]:


df.duplicated().sum()


# Дубликаты отсутсвуют

# In[20]:


df = df.drop('case_id', axis=1) 


# In[21]:


df['lighting'].unique() 


# In[22]:


def fix_value(cell):
     cell = cell.replace(' ', '_')  # функция, заменяющая все пробелы на подчеркивания
     return cell 


# In[23]:


df['lighting'] = df['lighting'].apply(lambda x: 'dark' if 'dark' in x else x)
df['lighting'] = df['lighting'].apply(fix_value)


# In[24]:


df['lighting'].unique() 


# Избавились от большого количество категорий, объединив все категории обозначающие улицу без света в одну

# In[25]:


df['party_sobriety'].unique() 


# In[26]:


df['party_sobriety'] = df['party_sobriety'].apply(lambda x: 'sober' if 'had not' in x else x)
df['party_sobriety'] = df['party_sobriety'].apply(lambda x: 'drunk' if 'had been' in x else x)
df['party_sobriety'] = df['party_sobriety'].apply(lambda x: 'drunk' if 'impairment' in x else x)

df['party_sobriety'] = df['party_sobriety'].apply(fix_value)

df['party_sobriety'].unique()


# Уменьшили количество категорий, ообъединив их по смыслу 

# Далее рассмотрим категории нарушений. Здесь можно обьединить нарушения правил пешеходами в одну группу, тк задача - рассмотреть причины только автомобильных аварий. Также можно объединить в одну категорию технические неисправности.

# In[27]:


df['pcf_violation_category'].unique() 


# In[28]:


df['pcf_violation_category'] = df['pcf_violation_category'].apply(lambda x: 'equipment_fault' if x == 'brakes' 
                                                                  or x == 'other equipment'
                                                                  or x == 'lights' else x)
df['pcf_violation_category'] = df['pcf_violation_category'].apply(lambda x: 'pedestrian_involved' if x == 'automobile right of way' 
                                                                  or x == 'pedestrian violation'  
                                                                  or x == 'other than driver (or pedestrian)'
                                                                  or x == 'pedestrian right of way'
                                                                  or x == 'improper passing' else x)

df['pcf_violation_category'] = df['pcf_violation_category'].apply(fix_value)

df['pcf_violation_category'].unique() 


# In[29]:


df['motor_vehicle_involved_with'].unique() 


# In[30]:


df['motor_vehicle_involved_with'] = df['motor_vehicle_involved_with'].apply(lambda x: 'other_vehicle' if x == 'other motor vehicle' 
                                                                  or x == 'motor vehicle on other roadway' else x)

df['motor_vehicle_involved_with'] = df['motor_vehicle_involved_with'].apply(lambda x: 'fixed object' if x == 'parked motor vehicle' else x)

df['motor_vehicle_involved_with'] = df['motor_vehicle_involved_with'].apply(fix_value)

df['motor_vehicle_involved_with'].unique() 


# In[31]:


# КОД РЕВЬЮЕРА
df.duplicated().sum()


# In[32]:


df= df.drop_duplicates()


# In[33]:


df.duplicated().sum()


# In[34]:


print('vehicle_age')
sum_outliers = (df['vehicle_age'] > 19).sum()
print(f'Количество выбросов {sum_outliers}')


# In[35]:


df = df.query('vehicle_age <= 19')


# In[37]:


df.describe()


# <div class="alert alert-success">
# <font size="4"><b>✔️ Комментарий ревьюера V2</b></font>
#     <br /> 
#     <font size="3", color = "black">
# <br />    
# 👌

# Понял

# In[38]:


# КОД РЕВЬЮЕРА
df.describe()


# Избавился от выбросов

# ## Проведите анализ важности факторов ДТП

# Проведем простое кодирование категориальных признаков для оценки зависимостей:

# In[111]:


categorial = ['weather_1', 
              'pcf_violation_category',
              'motor_vehicle_involved_with', 
              'road_surface',
              'control_device', 
              'lighting', 
              'vehicle_transmission', 
              'party_sobriety' ]


# In[112]:


df_tmp = pd.get_dummies(df, columns=categorial, drop_first=True)


# In[113]:


numeric =['cellphone_in_use']

scaler = StandardScaler()
scaler.fit(df_tmp[numeric])
df_tmp[numeric] = scaler.transform(df_tmp[numeric])

pd.options.mode.chained_assignment = None


# In[114]:


fig, ax = plt.subplots(figsize=(16,14))
df_tmp.corr().iloc[1].sort_values(ascending=False).drop('at_fault',axis = 0).plot.bar(ax=ax)
ax.set_title("Важность признаков")
ax.set_ylabel('Важность')
plt.grid()

fig.tight_layout()


# 
# Из графика видим, что прямое влияние на виновность в ДТП оказывает:
# 
# -Отсутвие контрольного устройства
# -Столкновения с неподвижными объектами
# -Различный нарушения ПДД
# -Мокрая или заснеженая дорога
# -Автомобили с МКПП
# -Использование телефона
# 
# 
# Обратная зависимость:
# 
# -Трезвый водитель
# -При дневном свете
# -По вине пешеходов

# ## Разделение на выборки и кодирование признаков

# In[115]:


X_train, X_test, y_train, y_test = train_test_split(df.drop('at_fault', axis=1), 
                                                    df['at_fault'], 
                                                    train_size=0.8, 
                                                    random_state=42, 
                                                    shuffle=True,
                                                   stratify=df['at_fault'])


# <div class="alert alert-info">
# <font size="4">🍕<b> Комментарий ревьюера</b></font>
#     <br /> 
#     <font size="3", color = "black">
# <br />
# Настоятельно тебе рекомендую рандом стейт (и другие глобальные константы) в начале работы сохранять в отдельную переменную и оперировать дальше ей. Иногда бывает нужно провести эксперимент с другим рандомом и менять по коду во всех местах где он испоьзуется явно хуже, чем одну переменную в начале поменять.

# In[116]:


tmp_train = X_train[categorial]
tmp_test= X_test[categorial]


encoder_ohe = OneHotEncoder(handle_unknown='ignore')
encoder_ohe.fit(X_train[categorial])

tmp_train = pd.DataFrame(encoder_ohe.transform(X_train[categorial]).toarray(), 
                                   columns=encoder_ohe.get_feature_names(),
                                   index=X_train.index)
tmp_test = pd.DataFrame(encoder_ohe.transform(X_test[categorial]).toarray(), 
                                   columns=encoder_ohe.get_feature_names(),
                                   index=X_test.index)

X_train.drop(categorial, axis=1, inplace=True)
X_train = X_train.join(tmp_train)

X_test.drop(categorial, axis=1, inplace=True)
X_test = X_test.join(tmp_test)


# In[117]:


class_frequency = y_train.value_counts(normalize=True)
print(class_frequency)
class_frequency.plot(kind='bar')


# In[118]:


class_frequency[0]/class_frequency[1]


# Соотношение классов близко к 1:1, их можно считать сбалансированными.
# 
# При сравнении моделей в качестве определяющей метрики будем использовать точность (precision), так как нам важно оценить, сколько водителей, определенных, как виновные в аварии, действительно оказалась виновными.
# 
# Для вычисления метрик и построения графика определим функцию metrics_func:

# In[119]:


def metrics_func(model):
    model.fit(X_train, y_train)
    predicted_test = model.predict(X_test)
    probabilities_test = model.predict_proba(X_test)
    probabilities_one_test = probabilities_test[:, 1]
    precision = precision_score(y_test, predicted_test)
    recall = recall_score(y_test, predicted_test)
    print('Точность:', precision)
    print('Полнота:', recall)
    print('Доля правильных ответов:', accuracy_score(y_test, predicted_test)) 
    print('F1-мера:', f1_score(y_test, predicted_test))
    print('AUC-ROC:', roc_auc_score(y_test, probabilities_one_test), '\n')
    print('Матрица ошибок:')
    print(confusion_matrix(y_test, predicted_test))

# строим ROC-кривую
    fpr, tpr, thresholds = roc_curve(y_test, probabilities_one_test)

    plt.figure()

    plt.plot([0, 1], [0, 1], linestyle='--')
    plt.plot(fpr, tpr)

    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC-кривая')
    
    plt.show()
    return precision, predicted_test, probabilities_one_test, model


# <div class="alert alert-info">
# <font size="4">🍕<b> Комментарий ревьюера</b></font>
#     <br /> 
#     <font size="3", color = "black">
# <br />
# Недостатком метрики F1 является то, что для неё точность и полнота равнозначны, а в нашем проекте это не так. Низкая точность снизит наше доверие к водителям, из-за чего мы можем потерять их лояльность. Низкая полнота - угроза здоровью и жизни людей. Второе важнее. Поэтому есть смысл взять <a href="https://scikit-learn.org/stable/modules/generated/sklearn.metrics.fbeta_score.html">F-beta</a> меру с уклоном на полноту (beta=2, например)

# ## Дерево решений

# In[120]:


params =  {'max_depth': range (1,20, 2),
           'min_samples_leaf': range (1,18),
           'min_samples_split': range (2,20,2)}
           
model = DecisionTreeClassifier()

grid = RandomizedSearchCV(model,
    param_distributions=params, 
    scoring='precision', 
    n_jobs=-1, 
    random_state=12345
)

grid.fit(X_train, y_train)

print(grid.best_params_)


# <div class="alert alert-warning">
# <font size="4"><b>⚠️ Комментарий ревьюера</b></font>
#     <br /> 
#     <font size="3", color = "black">
#         <b></b>
# 
# `scoring='precision'`
# 
# Очень небесспорное решение. Смотри, если у нас будет низкая полнота, но высокая точность, то мы в теории будем плохих водителей допускать до поездок. Если же высокая полнота и низкая точность, то будем в теории хороших недопускать. Первое чревато травмами и увечиями людей, второе - потерей лояльности клиентов. Кажется, всё-таки важнее первое.

# In[121]:


precision_dt, predictions_dt, probabilities_dt, model_dt  = metrics_func(DecisionTreeClassifier(max_depth=7, 
                                                                                             min_samples_split=14, 
                                                                                             min_samples_leaf=10, 
                                                                                             random_state=12345,
                                                                                             class_weight='balanced'))


# <div class="alert alert-warning">
# <font size="4"><b>⚠️ Комментарий ревьюера</b></font>
#     <br /> 
#     <font size="3", color = "black">
#         <b></b>
# 
# `DecisionTreeClassifier(max_depth=7, min_samples_split=14, 
#                         min_samples_leaf=10, random_state=12345,
#                         class_weight='balanced')`
# 
# Одно из золотых правил разработки - не надо <a href="https://ru.hexlet.io/blog/posts/ponimaem-sleng-programmistov-mini-slovar-dlya-nachinayuschih-razrabotchikov#:~:text=%D0%A5%D0%B0%D1%80%D0%B4%D0%BA%D0%BE%D0%B4%D0%B8%D1%82%D1%8C%20%E2%80%94%20%D1%81%D1%82%D0%B0%D1%82%D0%B8%D1%87%D0%B5%D1%81%D0%BA%D0%B8%20%D0%BF%D1%80%D0%BE%D0%BF%D0%B8%D1%81%D1%8B%D0%B2%D0%B0%D1%82%D1%8C%20%D0%B2%20%D0%BA%D0%BE%D0%B4%D0%B5%20%D0%B4%D0%B0%D0%BD%D0%BD%D1%8B%D0%B5%2C%20%D0%BA%D0%BE%D1%82%D0%BE%D1%80%D1%8B%D0%B5%20%D0%B4%D0%BE%D0%BB%D0%B6%D0%BD%D1%8B%20%D0%B2%D1%8B%D1%87%D0%B8%D1%81%D0%BB%D1%8F%D1%82%D1%8C%D1%81%D1%8F%20%D0%B4%D0%B8%D0%BD%D0%B0%D0%BC%D0%B8%D1%87%D0%B5%D1%81%D0%BA%D0%B8.">хардкодить</a> то, что можно не хардкодить.
# 
# Почему это плохо на твоём примере. Ты нашёл какие-то лучшие гиперпараметры, потом ручками их вписал перед тем как тестировать модели. А вот ревьюер взял и поменял количество итераций рандомизированного поиска гиперпараметров, чтобы код работал быстрее, нашлись другие. И у него на экране сначла горит надпись что "найденные лучшие параметры - такие", а парой блоков кода ниже студент использует другие. Понадобилось поменять рандом стейт - та же история, нашли одни, а используем другие. А уж если обновился датасет... И получается, тебе надо постоянно следить за тем, чтобы вписанные гиперпараметры соответствовали найденным как оптимальные. Ну такое. Говоря проф языком, такой код сложно поддерживать, поэтому хардкод и считается антипаттерном.
# 
# Что же с этим делать?
# 
# После того как грид серч или рандомайзд серч закончили поиск, модель, обученная на оптимальных гиперпараметрах на полном датасете сохраняется в атрибут `.best_estimator_`, так что ничего заново обучать вообще не нужно, берёшь и используешь этот самый `grid.best_estimator_`

# ## Случайный лес
# 

# In[61]:


params =  {'n_estimators': range(10, 110, 15),
           'max_depth': range (1,15),
           'min_samples_leaf': range (1,8),
           'min_samples_split': range (2,10,2)}
           
model = RandomForestClassifier()

grid = RandomizedSearchCV(model,
    param_distributions=params, 
    scoring='precision', 
    n_jobs=-1, 
    random_state=12345
)

grid.fit(X_train, y_train)

print(grid.best_params_)


# In[62]:


precision_rf, predictions_rf, probabilities_rf, model_rf = metrics_func(RandomForestClassifier(n_estimators=85, 
                                                                                 max_depth=8, 
                                                                                 min_samples_split=8, 
                                                                                 min_samples_leaf=1, 
                                                                                 random_state=12345,
                                                                                 class_weight='balanced'))


# ## LightGBM

# In[125]:


params =  {'n_estimators': range(10, 110, 15),
           'max_depth': range (1,15),
           'num_leaves': range (2,80, 10)}
           
model = LGBMClassifier()

grid_GBM = RandomizedSearchCV(model,
    param_distributions=params, 
    scoring='precision', 
    n_jobs=-1, 
    random_state=12345
)

grid.fit(X_train, y_train)

print(grid.best_params_)


# In[126]:


precision_lgbm, predictions_lgbm, probabilities_lgbm, model_lgbm = metrics_func(LGBMClassifier(boosting_type='gbdt',
                                                                                            random_state=12345, 
                                                                                            max_depth=7,
                                                                                            n_estimators=10,
                                                                                            learning_rate = 0.1,
                                                                                            num_leaves = 32,
                                                                                            class_weight='balanced'))


# ## Catboost

# In[48]:


params =  {'iterations': range(100,2500, 500),
           'depth': range (1,15)}
           
model = CatBoostClassifier()

grid = RandomizedSearchCV(model,
    param_distributions=params, 
    scoring='precision', 
    n_jobs=-1, 
    random_state=12345
)

grid.fit(X_train, y_train)

print(grid.best_params_)


# In[ ]:





# In[49]:


precision_cb, predictions_cb, probabilities_cb, model_cb = metrics_func(CatBoostClassifier(depth=5,
                                                                                        iterations=1100,
                                                                                        random_seed=60,
                                                                                        learning_rate=0.003))


# Проверим адекватность моделей на пустышке 

# In[50]:


precision_dumm, predictions_dumm, probabilities_dumm, model_dumm  = metrics_func(DummyClassifier(strategy='most_frequent'))


# In[91]:


X_train.columns
X_test.columns


# In[ ]:





# Оформим результат в виде таблицы:

# In[75]:


result = pd.DataFrame ([
    [precision_dt],
    [precision_rf],
    [precision_lgbm],
    [precision_cb],
    [precision_dumm]], 
    columns=['precision'],
    index=['DecisionTree','RandomForest', 'LightGBM', 'CatBoost', 'Dummy'])
result.sort_values(by='precision', ascending=False)


# Лучший результат показала модель LightGBM, продолжим работу с ней.

# <div class="alert alert-success">
# <font size="4"><b>✔️ Комментарий ревьюера</b></font>
#     <br /> 
#     <font size="3", color = "black">
# <br />    
# Молодец, что для всех моделей из своей работы искал оптимальные гиперпараметры, и что дамми-модель в твоей работе есть, ты можешь точно сказать, что модели адекватные

# ## Анализ лучшей модели

# Матрица ошибок модели

# In[101]:


confusion_matrix(y_test, predictions_lgbm)


# <div class="alert alert-info">
# <font size="4">🍕<b> Комментарий ревьюера</b></font>
#     <br /> 
#     <font size="3", color = "black">
# <br />
# Приемлемо, но лучше выводить данную матрицу через <a href="https://scikit-learn.org/stable/modules/generated/sklearn.metrics.ConfusionMatrixDisplay.html">ConfusionMatrixDisplay</a>, так будет красивее

# Здесь, как и на матрицах для остальных моделей, мы видим высокое число ложно-положительных ответов, и при этом малое число ложно-отрицательных , что неплохо. Лучше ошибочно прездсказать аварийную ситуацию, чем ошибочно не предсказать.

# In[102]:


precision, recall, thresholds = precision_recall_curve(y_test, probabilities_lgbm)

fig, ax = plt.subplots()
ax.plot(recall, precision)

ax.set_title('Precision-Recall Curve')
ax.set_ylabel('Precision')
ax.set_xlabel('Recall')

plt.show()


# <div class="alert alert-success">
# <font size="4"><b>✔️ Комментарий ревьюера</b></font>
#     <br /> 
#     <font size="3", color = "black">
# <br />    
# О, а вот тут прям молодец! Большинство студентов строят кривую точности-полноты не по предсказанным вероятностям, а по предсказанным меткам классов, получают "ступеньку" и даже не задумываются, что что-то может быть не так :)
# 
# Молодец, что оказался одним из немногих, кто построил правильно сразу!

# Проанализируем важность основынх факторов, влияющих на вероятность ДТП

# In[103]:


smf = SelectFromModel(model_lgbm, threshold=-np.inf, max_features = 25)
smf.fit(X_train, y_train)
features_index = smf.get_support()
features_1 = X_train.columns[features_index]
features_1


# In[104]:


fig, ax = plt.subplots(figsize=(16,14))
df_tmp.corr().iloc[1].sort_values(ascending=False).drop('at_fault',axis = 0).plot.bar(ax=ax)
ax.set_title("Важность признаков")
ax.set_ylabel('Важность')
plt.grid()

fig.tight_layout()


# In[131]:


importances = model_lgbm.feature_importances_

feature_importances = list(zip(X_train.columns, importances))
feature_importances.sort(key=lambda x: x[1], reverse=True)

top_10_features = feature_importances[:10]

plt.figure(figsize=(10, 6))
plt.bar(range(len(top_10_features)), [x[1] for x in top_10_features], tick_label=[x[0] for x in top_10_features])
plt.xlabel('Признаки')
plt.ylabel('Важность')
plt.title('Топ 10 признаков с наибольшей важностью')
plt.xticks(rotation=90)
plt.show()

for feature, importance in top_10_features:
    print(f'Признак: {feature}, Важность: {importance}')


# In[138]:


plt.figure(figsize=[5,6])
sns.barplot(x=y_train, y=X_train['cellphone_in_use'])
plt.title('Зависимость фактора и целевой переменной')
plt.xlabel('Факт вины в ДТП')
plt.ylabel('Фактор')
plt.grid()
plt.show()


# Очевидно, что водитель используюший мобильное устройсто имеет больше шансов попасть в дтп. 
# Для борьбы с этим возможен вариант установки контролирующей видеокамеры с оповещением клиентов об этом. Или система, которая подает звуковой сигнал, если на руле нет двух рук водителя в течении 5-10 секунд, похожая система с контролем удерживания руля используется в Tesla

# ## Выводы

# In[ ]:


Наилучшей является модель модель CatBoostClassifier() при параметре n_estimators равному 1100, параметре depth равному 5 и параметре learning_rate равному 0,003 имеет наилучшую метрику precision, которая равняется 0,665 на трейне при кросс-валидации и 0.665 на тесте.
Создание адекватной системы оценки риска при выдаче автомобиля возможно, метрики наилучшей модели получились достаточно высокими, для лучшей оценки риска при выдаче автомобиля необходимо собирать больше данных.
Чтобы улучшить модель можно:
уточнить данные водителя: опыт вождения, пол, возраст, участие в ДТП в прошлом и т.д.
уточнить данные предполагаемого маршрута у водителя для диагностирования опасных дорожных участков
сигнализировать водителю о факторах, влияющих на условия вождения: плохие погодные условия, плохие дорожные условия, освещенность
поставаить дополнительное оборудование (камеры, анализаторы алкольного опьянения и т.д) для диагностирования состояния водителя


# In[ ]:




