import pandas as pd
import numpy as np
fin_df = pd.read_csv('financial_data.csv')
prol_df = pd.read_csv('prolongations.csv')
prol_df_unique = prol_df.drop_duplicates(subset=['id'], keep='first')
df_merged = pd.merge(fin_df, prol_df_unique, on='id', how='left')
month_cols = [col for col in df_merged.columns if '2022' in col or '2023' in col or '2024' in col] 
# 3.2. Функция для очистки и преобразования денежных сумм
def clean_currency(value):
    if not isinstance(value, str):
        return value
    # Убираем невидимые пробелы, стандартные пробелы, меняем запятую на точку
    cleaned_value = value.replace('\xa0', '').replace(' ', '').replace(',', '.')
    try:
        return float(cleaned_value)
    except (ValueError, TypeError):
        # Возвращаем текстовые маркеры как есть (например, 'стоп', 'end', 'в ноль')
        return value.lower() # приводим к нижнему регистру для унификации

# 3.3. Применяем функцию очистки ко всем столбцам-месяцам
for col in month_cols:
    df_merged[col] = df_merged[col].apply(clean_currency)
print("-> Денежные суммы очищены и преобразованы в числа.")

# 3.4. Исключение проектов со "стопами"
# Приводим столбец 'month' к единому формату
df_merged['month'] = df_merged['month'].str.capitalize().str.replace(' ', ' ')

ids_to_exclude = set()
for _, row in df_merged.iterrows():
    last_month_project = row['month']
    if pd.isna(last_month_project):
        continue

    # Находим индекс последнего месяца проекта
    try:
        last_month_idx = month_cols.index(last_month_project)
    except ValueError:
        continue # Если месяц из prol_df не найден в колонках fin_df

    # Проверяем все месяцы до последнего (включительно) на наличие 'стоп' или 'end'
    for i in range(last_month_idx + 1):
        month_to_check = month_cols[i]
        if row[month_to_check] in ['стоп', 'end']:
            ids_to_exclude.add(row['id'])
            break

df_cleaned = df_merged[~df_merged['id'].isin(ids_to_exclude)].copy()
print(f"-> Исключено {len(ids_to_exclude)} проектов со 'стоп' или 'end'.")

# 3.5. Обработка значений "в ноль"
for i in range(len(month_cols) - 1, 0, -1): # Идем в обратном порядке, чтобы правильно обработать несколько "в ноль" подряд
    current_col = month_cols[i]
    prev_col = month_cols[i-1]
    
    mask = df_cleaned[current_col] == 'в ноль'
    # Заменяем 'в ноль' значением из предыдущего столбца, если оно не является текстом
    df_cleaned.loc[mask, current_col] = pd.to_numeric(df_cleaned.loc[mask, prev_col], errors='coerce')

# После всех замен, финально преобразуем все в числа. Оставшиеся тексты и ошибки станут 0.
for col in month_cols:
    df_cleaned[col] = pd.to_numeric(df_cleaned[col], errors='coerce').fillna(0)
print("-> Значения 'в ноль' обработаны.")

# 3.6. Агрегация данных по ID проекта
# Суммируем отгрузки для проектов, разбитых на несколько строк (например, "первая/вторая часть оплаты")
agg_functions = {col: 'sum' for col in month_cols}
agg_functions['AM'] = 'first' # Берем первого попавшегося менеджера
agg_functions['month'] = 'first' # Берем первый попавшийся месяц окончания

df_aggregated = df_cleaned.groupby('id').agg(agg_functions).reset_index()
print("-> Данные агрегированы по ID проекта.")

# 3.7. Очистка имен менеджеров и удаление "без А/М"
df_aggregated['AM'] = df_aggregated['AM'].str.strip()
df_aggregated = df_aggregated[df_aggregated['AM'] != 'без А/М']
print("-> Имена менеджеров очищены, проекты 'без А/М' удалены.")


# =============================================================================
# Блок 4: Трансформация данных для расчетов (Melt)
# =============================================================================

# "Плавим" широкую таблицу в длинную для удобства фильтрации
df_flat = df_aggregated.melt(
    id_vars=['id', 'AM', 'month'],
    value_vars=month_cols,
    var_name='отгрузка_месяц',
    value_name='сумма_отгрузки'
)
# Приводим месяц окончания к единому формату
df_flat['month'] = df_flat['month'].str.capitalize()
print("\nДанные преобразованы в 'длинный' формат для расчетов.")

# =============================================================================
# Блок 5: Расчет коэффициентов
# =============================================================================
print("\nНачало расчета коэффициентов...")

# Создаем удобные словари для работы с месяцами
month_map = {name: i for i, name in enumerate(month_cols)}
month_map_inv = {i: name for i, name in enumerate(month_cols)}
calc_months_2023 = [m for m in month_cols if '2023' in m]

results_data = []

# for month_name in calc_months_2023:
#     if month_map[month_name] < 2: # Пропускаем январь, т.к. для коэфф.2 нужен ноябрь 2022
#         continue

#     # --- ОПРЕДЕЛЯЕМ МЕСЯЦЫ ---
#     current_month = month_name                             # Пример: 'Март 2023'
#     prev_month_1 = month_map_inv[month_map[current_month] - 1] # 'Февраль 2023'
#     prev_month_2 = month_map_inv[month_map[current_month] - 2] # 'Январь 2023'

def calculate_coeffs(df, grouping_key=None):
    """Функция для расчета числителей и знаменателей, с группировкой или без."""
    # Коэффициент 1
    ended_prev_1 = df[df['month'] == prev_month_1]
    den1 = ended_prev_1.groupby(grouping_key)[prev_month_1].sum() if grouping_key else ended_prev_1[prev_month_1].sum()
    num1 = ended_prev_1.groupby(grouping_key)[current_month].sum() if grouping_key else ended_prev_1[current_month].sum()
    
    # Коэффициент 2
    ended_prev_2 = df[df['month'] == prev_month_2]
    not_prolonged_in_1 = ended_prev_2[ended_prev_2[prev_month_1] == 0]
    den2 = not_prolonged_in_1.groupby(grouping_key)[prev_month_2].sum() if grouping_key else not_prolonged_in_1[prev_month_2].sum()
    num2 = not_prolonged_in_1.groupby(grouping_key)[current_month].sum() if grouping_key else not_prolonged_in_1[current_month].sum()

    return den1, num1, den2, num2

# --- Расчет по месяцам ---
monthly_numerators = []
monthly_denominators = []

for month_name in calc_months_2023:
    if month_map.get(month_name, 0) < 2: continue # Пропускаем первые два месяца

    current_month = month_name
    prev_month_1 = month_map_inv[month_map[current_month] - 1]
    prev_month_2 = month_map_inv[month_map[current_month] - 2]

    # Расчет по менеджерам
    den1_am, num1_am, den2_am, num2_am = calculate_coeffs(df_aggregated, 'AM')
    
    # Расчет по всему отделу
    den1_total, num1_total, den2_total, num2_total = calculate_coeffs(df_aggregated)

    # Сохраняем числители и знаменатели для годового отчета
    monthly_numerators.append({'month': current_month, 'num1_am': num1_am, 'num2_am': num2_am, 'num1_total': num1_total, 'num2_total': num2_total})
    monthly_denominators.append({'month': current_month, 'den1_am': den1_am, 'den2_am': den2_am, 'den1_total': den1_total, 'den2_total': den2_total})

    # Собираем месячные результаты
    res_df1 = (num1_am / den1_am).fillna(0).rename('coeff_1')
    res_df2 = (num2_am / den2_am).fillna(0).rename('coeff_2')
    
    res_df_total1 = (num1_total / den1_total) if den1_total > 0 else 0
    res_df_total2 = (num2_total / den2_total) if den2_total > 0 else 0
    
    # Преобразуем в датафрейм
    month_res = pd.merge(res_df1, res_df2, left_index=True, right_index=True, how='outer').fillna(0).reset_index()
    month_res.loc['Total'] = ['Всего по отделу', res_df_total1, res_df_total2]
    month_res['month'] = current_month
    results_data.append(month_res)

# Итоговая таблица с месячными коэффициентами
final_monthly_report = pd.concat(results_data, ignore_index=True)

# --- Расчет годовых итогов ---
total_num1_am = sum(d['num1_am'] for d in monthly_numerators)
total_den1_am = sum(d['den1_am'] for d in monthly_denominators)
total_num2_am = sum(d['num2_am'] for d in monthly_numerators)
total_den2_am = sum(d['den2_am'] for d in monthly_denominators)

yearly_coeff1 = (total_num1_am / total_den1_am).fillna(0)
yearly_coeff2 = (total_num2_am / total_den2_am).fillna(0)

# По всему отделу за год
total_num1_total = sum(d['num1_total'] for d in monthly_numerators)
total_den1_total = sum(d['den1_total'] for d in monthly_denominators)
total_num2_total = sum(d['num2_total'] for d in monthly_numerators)
total_den2_total = sum(d['den2_total'] for d in monthly_denominators)

yearly_coeff1_total = (total_num1_total / total_den1_total) if total_den1_total > 0 else 0
yearly_coeff2_total = (total_num2_total / total_den2_total) if total_den2_total > 0 else 0

# Собираем годовой отчет
final_yearly_report = pd.merge(
    yearly_coeff1.rename('Годовой Коэфф. 1'),
    yearly_coeff2.rename('Годовой Коэфф. 2'),
    left_index=True, right_index=True, how='outer'
).fillna(0).reset_index()
final_yearly_report.loc['Total'] = ['Всего по отделу', yearly_coeff1_total, yearly_coeff2_total]


print("-> Расчеты завершены.")

# =============================================================================
# Блок 6: Формирование Excel-отчета
# =============================================================================

# Создаем сводные таблицы для месячных отчетов
pivot_monthly_c1 = final_monthly_report.pivot_table(index='AM', columns='month', values='coeff_1', aggfunc='sum')
pivot_monthly_c2 = final_monthly_report.pivot_table(index='AM', columns='month', values='coeff_2', aggfunc='sum')

# Записываем все в один Excel файл с разными листами
output_filename = 'Аналитический_отчет_по_пролонгациям_2023.xlsx'
with pd.ExcelWriter(output_filename, engine='xlsxwriter') as writer:
    final_yearly_report.to_excel(writer, sheet_name='Годовые итоги', index=False)
    pivot_monthly_c1.to_excel(writer, sheet_name='Коэфф_1_по_месяцам')
    pivot_monthly_c2.to_excel(writer, sheet_name='Коэфф_2_по_месяцам')
    df_aggregated.to_excel(writer, sheet_name='Очищенные_агрегированные_данные', index=False)

print(f"\nОтчет успешно сформирован и сохранен в файл: {output_filename}")
print("\nГодовые итоги:")
print(final_yearly_report.to_string(index=False))