import pandas as pd

# 1. טעינת הנתונים וסינון ראשוני
df = pd.read_csv('parsed_results_summary.csv')
df = df[df['num_experts'] == 3]
allowed_seeds = [4021, 4022, 4023, 4024, 4025]
df = df[df['seed'].isin(allowed_seeds)]
df = df[df['run_name'].str.contains('_ftM_', na=False)]

ds_map = {'national-illness': 'ILI', 'exchange-rate': 'Exchange'}
df['dataset'] = df['dataset'].replace(ds_map)
arch_map = {'MoG': 'MoGE'}
df['architecture'] = df['architecture'].replace(arch_map)

# 2. חישוב ממוצעים, סטיות תקן וספירת הרצות
results = {}
for name, group in df.groupby(['dataset', 'base_model', 'architecture', 'horizon']):
    ds, model, arch, horiz = name
    count = len(group)
    mse_mean = group['mse'].mean()
    mse_std = group['mse'].std() if count > 1 else 0.0
    mae_mean = group['mae'].mean()
    mae_std = group['mae'].std() if count > 1 else 0.0
    
    results[(ds, horiz, model, arch)] = {
        'mse_val': mse_mean, 'mae_val': mae_mean,
        'mse_str': f"{mse_mean:.4f} ± {mse_std:.4f}" if count > 1 else f"{mse_mean:.4f}",
        'mae_str': f"{mae_mean:.4f} ± {mae_std:.4f}" if count > 1 else f"{mae_mean:.4f}",
        'count': count
    }

# 3. יצירת המבנה ההיררכי של הטבלה (MultiIndex)
models = ['iTransformer', 'PatchTST', 'DLinear', 'TimeMixer']
archs = ['MoE', 'MoGE', 'MoGU']
metrics = ['MAE', 'MSE']

col_tuples = [(m, a, met) for m in models for a in archs for met in metrics]
columns = pd.MultiIndex.from_tuples(col_tuples, names=['Expert', 'Mixture Type', 'Metric'])

datasets = ['ETTh1', 'ETTh2', 'ETTm1', 'ETTm2', 'ILI', 'Exchange']
horizons_std = [96, 192, 336, 720]
horizons_ili = [24, 36, 48, 60]
row_tuples = []
for ds in datasets:
    hzs = horizons_ili if ds == 'ILI' else horizons_std
    for hz in hzs:
        row_tuples.append((ds, hz))
index = pd.MultiIndex.from_tuples(row_tuples, names=['Dataset', 'Horizon'])

out_df = pd.DataFrame(index=index, columns=columns)
color_df = pd.DataFrame('', index=index, columns=columns)
weight_df = pd.DataFrame('', index=index, columns=columns)

# 4. מילוי הנתונים והחלת הסינון המחמיר
for ds, hz in row_tuples:
    for m in models:
        # בדיקה האם לכל 3 הארכיטקטורות יש לפחות 2 הרצות
        valid_comparison = True
        for a in archs:
            if (ds, hz, m, a) not in results or results[(ds, hz, m, a)]['count'] < 2:
                valid_comparison = False
                break
                
        best_mse = float('inf')
        best_mae = float('inf')
        
        # מציאת המנצח (רק אם ההשוואה חוקית)
        if valid_comparison:
            for a in archs:
                if results[(ds, hz, m, a)]['mse_val'] < best_mse: best_mse = results[(ds, hz, m, a)]['mse_val']
                if results[(ds, hz, m, a)]['mae_val'] < best_mae: best_mae = results[(ds, hz, m, a)]['mae_val']
                
        for a in archs:
            if (ds, hz, m, a) in results:
                data = results[(ds, hz, m, a)]
                out_df.loc[(ds, hz), (m, a, 'MAE')] = data['mae_str']
                out_df.loc[(ds, hz), (m, a, 'MSE')] = data['mse_str']
                
                # הדגשה בבולד (Bold) - רק תחת הסינון המחמיר
                if valid_comparison:
                    if abs(data['mae_val'] - best_mae) < 1e-6:
                        weight_df.loc[(ds, hz), (m, a, 'MAE')] = 'font-weight: bold;'
                    if abs(data['mse_val'] - best_mse) < 1e-6:
                        weight_df.loc[(ds, hz), (m, a, 'MSE')] = 'font-weight: bold;'
                    
                # צביעת רקע לפי סטטוס הרצות
                if data['count'] == 5:
                    color_df.loc[(ds, hz), (m, a, 'MAE')] = 'background-color: #C6EFCE; color: #006100;'
                    color_df.loc[(ds, hz), (m, a, 'MSE')] = 'background-color: #C6EFCE; color: #006100;'
                elif 2 <= data['count'] <= 4:
                    color_df.loc[(ds, hz), (m, a, 'MAE')] = 'background-color: #FFEB9C; color: #9C5700;'
                    color_df.loc[(ds, hz), (m, a, 'MSE')] = 'background-color: #FFEB9C; color: #9C5700;'
            else:
                out_df.loc[(ds, hz), (m, a, 'MAE')] = ""
                out_df.loc[(ds, hz), (m, a, 'MSE')] = ""

# 5. עיצוב וייצוא לאקסל
def style_func(x):
    return color_df + weight_df

styler = out_df.style.apply(style_func, axis=None)
styler.set_properties(**{'text-align': 'center'})

output_filename = 'MoGU_Main_Results_Strict.xlsx'
styler.to_excel(output_filename, engine='openpyxl')
print(f"File successfully created: {output_filename}")