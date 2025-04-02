import pandas as pd
import scipy.stats as stats



# import efficiencies
df = pd.read_csv("New_measurement.csv")

# Groepeer de 'instruction efficiency' per configuratie (Partial)
groups_instruction_efficiency = df.groupby("Partial")["instruction efficiency"].apply(list)
print(groups_instruction_efficiency)
# Voer een eenweg-ANOVA uit voor instruction efficiency
f_val, p_val = stats.f_oneway(*groups_instruction_efficiency)
print("ANOVA voor instruction efficiency: F =", f_val, "p =", p_val)




# Eveneens voor 'value efficiency':
groups_value_efficiency = df.groupby("Partial")["value efficiency"].apply(list)
print(groups_value_efficiency)
f_val_val, p_val_val = stats.f_oneway(*groups_value_efficiency)
print("ANOVA voor value efficiency: F =", f_val_val, "p =", p_val_val)


#dd