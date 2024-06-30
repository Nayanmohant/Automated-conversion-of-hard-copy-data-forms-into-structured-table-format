import pandas as pd

values = ["rahul", "sachin", "virat", "gambhir", "kajal", "samantha", "ash", "sanju", "nayan"]

# Reshape the list into a 3-column structure
num_columns = 3
table_data = [values[i:i+num_columns] for i in range(0, len(values), num_columns)]

# Create DataFrame from the table data
df = pd.DataFrame(table_data, columns=["Column 1", "Column 2", "Column 3"])

# Print the DataFrame
#print(df)

df.to_csv('form_data.csv', index=False)
