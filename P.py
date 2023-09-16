import pandas as pd
from googletrans import Translator
import time

# Create a Translator object
translator = Translator()

# Translate the text to English
#translated_text = translator.translate(text_to_translate, src='fa', dest='en')


# Replace 'filename.csv' with the path to your CSV file
data_frame = pd.read_csv('fa_perdt-ud-train.conllu', delimiter='\t', comment='#', header=None)

# Now you can work with the data in the DataFrame
print(data_frame.head())

num_rows, num_columns = data_frame.shape
# Print the number of rows and columns
print("Number of rows:", num_rows)
print("Number of columns:", num_columns)

rows_to_loop=range(num_rows)
columns_to_loop=[1,2]

#translate everything using gtranslate
for row_index in rows_to_loop:
    for column_name in columns_to_loop:
        # Modify the value at the specified row and column
        #print(data_frame.iloc[row_index, column_name])
        if (row_index%80==0):
            time.sleep(5) # Sleep for 3 seconds
        try:
            # Translate the text from Persian to English
            translated = translator.translate(str(data_frame.iloc[row_index, column_name]), src='fa', dest='en')

            # Update the DataFrame with the translated text
            data_frame.loc[row_index, column_name] = translated.text
        except (IndexError, AttributeError) as e:
            print(f"Error at row {row_index}, column {column_name}: {str(e)}")
