import pandas as pd
from sklearn.preprocessing import LabelEncoder
import numpy as np

# Input Dataset
df = pd.read_csv(r'Mini DS\Dataset\aggregate_median_jams_Kota Bandung_fixed.csv')

# Remove unused columns
df = df.drop(['Unnamed: 0', 'kemendagri_kabupaten_kode', 'kemendagri_kabupaten_nama', 'id', 'median_level', 'geometry'], axis=1)

# Changing data form to "time", "hour", and "day"
df['date'] = pd.to_datetime(df['date'], format='%Y-%m-%d')
df['time'] = pd.to_datetime(df['time'])
df['time'] = df['time'].dt.time
df['time'] = pd.to_datetime(df['time'], format='%H:%M:%S')
df['hour'] = df['time'].dt.hour
df['day'] = df['date'].dt.weekday

# Create condition of every hour
## (1-4)        --> Early Morning   (0)
## (5-10)       --> Morning         (1)
## (11-13)      --> Noon            (2)
## (14-16)      --> Afternoon       (3)
## (17-23 & 0)  --> Night           (4)
conditions_hour = [
    (df['hour'] >= 1) & (df['hour'] <= 4),
    (df['hour'] >= 5) & (df['hour'] <= 10),
    (df['hour'] >= 11) & (df['hour'] <= 13),
    (df['hour'] >= 14) & (df['hour'] <= 16),
    ((df['hour'] >= 17) & (df['hour'] <= 23)) | ((df['hour'] == 0)) 
]
outputs_hour = [0, 1, 2, 3, 4]
df['hour_group'] = np.select(conditions_hour, outputs_hour)

# Create conditions from day
## (0-4) --> Weekdays (0)
## (5-6) --> Holidays (1)
conditions_day = [
    (df['day'] >= 0) & (df['day'] <= 4),
    (df['day'] >= 5) & (df['day'] <= 6)
]
outputs_day = [0, 1]
df['holiday'] = np.select(conditions_day, outputs_day)

# Filter data with 20% of the most data
value_counts = df['street'].value_counts()
threshold = 0.2  
max_count = value_counts.max()
min_count = threshold * max_count
names_to_keep = value_counts[value_counts > min_count].index
filtered_df = df[df['street'].isin(names_to_keep)]
filtered_df = filtered_df.drop(filtered_df[filtered_df['level'] == 5].index)

# Street name encoding
le = LabelEncoder()
filtered_df['street_en'] = le.fit_transform(filtered_df['street'])

# List of street names with encoding results
unique_names = filtered_df[['street', 'street_en']].drop_duplicates().sort_values('street_en')
print(unique_names)
unique_names.to_excel(r'Mini DS\unique_names.xlsx', index=False)

# Machine Learning #
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, mean_squared_error
import pickle

# Training Level
X = filtered_df[['hour_group','street_en','holiday']]
y = filtered_df['level']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model1 = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
model1.fit(X_train, y_train)
y_pred = model1.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model accuracy: {accuracy}")
with open(r'Mini DS\Model_level.pkl', 'wb') as file:
    pickle.dump(model1, file)

# Training Speed
y = filtered_df['median_speed_kmh']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model2 = RandomForestRegressor()
model2.fit(X_train, y_train)
y_pred = model2.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print("Mean squared error: ", mse)
with open(r'Mini DS\Model_speed.pkl', 'wb') as file:
    pickle.dump(model2, file)
