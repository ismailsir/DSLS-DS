from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
import sys
import pickle

# Input Data #2
print("## Bandung City Jam Prediction ##")

# Input day with Condition
day = int(input("Input day code: "))
if day <= 4 and day >= 0:
    day = 0
elif day <=6 and day >= 0:
    day = 1
else :
    sys.exit("The code is not in the list")

# Input street with Condition
street = int(input("Input street code: "))
if street < 0 or street > 123:
    sys.exit("The code is not in the list")

# Input hour with Condition
hour = int(input("Input Hour: "))
if hour >= 1 and hour <= 4:
    hour = 0
elif hour >= 5 and hour <= 10:
    hour = 1
elif hour >= 11 and hour <= 13:
    hour = 2
elif hour >= 14 and hour <= 16:
    hour = 3
elif (hour >=17  and hour <= 10) or (hour == 0):
    hour = 4
else :
    sys.exit("The code is not in the list")


# Prediction #
# Load Model
with open(r'Mini DS\Model_level.pkl', 'rb') as file:
    model1 = pickle.load(file)
with open(r'Mini DS\Model_speed.pkl', 'rb') as file:
    model2 = pickle.load(file)

# Predict
new_input = [[hour, street, day]]  # example input values2
prediction1 = model1.predict(new_input)
prediction2 = model2.predict(new_input)

# Result
print("Prediciton Result")
print("Level : ", prediction1[0])
print("Speed : ", prediction2[0])
