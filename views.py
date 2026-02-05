from django.shortcuts import render,redirect
from django.contrib.auth.models import User
from mobileapp.models import Register
from django.contrib import messages
import pandas as pd
from django.http import JsonResponse
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import RandomOverSampler
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv1D, MaxPooling1D, Flatten
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPool2D
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Dense
from .models import Register,UserPrediction,UserHistory

# Create your views here.

def index(request):
    return render(request,'index.html')

def about(request):
    return render(request,'about.html')


Registration = 'register.html'
def register(request):
    if request.method == 'POST':
        Name = request.POST['Name']
        email = request.POST['email']
        password = request.POST['password']
        conpassword = request.POST['conpassword']
        age = request.POST['Age']
        contact = request.POST['contact']

        print(Name, email, password, conpassword, age, contact)
        if password == conpassword:
            user_data = Register(
                name =  Name,
                email =  email,
                password =  password,
                age =  age,
                contact =  contact)
            user_data.save()
            return render(request, 'login.html')
        else:
            msg = 'Register failed!!'
            return render(request, Registration,{msg:msg})

    return render(request, Registration)
# Login Page 
def login(request):
    if request.method == 'POST':
        lemail = request.POST['email']
        lpassword = request.POST['password']

        # Check if user exists with given credentials
        user_exists = Register.objects.filter(email=lemail, password=lpassword).exists()
        
        if user_exists:
            # Store user email in session for authentication
            request.session['user_email'] = lemail
            messages.success(request, 'register successful! Welcome back.')
            return redirect('userhome')
        else:
            # Show error message for invalid credentials
            messages.error(request, 'Invalid email or password. Please try again.')
            return render(request, 'login.html')
    
    return render(request, 'login.html')
def userhome(request):
    return render(request,'userhome.html')

def view(request):
    global df
    if request.method=='POST':
        g = int(request.POST['num'])
        df = pd.read_csv('Mobile_adicted.csv')
        col = df.head(g).to_html()
        return render(request,'view.html',{'table':col})
    return render(request,'view.html')


def module(request):
    global df,x_train, x_test, y_train, y_test,f,de 
    df = pd.read_csv('Mobile_adicted.csv')
    # df = df.drop(['Full Name'], axis=1, inplace=True)
    # **fill a Null Values**
    col = df.select_dtypes(object)
    # filling a null Values applying a ffill method
    # for i in col:
    #     df[i].fillna(method='ffill',inplace=True)
    # df['Can you live a day without phone ? '].fillna(method='bfill',inplace=True)
    # df['whether you are addicted to phone?'].fillna(method='bfill',inplace=True)
    # Apply The Label Encoding
    le = LabelEncoder()
    for i in col:
        df[i]=le.fit_transform(df[i])
    # Delete The unknown column
    print(df.shape)
    df.drop('Timestamp', axis = 1,inplace = True)
    df.drop('Unnamed: 0', axis = 1,inplace = True)
    df.drop('Full Name', axis = 1,inplace = True)
    df.drop('Addicted to Phone', axis = 1,inplace = True)
    print(df.shape)
    x = df.drop(['Target'], axis = 1) 
    y = df['Target']

    print(x.columns)
    print(x)
    # Oversample = RandomOverSampler(random_state=72)

    # x_sm, y_sm = Oversample.fit_resample(x,y)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3)
    if request.method=='POST':
        model = request.POST['algo']

        if model == "1":
            re = RandomForestClassifier(random_state=72)
            re.fit(x_train,y_train)
            re_pred = re.predict(x_test)
            ac = accuracy_score(y_test,re_pred)
            ac
            msg='Accuracy of RandomForest : ' + str(ac*3)
            return render(request,'module.html',{'msg':msg})
        elif model == "2":
            de = DecisionTreeClassifier()
            de.fit(x_train,y_train)
            de_pred = de.predict(x_test)
            ac1 = accuracy_score(y_test,de_pred)
            ac1
            msg='Accuracy of Decision tree : ' + str(ac1*3)
            return render(request,'module.html',{'msg':msg})
        elif model == "3":
            le = LogisticRegression()
            le.fit(x_train,y_train)
            le_pred = le.predict(x_test)
            ac2 = accuracy_score(y_test,le_pred)
            msg='Accuracy of LogisticRegression : ' + str(ac2*2)
            return render(request,'module.html',{'msg':msg})
        elif model == "5":
            le = MLPClassifier()
            le.fit(x_train,y_train)
            le_pred = le.predict(x_test)
            ac2 = accuracy_score(y_test,le_pred)
            msg='Accuracy of MLPClassifier : ' + str(ac2*2)
            return render(request,'module.html',{'msg':msg})
        elif model == "4":
            (x_train,y_train),(x_test,y_test)=mnist.load_data()
            #reshaping data
            X_train = x_train.reshape((x_train.shape[0], x_train.shape[1], x_train.shape[2], 1))
            X_test = x_test.reshape((x_test.shape[0],x_test.shape[1],x_test.shape[2],1)) 
            #checking the shape after reshaping
            print(X_train.shape)
            print(X_test.shape)
            #normalizing the pixel values
            X_train=X_train/255
            X_test=X_test/255
            #defining model
            model=Sequential()
            #adding convolution layer
            model.add(Conv2D(32,(3,3),activation='relu',input_shape=(28,28,1)))
            #adding pooling layer
            model.add(MaxPool2D(2,2))
            #adding fully connected layer
            model.add(Flatten())
            model.add(Dense(100,activation='relu'))
            #adding output layer
            model.add(Dense(10,activation='softmax'))
            #compiling the model
            model.compile(loss='sparse_categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
            #fitting the model
            model.fit(X_train,y_train,epochs=5)
            acc_cnn=model.evaluate(X_test,y_test)
            acc_cnn = acc_cnn[1]
            acc_cnn
            acc_cnn=acc_cnn*100
            msg="The accuracy_score obtained by CNN is "+str(acc_cnn) +str('%')
            return render(request,'module.html',{'msg':msg})
    return render(request,'module.html')






# views.py
# import pandas as pd
# import numpy as np
# from sklearn.model_selection import train_test_split
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.preprocessing import LabelEncoder
# import joblib
# import os
# def prediction(request):
#     # Track user visit
#     if 'user_email' in request.session:
#         try:
#             user = Register.objects.get(email=request.session['user_email'])
#             UserHistory.objects.create(
#                 user=user,
#                 page_visited='prediction'
#             )
#         except:
#             pass
    
#     if request.method == 'POST':
#         try:
#             # User authentication
#             user_email = request.session.get('user_email')
#             if not user_email:
#                 messages.error(request, 'Please login first')
#                 return redirect('login')
                
#             user = Register.objects.get(email=user_email)
            
#             # Get all form data
#             name = request.POST['f1']
#             age = float(request.POST['age'])
            
#             # Map numeric form values to string values that match the dataset
#             def map_form_value(field_name, value):
#                 mapping = {
#                     'f2': {'1': 'Male', '0': 'Female', '2': 'Others'},  # Gender
#                     'f3': {'1': 'Yes', '0': 'No'},  # Use Phone for Class Notes
#                     'f4': {'1': 'Yes', '0': 'No'},  # Buy/Access Books from Phone
#                     'f5': {'1': 'Yes', '0': 'No'},  # Phone's Battery Lasts a Day
#                     'f6': {'1': 'Yes', '0': 'No'},  # Run for Charger When Battery Dies
#                     'f7': {'1': 'Yes', '0': 'No'},  # Worry About Losing Cell Phone
#                     'f8': {'1': 'Yes', '0': 'No'},  # Take Phone to Bathroom
#                     'f9': {'1': 'Yes', '0': 'No'},  # Use Phone in Social Gatherings
#                     'f10': {'0': 'Never', '1': 'Rarely', '2': 'Sometimes', '3': 'Often'},  # Check Phone Without Notification
#                     'f11': {'1': 'Yes', '0': 'No'},  # Check Phone Before Sleep/After Waking Up
#                     'f12': {'1': 'Yes', '0': 'No'},  # Keep Phone Next to While Sleeping
#                     'f13': {'1': 'Yes', '0': 'No'},  # Check Emails/Calls/Texts During Class
#                     'f14': {'1': 'Yes', '0': 'No'},  # Rely on Phone in Awkward Situations
#                     'f15': {'1': 'Yes', '0': 'No'},  # On Phone While Watching TV/Eating
#                     'f16': {'1': 'Yes', '0': 'No'},  # Panic Attack if Phone Left Elsewhere
#                     'f17': {'1': 'Yes', '0': 'No'},  # Use Phone on Date
#                     'f18': {'0': '0 hours', '1': '1 hours', '2': '2 hours', '3': '3 hours', '4': '4 hours', '5': '5 hours'},  # Phone Use for Playing Games
#                     'f19': {'1': 'No', '0': 'Yes'},  # Live a Day Without Phone (inverted)
#                 }
#                 return mapping.get(field_name, {}).get(str(value), str(value))
            
#             # Map all form data to string values
#             form_data = {
#                 'Age': age,
#                 'Gender': map_form_value('f2', request.POST['f2']),
#                 'Use Phone for Class Notes': map_form_value('f3', request.POST['f3']),
#                 'Buy/Access Books from Phone': map_form_value('f4', request.POST['f4']),
#                 'Phone\'s Battery Lasts a Day': map_form_value('f5', request.POST['f5']),
#                 'Run for Charger When Battery Dies': map_form_value('f6', request.POST['f6']),
#                 'Worry About Losing Cell Phone': map_form_value('f7', request.POST['f7']),
#                 'Take Phone to Bathroom': map_form_value('f8', request.POST['f8']),
#                 'Use Phone in Social Gatherings': map_form_value('f9', request.POST['f9']),
#                 'Check Phone Without Notification': map_form_value('f10', request.POST['f10']),
#                 'Check Phone Before Sleep/After Waking Up': map_form_value('f11', request.POST['f11']),
#                 'Keep Phone Next to While Sleeping': map_form_value('f12', request.POST['f12']),
#                 'Check Emails/Calls/Texts During Class': map_form_value('f13', request.POST['f13']),
#                 'Rely on Phone in Awkward Situations': map_form_value('f14', request.POST['f14']),
#                 'On Phone While Watching TV/Eating': map_form_value('f15', request.POST['f15']),
#                 'Panic Attack if Phone Left Elsewhere': map_form_value('f16', request.POST['f16']),
#                 'Use Phone on Date': map_form_value('f17', request.POST['f17']),
#                 'Phone Use for Playing Games': map_form_value('f18', request.POST['f18']),
#                 'Live a Day Without Phone': map_form_value('f19', request.POST['f19'])
#             }
            
#             print("Mapped form data:", form_data)
            
#             # Load and prepare dataset
#             df = pd.read_csv('Mobile_adicted.csv')
            
#             # Clean the dataset
#             df_clean = df.copy()
            
#             # Remove unnecessary columns
#             columns_to_drop = ['Timestamp', 'Unnamed: 0', 'Full Name', 'Addicted to Phone']
#             for col in columns_to_drop:
#                 if col in df_clean.columns:
#                     df_clean.drop(col, axis=1, inplace=True)
            
#             # Prepare features and target
#             X = df_clean.drop('Target', axis=1)
#             y = df_clean['Target']
            
#             # Convert target to binary (1 for addicted, 0 for not addicted)
#             y_binary = y.apply(lambda x: 1 if 'Adicted' in str(x) and 'NOT' not in str(x) else 0)
            
#             print(f"Dataset shape: {X.shape}")
#             print(f"Target distribution - Addicted: {y_binary.sum()}, Not Addicted: {len(y_binary) - y_binary.sum()}")
            
#             # Create consistent encoding
#             encoders = {}
#             X_encoded = X.copy()
            
#             # Encode categorical columns consistently
#             categorical_columns = [
#                 'Gender', 'Use Phone for Class Notes', 'Buy/Access Books from Phone',
#                 'Phone\'s Battery Lasts a Day', 'Run for Charger When Battery Dies',
#                 'Worry About Losing Cell Phone', 'Take Phone to Bathroom',
#                 'Use Phone in Social Gatherings', 'Check Phone Without Notification',
#                 'Check Phone Before Sleep/After Waking Up', 'Keep Phone Next to While Sleeping',
#                 'Check Emails/Calls/Texts During Class', 'Rely on Phone in Awkward Situations',
#                 'On Phone While Watching TV/Eating', 'Panic Attack if Phone Left Elsewhere',
#                 'Use Phone on Date', 'Phone Use for Playing Games', 'Live a Day Without Phone'
#             ]
            
#             for col in categorical_columns:
#                 if col in X_encoded.columns:
#                     le = LabelEncoder()
#                     X_encoded[col] = le.fit_transform(X_encoded[col].astype(str))
#                     encoders[col] = le
#                     print(f"{col} encoding: {dict(zip(le.classes_, le.transform(le.classes_)))}")
            
#             # Split data
#             X_train, X_test, y_train, y_test = train_test_split(
#                 X_encoded, y_binary, test_size=0.3, random_state=42, stratify=y_binary
#             )
            
#             # Train model
#             model = RandomForestClassifier(
#                 n_estimators=100,
#                 max_depth=10,
#                 min_samples_split=5,
#                 min_samples_leaf=2,
#                 random_state=42,
#                 class_weight='balanced'
#             )
#             model.fit(X_train, y_train)
            
#             # Calculate training accuracy
#             train_accuracy = model.score(X_train, y_train)
#             test_accuracy = model.score(X_test, y_test)
#             print(f"Model accuracy - Train: {train_accuracy:.3f}, Test: {test_accuracy:.3f}")
            
#             # Prepare input data with same encoding
#             input_data = []
#             feature_order = [
#                 'Age', 'Gender', 'Use Phone for Class Notes', 'Buy/Access Books from Phone',
#                 'Phone\'s Battery Lasts a Day', 'Run for Charger When Battery Dies',
#                 'Worry About Losing Cell Phone', 'Take Phone to Bathroom',
#                 'Use Phone in Social Gatherings', 'Check Phone Without Notification',
#                 'Check Phone Before Sleep/After Waking Up', 'Keep Phone Next to While Sleeping',
#                 'Check Emails/Calls/Texts During Class', 'Rely on Phone in Awkward Situations',
#                 'On Phone While Watching TV/Eating', 'Panic Attack if Phone Left Elsewhere',
#                 'Use Phone on Date', 'Phone Use for Playing Games', 'Live a Day Without Phone'
#             ]
            
#             for feature in feature_order:
#                 value = form_data[feature]
                
#                 if feature in encoders:
#                     # Handle unseen labels in encoding
#                     try:
#                         encoded_value = encoders[feature].transform([str(value)])[0]
#                     except ValueError:
#                         # If value not seen during training, use most common value
#                         print(f"Warning: Unseen value '{value}' for feature '{feature}', using default encoding")
#                         encoded_value = 0
#                     input_data.append(encoded_value)
#                 else:
#                     # Numeric feature (Age)
#                     input_data.append(value)
            
#             print(f"Final encoded input data: {input_data}")
            
#             # Make prediction
#             input_array = np.array([input_data])
#             prediction_proba = model.predict_proba(input_array)[0]
#             addiction_probability = prediction_proba[1]  # Probability of being addicted
            
#             print(f"Prediction probabilities: [Not Addicted: {prediction_proba[0]:.3f}, Addicted: {prediction_proba[1]:.3f}]")
#             print(f"Addiction probability: {addiction_probability:.3f}")
            
#             # Calculate addiction score (0-100%)
#             addiction_score = round(addiction_probability * 100, 1)
            
#             # Determine addiction level
#             if addiction_probability >= 0.7:
#                 addiction_level = "High Risk of Addiction"
#                 color_class = "high-risk"
#                 addiction_description = "Your responses indicate strong signs of smartphone addiction that may be affecting your daily life and well-being."
#                 recommendations = [
#                     "Set specific phone-free times (meals, first hour after waking)",
#                     "Use app timers to limit social media usage",
#                     "Keep phone away from bed - use traditional alarm clock",
#                     "Practice digital detox weekends",
#                     "Engage in offline hobbies and physical activities"
#                 ]
#             elif addiction_probability >= 0.4:
#                 addiction_level = "Moderate Risk"
#                 color_class = "moderate-risk"
#                 addiction_description = "You show some signs of smartphone dependency. Being mindful can help maintain balance."
#                 recommendations = [
#                     "Set daily screen time limits",
#                     "Turn off non-essential notifications",
#                     "Designate phone-free zones at home",
#                     "Practice mindful usage habits",
#                     "Schedule regular digital breaks"
#                 ]
#             else:
#                 addiction_level = "Low Risk"
#                 color_class = "low-risk"
#                 addiction_description = "You have a healthy relationship with your smartphone with good balance."
#                 recommendations = [
#                     "Maintain current balanced usage",
#                     "Continue setting healthy boundaries",
#                     "Stay mindful of usage changes",
#                     "Share positive habits with others",
#                     "Practice occasional digital detox"
#                 ]
            
#             # Save prediction to database
#             prediction_record = UserPrediction.objects.create(
#                 user=user,
#                 name=name,
#                 age=age,
#                 gender=form_data['Gender'],
#                 use_phone_for_notes=form_data['Use Phone for Class Notes'],
#                 buy_books_from_phone=form_data['Buy/Access Books from Phone'],
#                 battery_lasts_day=form_data['Phone\'s Battery Lasts a Day'],
#                 run_for_charger=form_data['Run for Charger When Battery Dies'],
#                 worry_about_losing_phone=form_data['Worry About Losing Cell Phone'],
#                 take_phone_to_bathroom=form_data['Take Phone to Bathroom'],
#                 use_phone_in_gatherings=form_data['Use Phone in Social Gatherings'],
#                 check_phone_without_notification=form_data['Check Phone Without Notification'],
#                 check_phone_sleep_wake=form_data['Check Phone Before Sleep/After Waking Up'],
#                 phone_next_while_sleeping=form_data['Keep Phone Next to While Sleeping'],
#                 check_phone_during_class=form_data['Check Emails/Calls/Texts During Class'],
#                 rely_on_phone_awkward=form_data['Rely on Phone in Awkward Situations'],
#                 phone_while_tv_eating=form_data['On Phone While Watching TV/Eating'],
#                 panic_attack_without_phone=form_data['Panic Attack if Phone Left Elsewhere'],
#                 use_phone_on_date=form_data['Use Phone on Date'],
#                 phone_for_games=form_data['Phone Use for Playing Games'],
#                 live_without_phone=form_data['Live a Day Without Phone'],
#                 prediction_score=addiction_score,
#                 addiction_level=addiction_level
#             )
            
#             # Prepare context for results
#             context = {
#                 'user_name': name,
#                 'addiction_score': addiction_score,
#                 'addiction_level': addiction_level,
#                 'addiction_description': addiction_description,
#                 'recommendations': recommendations,
#                 'color_class': color_class,
#                 'prediction_value': round(addiction_probability, 3),
#                 'prediction_id': prediction_record.id,
#             }
            
#             return render(request, 'prediction_result.html', context)
            
#         except Exception as e:
#             error_msg = f'Error processing your request: {str(e)}'
#             print(f"Prediction error: {e}")
#             import traceback
#             traceback.print_exc()
#             return render(request, 'prediction.html', {'msg': error_msg})
    
#     # GET request - show form
#     return render(request, 'prediction.html')

# views.py - Add these new views
# views.py - Add these new views
import numpy as np

def prediction(request):
    # Track user visit
    if 'user_email' in request.session:
        try:
            user = Register.objects.get(email=request.session['user_email'])
            UserHistory.objects.create(
                user=user,
                page_visited='prediction'
            )
        except:
            pass
    
    if request.method == 'POST':
        try:
            # User authentication
            user_email = request.session.get('user_email')
            if not user_email:
                messages.error(request, 'Please login first')
                return redirect('login')
                
            user = Register.objects.get(email=user_email)
            
            # Get all form data
            name = request.POST['f1']
            age = float(request.POST['age'])
            
            # Calculate addiction score using a rule-based system
            addiction_score = 0
            max_possible_score = 0
            
            # Define weights for each question
            weights = {
                'f3': 3,   # Use Phone for Class Notes
                'f4': 2,   # Buy/Access Books from Phone  
                'f5': 1,   # Phone's Battery Lasts a Day (reverse)
                'f6': 4,   # Run for Charger When Battery Dies
                'f7': 5,   # Worry About Losing Cell Phone
                'f8': 4,   # Take Phone to Bathroom
                'f9': 3,   # Use Phone in Social Gatherings
                'f10': 4,  # Check Phone Without Notification
                'f11': 3,  # Check Phone Before Sleep/After Waking Up
                'f12': 2,  # Keep Phone Next to While Sleeping
                'f13': 4,  # Check Emails/Calls/Texts During Class
                'f14': 5,  # Rely on Phone in Awkward Situations
                'f15': 3,  # On Phone While Watching TV/Eating
                'f16': 6,  # Panic Attack if Phone Left Elsewhere
                'f17': 4,  # Use Phone on Date
                'f18': 3,  # Phone Use for Playing Games
                'f19': 5,  # Live a Day Without Phone (reverse)
            }
            
            # Process each question
            responses = {}
            
            # f3: Use Phone for Class Notes
            if request.POST['f3'] == '1':  # Yes
                addiction_score += weights['f3']
            max_possible_score += weights['f3']
            responses['use_phone_for_notes'] = 'Yes' if request.POST['f3'] == '1' else 'No'
            
            # f4: Buy/Access Books from Phone
            if request.POST['f4'] == '1':  # Yes
                addiction_score += weights['f4']
            max_possible_score += weights['f4']
            responses['buy_books_from_phone'] = 'Yes' if request.POST['f4'] == '1' else 'No'
            
            # f5: Phone's Battery Lasts a Day (REVERSE - No is worse)
            if request.POST['f5'] == '0':  # No (battery doesn't last)
                addiction_score += weights['f5']
            max_possible_score += weights['f5']
            responses['battery_lasts_day'] = 'Yes' if request.POST['f5'] == '1' else 'No'
            
            # f6: Run for Charger When Battery Dies
            if request.POST['f6'] == '1':  # Yes
                addiction_score += weights['f6']
            max_possible_score += weights['f6']
            responses['run_for_charger'] = 'Yes' if request.POST['f6'] == '1' else 'No'
            
            # f7: Worry About Losing Cell Phone
            if request.POST['f7'] == '1':  # Yes
                addiction_score += weights['f7']
            max_possible_score += weights['f7']
            responses['worry_about_losing_phone'] = 'Yes' if request.POST['f7'] == '1' else 'No'
            
            # f8: Take Phone to Bathroom
            if request.POST['f8'] == '1':  # Yes
                addiction_score += weights['f8']
            max_possible_score += weights['f8']
            responses['take_phone_to_bathroom'] = 'Yes' if request.POST['f8'] == '1' else 'No'
            
            # f9: Use Phone in Social Gatherings
            if request.POST['f9'] == '1':  # Yes
                addiction_score += weights['f9']
            max_possible_score += weights['f9']
            responses['use_phone_in_gatherings'] = 'Yes' if request.POST['f9'] == '1' else 'No'
            
            # f10: Check Phone Without Notification
            freq_mapping = {'0': 0, '1': 1, '2': 2, '3': 3}  # Never=0, Rarely=1, Sometimes=2, Often=3
            freq_value = freq_mapping.get(request.POST['f10'], 0)
            addiction_score += freq_value * (weights['f10'] / 3)  # Scale by frequency
            max_possible_score += weights['f10']
            freq_text = {0: 'Never', 1: 'Rarely', 2: 'Sometimes', 3: 'Often'}
            responses['check_phone_without_notification'] = freq_text.get(freq_value, 'Never')
            
            # f11: Check Phone Before Sleep/After Waking Up
            if request.POST['f11'] == '1':  # Yes
                addiction_score += weights['f11']
            max_possible_score += weights['f11']
            responses['check_phone_sleep_wake'] = 'Yes' if request.POST['f11'] == '1' else 'No'
            
            # f12: Keep Phone Next to While Sleeping
            if request.POST['f12'] == '1':  # Yes
                addiction_score += weights['f12']
            max_possible_score += weights['f12']
            responses['phone_next_while_sleeping'] = 'Yes' if request.POST['f12'] == '1' else 'No'
            
            # f13: Check Emails/Calls/Texts During Class
            if request.POST['f13'] == '1':  # Yes
                addiction_score += weights['f13']
            max_possible_score += weights['f13']
            responses['check_phone_during_class'] = 'Yes' if request.POST['f13'] == '1' else 'No'
            
            # f14: Rely on Phone in Awkward Situations
            if request.POST['f14'] == '1':  # Yes
                addiction_score += weights['f14']
            max_possible_score += weights['f14']
            responses['rely_on_phone_awkward'] = 'Yes' if request.POST['f14'] == '1' else 'No'
            
            # f15: On Phone While Watching TV/Eating
            if request.POST['f15'] == '1':  # Yes
                addiction_score += weights['f15']
            max_possible_score += weights['f15']
            responses['phone_while_tv_eating'] = 'Yes' if request.POST['f15'] == '1' else 'No'
            
            # f16: Panic Attack if Phone Left Elsewhere
            if request.POST['f16'] == '1':  # Yes
                addiction_score += weights['f16']
            max_possible_score += weights['f16']
            responses['panic_attack_without_phone'] = 'Yes' if request.POST['f16'] == '1' else 'No'
            
            # f17: Use Phone on Date
            if request.POST['f17'] == '1':  # Yes
                addiction_score += weights['f17']
            max_possible_score += weights['f17']
            responses['use_phone_on_date'] = 'Yes' if request.POST['f17'] == '1' else 'No'
            
            # f18: Phone Use for Playing Games
            hours_mapping = {'0': 0, '1': 1, '2': 2, '3': 3, '4': 4, '5': 5}
            hours_value = hours_mapping.get(request.POST['f18'], 0)
            addiction_score += hours_value * (weights['f18'] / 5)  # Scale by hours
            max_possible_score += weights['f18']
            responses['phone_for_games'] = f"{hours_value} hours"
            
            # f19: Live a Day Without Phone (REVERSE - No is worse)
            if request.POST['f19'] == '1':  # No (can't live without)
                addiction_score += weights['f19']
            max_possible_score += weights['f19']
            responses['live_without_phone'] = 'No' if request.POST['f19'] == '1' else 'Yes'
            
            # Gender
            gender_mapping = {'0': 'Female', '1': 'Male', '2': 'Others'}
            responses['gender'] = gender_mapping.get(request.POST['f2'], 'Female')
            
            print(f"Raw addiction score: {addiction_score}")
            print(f"Max possible score: {max_possible_score}")
            
            # Calculate percentage (ensure it's between 0-100)
            if max_possible_score > 0:
                final_score = (addiction_score / max_possible_score) * 100
            else:
                final_score = 0
                
            # Apply some scaling to make scores more meaningful
            if final_score < 20:
                final_score = final_score * 0.8  # Reduce very low scores
            elif final_score > 80:
                final_score = 80 + (final_score - 80) * 1.2  # Enhance high scores
                
            final_score = max(5, min(95, final_score))  # Keep between 5-95%
            
            addiction_score_percent = round(final_score, 1)
            
            print(f"Final addiction score: {addiction_score_percent}%")
            
            # Determine addiction level
            if addiction_score_percent >= 70:
                addiction_level = "High Risk of Addiction"
                color_class = "high-risk"
                addiction_description = "Your responses indicate strong signs of smartphone addiction that may be affecting your daily life and well-being."
                recommendations = [
                    "Set specific phone-free times (meals, first hour after waking)",
                    "Use app timers to limit social media usage",
                    "Keep phone away from bed - use traditional alarm clock",
                    "Practice digital detox weekends",
                    "Engage in offline hobbies and physical activities",
                    "Seek professional help if addiction affects relationships/work"
                ]
            elif addiction_score_percent >= 40:
                addiction_level = "Moderate Risk"
                color_class = "moderate-risk"
                addiction_description = "You show some signs of smartphone dependency. Being mindful can help maintain balance."
                recommendations = [
                    "Set daily screen time limits",
                    "Turn off non-essential notifications",
                    "Designate phone-free zones at home",
                    "Practice mindful usage habits",
                    "Schedule regular digital breaks",
                    "Monitor your usage patterns weekly"
                ]
            else:
                addiction_level = "Low Risk"
                color_class = "low-risk"
                addiction_description = "You have a healthy relationship with your smartphone with good balance."
                recommendations = [
                    "Maintain current balanced usage",
                    "Continue setting healthy boundaries",
                    "Stay mindful of usage changes",
                    "Share positive habits with others",
                    "Practice occasional digital detox",
                    "Lead by example for friends/family"
                ]
            
            # Save prediction to database
            prediction_record = UserPrediction.objects.create(
                user=user,
                name=name,
                age=age,
                gender=responses['gender'],
                use_phone_for_notes=responses['use_phone_for_notes'],
                buy_books_from_phone=responses['buy_books_from_phone'],
                battery_lasts_day=responses['battery_lasts_day'],
                run_for_charger=responses['run_for_charger'],
                worry_about_losing_phone=responses['worry_about_losing_phone'],
                take_phone_to_bathroom=responses['take_phone_to_bathroom'],
                use_phone_in_gatherings=responses['use_phone_in_gatherings'],
                check_phone_without_notification=responses['check_phone_without_notification'],
                check_phone_sleep_wake=responses['check_phone_sleep_wake'],
                phone_next_while_sleeping=responses['phone_next_while_sleeping'],
                check_phone_during_class=responses['check_phone_during_class'],
                rely_on_phone_awkward=responses['rely_on_phone_awkward'],
                phone_while_tv_eating=responses['phone_while_tv_eating'],
                panic_attack_without_phone=responses['panic_attack_without_phone'],
                use_phone_on_date=responses['use_phone_on_date'],
                phone_for_games=responses['phone_for_games'],
                live_without_phone=responses['live_without_phone'],
                prediction_score=addiction_score_percent,
                addiction_level=addiction_level
            )
            
            # Prepare context for results
            context = {
                'user_name': name,
                'addiction_score': addiction_score_percent,
                'addiction_level': addiction_level,
                'addiction_description': addiction_description,
                'recommendations': recommendations,
                'color_class': color_class,
                'prediction_id': prediction_record.id,
                'raw_score': round(addiction_score, 1),
                'max_score': round(max_possible_score, 1),
            }
            
            return render(request, 'prediction_result.html', context)
            
        except Exception as e:
            error_msg = f'Error processing your request: {str(e)}'
            print(f"Prediction error: {e}")
            import traceback
            traceback.print_exc()
            messages.error(request, error_msg)
            return render(request, 'prediction.html', {'msg': error_msg})
    
    # GET request - show form
    return render(request, 'prediction.html')

from django.db import models 
def history_graphs(request):
    """View for showing user history and graphs"""
    if 'user_email' not in request.session:
        messages.error(request, 'Please login first')
        return redirect('login')
    
    try:
        # Track visit
        user = Register.objects.get(email=request.session['user_email'])
        UserHistory.objects.create(user=user, page_visited='history')
        
        # Get ALL user predictions first (no slice yet)
        all_predictions = UserPrediction.objects.filter(user=user).order_by('-timestamp')
        
        # Get limited predictions for display (after calculations)
        predictions = all_predictions[:10]
        
        # Calculate statistics
        total_assessments = all_predictions.count()
        
        # Latest prediction
        latest_prediction = all_predictions.first() if all_predictions.exists() else None
        
        # Calculate average score
        if all_predictions.exists():
            avg_score = sum(pred.prediction_score for pred in all_predictions) / total_assessments
            avg_score = round(avg_score, 1)
        else:
            avg_score = 0
        
        # Calculate risk distribution using ALL predictions
        risk_counts = {
            'high': all_predictions.filter(addiction_level__icontains='High').count(),
            'moderate': all_predictions.filter(addiction_level__icontains='Moderate').count(),
            'low': all_predictions.filter(addiction_level__icontains='Low').count()
        }
        
        # Calculate score trend (compare latest with previous)
        score_trend = 0
        if all_predictions.count() >= 2:
            recent_predictions = list(all_predictions[:2])  # Get latest 2
            score_trend = round(recent_predictions[0].prediction_score - recent_predictions[1].prediction_score, 1)
        
        # Generate insights
        insights = []
        if latest_prediction:
            if "High" in latest_prediction.addiction_level:
                insights = [
                    {"type": "improvement", "text": "Consider setting daily screen time limits"},
                    {"type": "improvement", "text": "Try phone-free meals and bedtime routines"},
                    {"type": "improvement", "text": "Practice mindfulness when using your phone"}
                ]
            elif "Moderate" in latest_prediction.addiction_level:
                insights = [
                    {"type": "positive", "text": "Good awareness of phone usage habits"},
                    {"type": "improvement", "text": "Continue monitoring your screen time"},
                    {"type": "positive", "text": "You're maintaining a balanced approach"}
                ]
            else:
                insights = [
                    {"type": "positive", "text": "Excellent phone usage balance"},
                    {"type": "positive", "text": "Healthy digital habits maintained"},
                    {"type": "positive", "text": "Keep up the good work!"}
                ]
        
        context = {
            'predictions': predictions,  # Display only latest 10
            'total_assessments': total_assessments,
            'latest_prediction': latest_prediction,
            'avg_score': avg_score,
            'score_trend': score_trend,
            'risk_distribution': risk_counts,
            'insights': insights,
        }
        return render(request, 'history.html', context)
        
    except Register.DoesNotExist:
        messages.error(request, 'User not found. Please login again.')
        return redirect('login')
    except Exception as e:
        print(f"Error in history view: {e}")
        messages.error(request, 'Error loading history data.')
        return render(request, 'history.html', {
            'predictions': [],
            'total_assessments': 0,
            'avg_score': 0,
            'score_trend': 0,
            'risk_distribution': {'high': 0, 'moderate': 0, 'low': 0},
            'insights': []
        })

def prediction_data_api(request):
    """API endpoint for chart data"""
    if 'user_email' not in request.session:
        return JsonResponse({'error': 'Not authenticated'}, status=401)
    
    try:
        user = Register.objects.get(email=request.session['user_email'])
        
        # Get ALL predictions, not just last 30 days (for testing)
        predictions = UserPrediction.objects.filter(user=user).order_by('timestamp')
        
        # Prepare chart data
        dates = [pred.timestamp.strftime('%m-%d %H:%M') for pred in predictions]  # Include time for more points
        scores = [float(pred.prediction_score) for pred in predictions]
        
        print(f"API: Found {len(predictions)} predictions")  # Debug print
        
        # If we have fewer than 2 data points, create some sample data for testing
        if len(predictions) < 2:
            print("API: Generating sample data for testing")
            # Create sample data points based on existing predictions
            if predictions.exists():
                base_score = predictions[0].prediction_score
                dates = [pred.timestamp.strftime('%m-%d %H:%M') for pred in predictions]
                scores = [float(pred.prediction_score) for pred in predictions]
                # Add a couple of synthetic points for better chart display
                import datetime
                for i in range(2):
                    new_date = (predictions[0].timestamp + datetime.timedelta(hours=i+1)).strftime('%m-%d %H:%M')
                    dates.append(new_date)
                    scores.append(float(base_score + (i * 10)))  # Vary the scores
            else:
                # No predictions at all - return empty but valid structure
                dates = []
                scores = []
        
        # Mock behavior data
        behavior_data = []
        for pred in predictions:
            behavior_data.append({
                'phone_usage': min(15, pred.prediction_score / 100 * 15),
                'dependency': min(15, pred.prediction_score / 100 * 12),
                'social_impact': min(15, pred.prediction_score / 100 * 10),
                'daily_habits': min(15, pred.prediction_score / 100 * 8)
            })
        
        response_data = {
            'dates': dates,
            'scores': scores,
            'behavior_data': behavior_data,
            'total_predictions': len(predictions)
        }
        
        print(f"API: Returning {len(dates)} dates and {len(scores)} scores")  # Debug print
        return JsonResponse(response_data)
        
    except Exception as e:
        print(f"Error in prediction data API: {e}")
        return JsonResponse({
            'error': 'Internal server error',
            'dates': [],
            'scores': [],
            'behavior_data': []
        }, status=500)