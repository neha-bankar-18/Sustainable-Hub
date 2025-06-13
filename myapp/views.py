import random
from django.core.mail import send_mail
from django.shortcuts import render, redirect, get_object_or_404
from django.contrib import messages
from django.contrib.auth.models import User
from django.conf import settings
from django.http import JsonResponse
from django.shortcuts import render, redirect
from django.contrib.auth import authenticate, login, logout
from .models import CustomUser, AIModel
from .models import AIModel
from django.contrib.auth.decorators import login_required
from django.contrib.auth.decorators import user_passes_test

def home(request):
    return render(request, 'home/index.html')

def contact(request):
    if request.method == 'POST':
        name = request.POST.get('name')
        email = request.POST.get('email')
        subject = request.POST.get('subject')
        message = request.POST.get('message')

        full_message = f"Message from {name} <{email}>:\n\n{message}"

        try:
            send_mail(
                subject,
                full_message,
                email,
                ['sustainablehub04@gmail.com'],  # your destination email
                fail_silently=False,
            )
            messages.success(request, '✅ Your message has been sent. Thank you!')
        except Exception as e:
            messages.error(request, f"❌ Error sending email: {e}")

        return redirect('contact')

    return render(request, 'home/contact.html')



def about(request):
    return render(request,'home/about.html')

def services(request):
    return render(request,'home/services.html')


def adminregister(request):
    if request.method == 'POST':
        username=request.POST['username']
        first_name=request.POST['firstname']
        last_name=request.POST['lastname']
        password=request.POST['password']
        email=request.POST['email']
        profile_pic = request.FILES.get('image') 
        
        if not username or not password or not email:
            return JsonResponse({'error': 'All fields are required'}, status=400)

        # Check if user already exists
        if CustomUser.objects.filter(username=username).exists():
            return JsonResponse({'error': 'Username already taken'}, status=400)

        if CustomUser.objects.filter(email=email).exists():
            return JsonResponse({'error': 'Email already in use'}, status=400)
        user= CustomUser.objects.create_user(username=username, email=email, password=password,first_name=first_name, last_name=last_name, profile_pic=profile_pic, is_staff=True, is_superuser=True,user_type='admin')
        user.save()

        return redirect(user_login)
    else:
        return render(request, 'admin-register.html')
    
def userregister(request):
    if request.method == 'POST':
        username=request.POST['username']
        password=request.POST['password']
        first_name=request.POST['firstname']
        last_name=request.POST['lastname']
        email=request.POST['email']
        profile_pic = request.FILES.get('image') 
        
        if not username or not password or not email:
            return JsonResponse({'error': 'All fields are required'}, status=400)

        # Check if user already exists
        if CustomUser.objects.filter(username=username).exists():
            return JsonResponse({'error': 'Username already taken'}, status=400)

        if CustomUser.objects.filter(email=email).exists():
            return JsonResponse({'error': 'Email already in use'}, status=400)
        user= CustomUser.objects.create_user(username=username, email=email, password=password, first_name=first_name, last_name=last_name,  profile_pic=profile_pic, user_type='user')    
        user.save()
        
        return redirect(user_login)
    else:
        return render(request, 'user-register.html')
    

def user_login(request):
    if request.method =='POST':
        username = request.POST['username']
        password = request.POST['password']
        user = authenticate(request, username=username, password=password)
        if user is not None:
            login(request, user)
            if user.user_type =='admin':
                return redirect(admin_dashboard)
            else:
                return redirect(user_dashboard)
        else:
            print("Authentication failed!")
            return render(request, 'user-login.html', {'error': 'Invalid credentials'})
    return render(request, 'user-login.html')



@login_required
def admin_dashboard(request):
    admin_count=CustomUser.objects.filter(user_type='admin').count()
    user_count=CustomUser.objects.filter(user_type='user').count()
    model_count=AIModel.objects.count()
    context={
        'admin_count': admin_count,
        'user_count': user_count,
        'model_count':model_count,
        
    }
    return render(request, 'admin_dashboard/index.html', context)


@login_required
def user_dashboard(request):
    # Get user's reward points
    reward, _ = Reward.objects.get_or_create(user=request.user)
    
    # Badge progress calculation
    badges = [
        {'name': 'Eco Beginner', 'threshold': 100, 'icon': '🌱'},
        {'name': 'Green Warrior', 'threshold': 250, 'icon': '🛡️'},
        {'name': 'Planet Saver', 'threshold': 500, 'icon': '🌍'},
    ]
    
    current_badge = next(
        (b for b in reversed(badges) if reward.points >= b['threshold']),
        {'name': 'Newcomer', 'threshold': 0}
    )
    
    next_badge = next(
        (b for b in badges if reward.points < b['threshold']),
        None
    )
    
    progress_percentage = 0
    points_to_next = 0
    
    if next_badge:
        points_to_next = next_badge['threshold'] - reward.points
        progress_percentage = min(
            100, 
            int((reward.points / next_badge['threshold']) * 100)
        )
    
    # User statistics using only existing models
    stats = {
        'waste_count': WasteClassification.objects.filter(
            user=request.user).count(),
        'crop_recommendations': CropRecommendationHistory.objects.filter(
            user=request.user).count(),
        'carbon_calculations': CarbonFootprint.objects.filter(
            user=request.user).count(),
    }
    
    # Create simple recent activities from existing models
    recent_activities = []
    
    # Add waste classifications
    for waste in WasteClassification.objects.filter(user=request.user).order_by('-created_at')[:2]:
        recent_activities.append({
            'description': f"Classified {waste.predicted_class} waste",
            'points': 10,
            'icon': 'trash-alt',
            'color': 'primary',
            'date': waste.created_at
        })
    
    # Add crop recommendations
    for crop in CropRecommendationHistory.objects.filter(user=request.user).order_by('-created_at')[:2]:
        recent_activities.append({
            'description': f"Got recommendation for {crop.predicted_crop}",
            'points': 15,
            'icon': 'seedling',
            'color': 'warning',
            'date': crop.created_at
        })
    
    # Add carbon calculations
    for carbon in CarbonFootprint.objects.filter(user=request.user).order_by('-created_at')[:1]:
        recent_activities.append({
            'description': f"Calculated carbon footprint: {carbon.prediction:.2f} kg CO₂",
            'points': carbon.tree_count * 10,
            'icon': 'leaf',
            'color': 'success',
            'date': carbon.created_at
        })
    
    # Sort activities by date
    recent_activities.sort(key=lambda x: x['date'], reverse=True)
    
    # Hardcoded eco tips (since no EcoTip model)
    eco_tips = [
        "Turning off lights when not in use can save up to 15% on energy bills.",
        "Recycling one aluminum can saves enough energy to power a TV for 3 hours.",
        "Planting native species helps local ecosystems thrive with less water.",
        "Meatless Mondays can reduce your carbon footprint by 8% annually."
    ]
    
    context = {
        'user': request.user,
        'reward_points': reward.points,
        'recent_activities': recent_activities[:5],  # Only show 5 most recent
        'random_eco_tip': random.choice(eco_tips),
        'current_badge': current_badge,
        'next_badge': next_badge,
        'progress_percentage': progress_percentage,
        'points_to_next': points_to_next,
        'stats': stats,
        'badges': badges,
    }
    
    return render(request, 'user_dashboard/index.html', context)

def logout(request):
    return redirect(user_login)

def forgot_password(request):
    if request.method == "POST":
        email = request.POST.get("email")
        try:
            user = CustomUser.objects.get(email=email)
            otp = random.randint(100000, 999999)  # Generate a 6-digit OTP

            # Store OTP in session (valid only for the session)
            request.session['reset_otp'] = otp
            request.session['reset_email'] = email  # Store email in session too

            # Send OTP via Email
            send_mail(
                subject="Password Reset OTP",
                message=f"Your OTP for password reset is: {otp}",
                from_email="sustainablehub04@gmail.com",  # Update this
                recipient_list=[email],
                fail_silently=False,
            )

            messages.success(request, "OTP sent to your email.")
            return redirect("verify_otp")  # Redirect to OTP verification page

        except CustomUser.DoesNotExist:
            messages.error(request, "No account found with this email.")

    return render(request, "forgot_password.html")


def verify_otp(request):
    if request.method == "POST":
        entered_otp = request.POST.get("otp")
        session_otp = request.session.get("reset_otp")

        if session_otp and str(session_otp) == entered_otp:
            return redirect("reset_password")  # Redirect to reset password page
        else:
            messages.error(request, "Invalid OTP. Try again.")

    return render(request, "verify_otp.html")


def reset_password(request):
    if request.method == "POST":
        new_password = request.POST.get("new_password")
        confirm_password = request.POST.get("confirm_password")
        email = request.session.get("reset_email")

        if new_password == confirm_password and email:
            user = CustomUser.objects.get(email=email)
            user.set_password(new_password)
            user.save()

            # Clear session data after password reset
            request.session.pop("reset_otp", None)
            request.session.pop("reset_email", None)

            return redirect("login")

        messages.error(request, "Passwords do not match.")

    return render(request, "reset_password.html")


@user_passes_test(lambda u: u.is_superuser)
def upload_ai_model(request):
    if request.method == 'POST':
        name = request.POST.get('name')
        description = request.POST.get('description')
        model_file = request.FILES.get('model_file')

        if name and model_file:
            AIModel.objects.create(
                name=name,
                description=description,
                model_file=model_file,
                uploaded_by=request.user
            )
            return redirect('upload_ai_model')  # Update with your URL name

    models = AIModel.objects.all().order_by('-uploaded_at')
    return render(request, 'admin_dashboard/ai_model/upload_ai_model.html', {'models': models})




# Only allow admins
@user_passes_test(lambda u: u.is_superuser)
def update_ai_models(request):
    models = AIModel.objects.all()

    # Delete Model
    if request.method == "POST":
        action = request.POST.get("action")
        model_id = request.POST.get("edit_model_id")
        model = get_object_or_404(AIModel, id=model_id)

        if action == "save":
             model.name = request.POST.get("name")
             model.description = request.POST.get("description")
             if request.FILES.get("model_file"):
                model.model_file = request.FILES["model_file"]
             model.save()

        elif action == "delete":
            model.delete()

        return redirect('update_ai_models')
    return render(request, 'admin_dashboard/ai_model/update_ai_models.html', {'models': models})

from django.contrib.auth.decorators import user_passes_test


# Helper to ensure only superusers can access
admin_only = user_passes_test(lambda u: u.is_superuser)


@admin_only
def manage_admins(request):
    """Display users and allow deletion."""
    admins = CustomUser.objects.filter(user_type='admin').order_by('username')

    if request.method == "POST":
        user_id = request.POST.get("delete_user_id")
        admin = get_object_or_404(CustomUser, id=user_id)
        admin.delete()
        messages.success(request, f"Admin {admin.first_name} {admin.last_name} deleted successfully.")
        return redirect('manage_users')

    return render(request, 'admin_dashboard/manage_users/manageadmin.html', {'admins': admins})


@admin_only
def manage_users(request):
    """Display users and allow deletion."""
    users = CustomUser.objects.filter(user_type='user').order_by('username')

    if request.method == "POST":
        user_id = request.POST.get("delete_user_id")
        user = get_object_or_404(CustomUser, id=user_id)
        user.delete()
        messages.success(request, f"User {user.first_name} {user.last_name} deleted successfully.")
        return redirect('manage_users')

    return render(request, 'admin_dashboard/manage_users/manageusers.html', {'users': users})

@login_required
def my_profile(request):
    return render(request, 'admin_dashboard/profile/myprofile.html', {'user': request.user})



@login_required
def edit_profile(request):
    if request.method == 'POST':
        user = request.user  # this will already be an instance of CustomUser

        username = request.POST.get('username')
        name = request.POST.get('name')
        email = request.POST.get('email')
        profile_pic = request.FILES.get('profile_pic')

        if username:
            user.username = username
        if name:
            user.name = name
        if email:
            user.email = email
        if profile_pic:
            user.profile_pic = profile_pic

        user.save()
        messages.success(request, 'Profile updated successfully!')
        return redirect('edit_profile')  # update this name if your URL name is different

    return render(request, 'admin_dashboard/profile/editprofile.html', {'user': request.user})


#waste management module

import os
import random
import numpy as np
import cv2
import tensorflow as tf
from .models import Reward, WasteClassification


ai_model = tf.keras.models.load_model('media/models/transfer_learning_model.h5')

FACTS = [
    "Recycling one aluminum can saves enough energy to run a TV for 3 hours.",
    "Plastic can take over 400 years to degrade in landfills.",
    "Composting reduces methane emissions from landfills.",
    "One tree can absorb up to 48 pounds of CO2 per year.",
]

CLASS_NAMES = {
    0: 'Plastic', 1: 'Paper', 2: 'Glass', 3: 'Metal',
    4: 'Cardboard', 5: 'Food Organics', 6: 'Miscellaneous Trash',
    7: 'Textile Trash', 8: 'Vegetation'
}

DISPOSAL_TASKS = {
    'Plastic': 'Take plastic to the nearest recycling bin.',
    'Paper': 'Place paper in the blue paper bin.',
    'Glass': 'Drop off glass at the designated glass recycling point.',
    'Metal': 'Send metal waste to a scrap metal collection center.',
    'Cardboard': 'Flatten and recycle cardboard in the brown bin.',
    'Food Organics': 'Compost food waste or place it in the green bin.',
    'Miscellaneous Trash': 'Dispose of this in general waste.',
    'Textile Trash': 'Donate or reuse textile waste if possible.',
    'Vegetation': 'Compost garden waste or use a green waste bin.'
}

def preprocess_image(image_path):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (299, 299))  # InceptionV3 input size
    img = img / 255.0
    return np.expand_dims(img, axis=0)

@login_required
def waste_classifier(request):
    prediction = None
    image_url = None
    task = None
    class_name = None

    reward_entry, _ = Reward.objects.get_or_create(user=request.user)

    if request.method == 'POST':
        if 'complete_task' in request.POST:
            reward_entry.points += 10
            reward_entry.save()
            return redirect('waste_classifier')

        elif request.FILES.get('image'):
            image = request.FILES['image']
            image_path = os.path.join(settings.MEDIA_ROOT, image.name)

            with open(image_path, 'wb+') as f:
                for chunk in image.chunks():
                    f.write(chunk)

            input_data = preprocess_image(image_path)
            pred = ai_model.predict(input_data)
            class_idx = np.argmax(pred)
            class_name = CLASS_NAMES[class_idx]
            confidence = float(np.max(pred))
            prediction = f"{class_name} ({confidence*100:.2f}%)"
            image_url = settings.MEDIA_URL + image.name
            task = DISPOSAL_TASKS.get(class_name, "No specific instructions available.")

            # Save to classification history
            WasteClassification.objects.create(
                user=request.user,
                image=image,
                predicted_class=class_name,
                confidence=confidence,
                
            )

    # Get history
    history = WasteClassification.objects.filter(user=request.user).order_by('-created_at')

    return render(request, 'user_dashboard/waste_classifier.html', {
        'prediction': prediction,
        'image_url': image_url,
        'reward_points': reward_entry.points,
        'task': task,
        'fact': random.choice(FACTS),
        'history': history,
        'test_lat': 40.7128,  # New York coordinates for testing
        'test_lon': -74.0060,
        'waste_keyword': class_name.lower() if class_name else 'recycling',
    })



from .models import CropRecommendationHistory
import pickle

# Load pre-trained models
model = pickle.load(open(os.path.join(settings.MEDIA_ROOT, 'models/model.pkl'), 'rb'))
sc = pickle.load(open(os.path.join(settings.MEDIA_ROOT, 'models/standscaler.pkl'), 'rb'))
ms = pickle.load(open(os.path.join(settings.MEDIA_ROOT, 'models/minmaxscaler.pkl'), 'rb'))

CROP_DICT = {
    1: "Rice", 2: "Maize", 3: "Jute", 4: "Cotton", 5: "Coconut", 6: "Papaya", 7: "Orange",
    8: "Apple", 9: "Muskmelon", 10: "Watermelon", 11: "Grapes", 12: "Mango", 13: "Banana",
    14: "Pomegranate", 15: "Lentil", 16: "Blackgram", 17: "Mungbean", 18: "Mothbeans",
    19: "Pigeonpeas", 20: "Kidneybeans", 21: "Chickpea", 22: "Coffee"
}

CROP_FACTS = [
    "Using crop rotation helps improve soil health and reduce pests.",
    "Drip irrigation can save up to 50% more water than traditional methods.",
    "Composting enriches soil and reduces the need for chemical fertilizers.",
    "Growing native crops can increase yield with fewer resources.",
    "Cover crops can prevent erosion and enhance soil fertility.",
    "Organic farming reduces pollution and is better for biodiversity.",
    "Intercropping increases land productivity and suppresses weeds.",
]

@login_required
def crop_recommendation(request):
    result = None

    reward_entry, _ = Reward.objects.get_or_create(user=request.user)

    if request.method == 'POST':
        if 'complete_task' in request.POST:
            reward_entry.points += 15  # 15 points for completing crop planting
            reward_entry.save()
            return redirect('crop_recommendation')

        N = request.POST.get('Nitrogen')
        P = request.POST.get('Phosporus')
        K = request.POST.get('Potassium')
        temp = request.POST.get('Temperature')
        humidity = request.POST.get('Humidity')
        ph = request.POST.get('Ph')
        rainfall = request.POST.get('Rainfall')

        features = [N, P, K, temp, humidity, ph, rainfall]
        if all(features):
            input_data = np.array(features).reshape(1, -1)
            scaled_features = ms.transform(input_data)
            final_features = sc.transform(scaled_features)

            prediction = model.predict(final_features)[0]
            crop = CROP_DICT.get(prediction, "Unknown Crop")

            result = f"{crop} is the best crop to be cultivated at your location."

            # Save to crop recommendation history
            CropRecommendationHistory.objects.create(
                user=request.user,
                nitrogen=N,
                phosphorus=P,
                potassium=K,
                temperature=temp,
                humidity=humidity,
                ph=ph,
                rainfall=rainfall,
                predicted_crop=crop
            )

    # Fetch user's crop recommendation history
    history = CropRecommendationHistory.objects.filter(user=request.user).order_by('-created_at')

    return render(request, 'user_dashboard/crop_recommendation.html', {
        'result': result,
        'reward_points': reward_entry.points,
        'history': history,
        'fact': random.choice(CROP_FACTS),
    })




import pickle
import numpy as np
import pandas as pd
from django.conf import settings

# Corrected preprocessing function
def preprocess_input(data):
    df = pd.DataFrame([data])

    # Categorical columns to one-hot encode
    categorical_cols = [
        'Body Type', 'Sex', 'Diet', 'How Often Shower', 'Heating Energy Source',
        'Transport', 'Social Activity', 'Frequency of Traveling by Air',
        'Waste Bag Size', 'Vehicle Type', 'Energy efficiency'
    ]

    # One-hot encode the categorical columns
    df = pd.get_dummies(df, columns=categorical_cols)

    # Add missing columns from training
    expected_columns = pickle.load(open(os.path.join(settings.MEDIA_ROOT, 'models/expected_columns.sav'), 'rb'))
    for col in expected_columns:
        if col not in df.columns:
            df[col] = 0  # Add missing columns with 0
    df = df[expected_columns]  # Ensure column order matches training

    return df



from .models import CarbonFootprint


from django.contrib import messages
from django.shortcuts import redirect
from .models import Reward

@login_required
def carbon_footprint_view(request):
    prediction = None
    tree_count = None

    # Handle task completion
    if request.method == 'POST' and 'complete_task' in request.POST:
        tree_count = int(request.POST.get('tree_count', 0))
        reward, created = Reward.objects.get_or_create(user=request.user)
        points_earned = tree_count * 10  # 10 points per tree
        reward.points += points_earned
        reward.save()
        messages.success(request, f'Task completed! You earned {points_earned} reward points!')
        return redirect(carbon_footprint_view)

    # Handle carbon footprint calculation
    if request.method == 'POST' and 'complete_task' not in request.POST:
        try:
            # Extract form input
            height = float(request.POST.get('height'))
            weight = float(request.POST.get('weight'))
            sex = request.POST.get('sex')
            diet = request.POST.get('diet')
            social = request.POST.get('social')
            shower = request.POST.get('shower')
            heating_energy = request.POST.get('heating_energy')
            transport = request.POST.get('transport')
            vehicle_type = request.POST.get('vehicle_type', 'None')
            vehicle_km = int(request.POST.get('vehicle_km', 0) or 0)
            air_travel = request.POST.get('air_travel')
            waste_bag = request.POST.get('waste_bag')
            waste_count = int(request.POST.get('waste_count', 0) or 0)
            energy_efficiency = request.POST.get('energy_efficiency')
            daily_tv_pc = int(request.POST.get('daily_tv_pc', 0) or 0)
            internet_daily = int(request.POST.get('internet_daily', 0) or 0)
            grocery_bill = int(request.POST.get('grocery_bill', 0) or 0)
            clothes_monthly = int(request.POST.get('clothes_monthly', 0) or 0)
            recycle = request.POST.getlist('recycle')
            for_cooking = request.POST.getlist('for_cooking')

            bmi = weight / (height / 100) ** 2
            body_type = "underweight" if bmi < 18.5 else "normal" if bmi < 25 else "overweight" if bmi < 30 else "obese"

            data = {
                'Body Type': body_type,
                "Sex": sex,
                'Diet': diet,
                "How Often Shower": shower,
                "Heating Energy Source": heating_energy,
                "Transport": transport,
                "Social Activity": social,
                'Monthly Grocery Bill': grocery_bill,
                "Frequency of Traveling by Air": air_travel,
                "Vehicle Monthly Distance Km": vehicle_km,
                "Waste Bag Size": waste_bag,
                "Waste Bag Weekly Count": waste_count,
                "How Long TV PC Daily Hour": daily_tv_pc,
                "Vehicle Type": vehicle_type,
                "How Many New Clothes Monthly": clothes_monthly,
                "How Long Internet Daily Hour": internet_daily,
                "Energy efficiency": energy_efficiency
            }

            # Manual one-hot values for multi-options
            for item in for_cooking:
                data[f"Cooking_with_{item}"] = 1
            for item in recycle:
                data[f"Do You Recyle_{item}"] = 1

            # Load model and scaler
            model_path = os.path.join(settings.MEDIA_ROOT, 'models/model.sav')
            scale_path = os.path.join(settings.MEDIA_ROOT, 'models/scale.sav')
            model = pickle.load(open(model_path, 'rb'))
            scaler = pickle.load(open(scale_path, 'rb'))

            # Preprocess and predict
            df = preprocess_input(data)
            df_scaled = scaler.transform(df)
            pred = model.predict(df_scaled)

            prediction = round(np.exp(pred[0]))
            tree_count = round(prediction / 411.4)

            # Save to DB
            CarbonFootprint.objects.create(
                user=request.user,
                prediction=prediction,
                tree_count=tree_count,
                data=data
            )

        except Exception as e:
            print("Error in prediction:", e)
            prediction = "Error"
            tree_count = "Error"

    # Show history for current user
    history = CarbonFootprint.objects.filter(user=request.user).order_by('-created_at')

    return render(request, 'user_dashboard/carbon_footprint.html', {
        'prediction': prediction,
        'tree_count': tree_count,
        'history': history
    })


@login_required 
def rewards_dashboard(request):
    reward, created = Reward.objects.get_or_create(user=request.user)
    
    milestones = [100, 250, 500, 1000, 2000, 5000]
    next_milestone = next((m for m in milestones if reward.points < m), None)
    
    if next_milestone:
        progress_percentage = min(100, int((reward.points / next_milestone) * 100))
    else:
        progress_percentage = 100
    
    # Prepare badge data with points needed calculation
    badges = []
    for threshold in milestones:
        badges.append({
            'threshold': threshold,
            'name': {
                100: "Eco Beginner",
                250: "Green Warrior",
                500: "Planet Saver",
                1000: "Climate Hero",
                2000: "Eco Legend",
                5000: "Earth Guardian"
            }[threshold],
            'icon': {
                100: "🌱",
                250: "🛡️",
                500: "🌍",
                1000: "🦸",
                2000: "🏆",
                5000: "👑"
            }[threshold],
            'points_needed': max(0, threshold - reward.points)
        })
    
    footprint_history = CarbonFootprint.objects.filter(user=request.user).order_by('-created_at')[:5]
    
    context = {
        'reward': reward,
        'next_milestone': next_milestone,
        'progress_percentage': progress_percentage,
        'badges': badges,
        'footprint_history': footprint_history
    }
    
    return render(request, 'user_dashboard/rewards.html', context)