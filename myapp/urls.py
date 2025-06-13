from django.urls import path
from.import views

urlpatterns = [
    path('userreg/',views.userregister,name='userregister'),
    path('login/',views.user_login,name='login'),
    path('userdash/',views.user_dashboard,name='userdash'),
    path('admindash/',views.admin_dashboard,name='admindash'),  
    path('adminreg/',views.adminregister,name='adminregister'),
    path('forgotpass/',views.forgot_password,name='forgot_password'),
    path('resetpass/',views.reset_password,name='reset_password'),
    path('verify/',views.verify_otp,name='verify_otp'),
    path('services/',views.services,name='services'),
    path('',views.home,name='home'),
    path('contact/',views.contact,name='contact'),
    path('about/',views.about,name='about'),
    path('logout/',views.logout,name='logout'),
    
    path('admindash/uploads/',views.upload_ai_model,name='upload_ai_model'),
    path('admindash/updates/',views.update_ai_models,name='update_ai_models'),
    path('admindash/manageadmins/', views.manage_admins,name='manage_admins'),
    path('admindash/manageusers/', views.manage_users,name='manage_users'),
    path('admindash/myprofile/',views.my_profile,name='my_profile'),
    path('admindash/editprofile/',views.edit_profile,name='edit_profile'),
    
    
    path('userdash/wasteclassifier/',views.waste_classifier,name='waste_classifier'),
    path('userdash/croprecommend/',views.crop_recommendation,name='crop_recommendation'),
    path('userdash/carbonfoot/', views.carbon_footprint_view, name='carbon_footprint'),
    path('userdash/rewards/',views.rewards_dashboard,name='rewards')
    
]
