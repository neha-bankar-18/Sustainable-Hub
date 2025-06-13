from django.db import models
from django.contrib.auth.models import AbstractUser, Group, Permission


class CustomUser(AbstractUser):
    USER_Types=[
        ('admin', 'admin'),
       ('user', 'user'),
    ]
    user_type=models.CharField(max_length=10,choices=USER_Types,default='user')
    profile_pic = models.ImageField(upload_to='profile_pics/', null=True)
    uploaded_at=models.DateTimeField(auto_now_add=True)
    groups = models.ManyToManyField(Group, related_name="customuser_groups", blank=True)
    user_permissions = models.ManyToManyField(Permission, related_name="customuser_permissions", blank=True)
    


from django.conf import settings

class AIModel(models.Model):
    name = models.CharField(max_length=255)
    description = models.TextField(blank=True)
    model_file = models.FileField(upload_to='models/')  # Optional, if you're uploading models
    uploaded_at = models.DateTimeField(auto_now_add=True)
    uploaded_by = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name='uploaded_models'
    )

    def __str__(self):
        return self.name


from django.contrib.auth import get_user_model

CustomUser = get_user_model()

class Reward(models.Model):
    user = models.OneToOneField(CustomUser, on_delete=models.CASCADE)
    points = models.IntegerField(default=0)

    def __str__(self):
        return f"{self.user.username} - {self.points} points"

class WasteClassification(models.Model):
    user = models.ForeignKey(CustomUser, on_delete=models.CASCADE)
    image = models.ImageField(upload_to='classified_waste/')
    predicted_class = models.CharField(max_length=100)
    confidence = models.FloatField()
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.user.username} - {self.predicted_class}"

class CropRecommendationHistory(models.Model):
    user = models.ForeignKey(CustomUser, on_delete=models.CASCADE)
    nitrogen = models.FloatField()
    phosphorus = models.FloatField()
    potassium = models.FloatField()
    temperature = models.FloatField()
    humidity = models.FloatField()
    ph = models.FloatField()
    rainfall = models.FloatField()
    predicted_crop = models.CharField(max_length=100)
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.user.username} - {self.predicted_crop} on {self.created_at.strftime('%Y-%m-%d')}"




class CarbonFootprint(models.Model):
    user = models.ForeignKey(CustomUser, on_delete=models.CASCADE)
    prediction = models.FloatField()
    tree_count = models.FloatField()
    created_at = models.DateTimeField(auto_now_add=True)

    # Optional: Save raw input for history/debug
    data = models.JSONField()

    def __str__(self):
        return f"{self.user.username} - {self.prediction:.2f} kg CO₂"

