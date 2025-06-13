# myapp/admin.py

from django.contrib import admin
from .models import AIModel, CustomUser
from django.contrib.auth.admin import UserAdmin

admin.site.register(CustomUser, UserAdmin)
admin.site.register(AIModel)
