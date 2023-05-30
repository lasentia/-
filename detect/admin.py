from django.contrib import admin
from .models import Foot_Traffic, Foot_Record, Population_Record

# Register your models here.
admin.site.register(Foot_Traffic)
admin.site.register(Foot_Record)
admin.site.register(Population_Record)