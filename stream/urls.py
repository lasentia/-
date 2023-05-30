from django.contrib import admin
from django.urls import path, include

urlpatterns = [
    path('', include('detect.urls')),
    path('admin/', admin.site.urls),
       
    # path('realtime/', include('graph.urls')),
]
