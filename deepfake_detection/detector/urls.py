from django.urls import path
from django.conf import settings
from django.conf.urls.static import static
from . import views

urlpatterns = [
    path('', views.HomeView.as_view(), name='home'),
    path('image/', views.ImageAnalysisView.as_view(), name='image_analysis'),
    path('video/', views.VideoAnalysisView.as_view(), name='video_analysis'),
    path('image/result/', views.ImageResultView.as_view(), name='image_result'),
    path('video/result/', views.VideoResultView.as_view(), name='video_result'),
]

if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
