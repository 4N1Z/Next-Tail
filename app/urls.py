from django.urls import path
from . import views

urlpatterns = [
    path('',views.HomeView.as_view(),name='home'),
    path('CSStoTailwind',views.cssToTailwindView.as_view(),name='css_to_tailwind'),
    path('nextAI',views.nextAIView.as_view(),name='nextai'),
    path('UItoCode',views.UItoCodeView.as_view(),name='ui_to_code'),
    # path('generate-stream',views.StreamGeneratorView.as_view(),name='generate_stream'),
    path('gemini',views.Gemini.as_view(),name='gemini'),
]