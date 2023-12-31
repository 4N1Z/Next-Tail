from django.urls import path
from . import views

urlpatterns = [
    path('',views.HomeView.as_view(),name='home'),
    path('CSStoTailwind',views.cssToTailwindView.as_view(),name='css_to_tailwind'),
    path('nextAI',views.nextAIView.as_view(),name='nextAI'),
    path('UItoCode',views.UItoCodeView.as_view(),name='ui_to_code'),
    path('ui',views.UItoCode.as_view(),name='ui'),
    path("NextTail", views.NextTailLLM.as_view(), name="nexttail"),
    # path('generate-stream',views.StreamGeneratorView.as_view(),name='generate_stream'),
    path('gemini',views.GeminiAPI.as_view(),name='gemini'),

]