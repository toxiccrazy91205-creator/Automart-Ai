from django.urls import path
from . import views

urlpatterns = [
    path('', views.sales_inventory, name='sales_inventory'),
    path('customers/', views.customers_recommendations, name='customers_recommendations'),
    path('forecasting/', views.forecasting, name='forecasting'),
    path('run_lstm/', views.run_lstm, name='run_lstm'),
]
