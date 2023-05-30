from django.urls import path
from detect import views
from .views import StoreAggregatedDataView, line_chart,crowd_line_chart,month_line_chart, crowd_analysis,crowd_line_chart_minute,foot_analysis, ResetRecordView, ResetTrafficView, CrowdStoreAggregatedDataView, ResetPopulationRecordView

app_name='detect'

urlpatterns = [
    # 기본 페이지
    path('', views.index, name = 'index'),
    path('home_foot_traffic/', views.home_foot_traffic, name='home_foot_traffic'),
    path('home_crowd_estimation/', views.home_crowd_estimation, name='home_crowd_estimation'),
    
    # 모델 동작 페이지
    
    path('stream_traffic/', views.stream_traffic, name = 'stream_traffic'),
    path('stream_crowd/', views.stream_crowd, name = 'stream_crowd'),
    path('set_stream_crowd/', views.set_stream_crowd, name = 'set_stream_crowd'),
    # path('analyze_crowd/<int:people_count>/', views.analyze_crowd, name='analyze_crowd'), ## 이거 추가함
    # 모델 연산 기능
    path('video_feed', views.video_feed, name = 'video_feed'),
    path('video', views.video, name = 'video'),

    # DB aggregate
    path('reset_record/', ResetRecordView.as_view(), name='reset_record'),
    path('crowd_reset_record/', ResetPopulationRecordView.as_view(), name='crowd_reset_record'),
    path('reset_traffic/', ResetTrafficView.as_view(), name='reset_traffic'),
    path('store_aggregated_data/', StoreAggregatedDataView.as_view(), name='store_aggregated_data'),
    path('store_aggregated_data/', views.month_line_chart, name='month_store_aggregated_data'),
    path('crowd_store_aggregated_data/', CrowdStoreAggregatedDataView.as_view(), name='crowd_store_aggregated_data'),

    #make Chart
    path('crowd_analysis/',crowd_analysis, name='crowd_analysis'),
    path('foot_analysis/',foot_analysis, name='foot_analysis'),
    path('foot_line_chart/', line_chart, name='foot_line_chart'),
    path('foot_line_chart_month/', month_line_chart, name='foot_line_chart_month'),
    path('crowd_line_chart/', crowd_line_chart, name='crowd_line_chart'),
    path('crowd_line_chart_minute/',crowd_line_chart_minute , name='crowd_line_chart_minute'),
    path('get_data/', views.get_data, name='get_data'),
]
