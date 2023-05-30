import datetime
from django.shortcuts import render
from django.http import StreamingHttpResponse
import cv2, torch
from requests import request
import yolov5
from detect.line import draw_line
from .models import Foot_Traffic, Foot_Record
from django.utils import timezone
# scale_coords 삭제함
from yolov5.utils.general import (check_img_size, non_max_suppression,
                                  check_imshow, xyxy2xywh, increment_path)
from yolov5.utils.torch_utils import select_device, time_sync
from yolov5.utils.plots import Annotator, colors
from deep_sort.utils.parser import get_config
from deep_sort.deep_sort import DeepSort
from PIL import Image as im
from detect.models import Foot_Record
from django.views.decorators.clickjacking import xframe_options_exempt

# Create your views here.

################# Yolo Start
# load model
# model = yolov5.load('yolov5s.pt')
print(torch.cuda.is_available())
model = torch.hub.load('ultralytics/yolov5', 'yolov5s') 
device = select_device() # 0 for gpu, '' for cpu

# initialize deepsort

cfg = get_config()
cfg.merge_from_file("deep_sort/configs/deep_sort.yaml")
deepsort = DeepSort('osnet_x0_25',
                    max_dist=cfg.DEEPSORT.MAX_DIST,
                    max_iou_distance=cfg.DEEPSORT.MAX_IOU_DISTANCE,
                    max_age=cfg.DEEPSORT.MAX_AGE, n_init=cfg.DEEPSORT.N_INIT, nn_budget=cfg.DEEPSORT.NN_BUDGET,
                    use_cuda=True)

# Get names and colors
names = model.module.names if hasattr(model, 'module') else model.names
yolo_video_path = "D:\yolov5-deepsort-web2\detect\walk3.mp4"

def get_center(bbox):
    """Calculate and return the center coordinates of a bounding box."""
    x1, y1, x2, y2 = bbox
    x_center = int((x1 + x2) / 2)
    y_center = int((y1 + y2) / 2)
    return (x_center, y_center)

def is_crossing_line(prev_center, curr_center, slope, y_intercept):
    # check if both points are on opposite sides of the line
    prev_above = (prev_center[1] < slope * prev_center[0] + y_intercept)
    curr_above = (curr_center[1] < slope * curr_center[0] + y_intercept)
    return prev_above != curr_above

def stream():
    cap = cv2.VideoCapture(yolo_video_path)
    model.conf = 0.45
    model.iou = 0.5
    model.classes = [0,64,39]
    flag = 1
    counter = 0
    count_list = set() #count값을 저장할 set
    above_sub_list = set()  #선을 넘어 가기전의 id값을 저장할 Set 상단
    below_sub_list = set()  #선을 넘어 가기전의 id값을 저장할 Set 하단
    flag = 1
    clicked_points = []
    save_txt=False
    slope = 1  # Change this value to the desired slope
    y_intercept = 0  # Change this value to the desired y-intercept

    current_time = timezone.now()

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: failed to capture image")
            break
    
        results = model(frame, augment=True)

        # proccess
        annotator = Annotator(frame, line_width=2, pil=not ascii)
        det = results.pred[0]

        if det is not None and len(det):    
            xywhs = xyxy2xywh(det[:, 0:4])
            confs = det[:, 4]
            clss = det[:, 5]
            
            # 0번(첫 번째) 인덱스가 person 클래스에 해당합니다.
            person_indices = (clss == 0)

            # person 클래스에 해당하는 탐지 결과만 추출합니다.
            xywhs_person = xywhs[person_indices].cpu()
            confs_person = confs[person_indices].cpu()
            clss_person = clss[person_indices].cpu()
            outputs = deepsort.update(xywhs_person, confs_person, clss_person, frame)

            if len(outputs) > 0:
                for j, (output, conf) in enumerate(zip(outputs, confs)):

                    bboxes = output[0:4]
                    id = output[4]
                    cls = output[5]
                    
                    bbox_center_widht = output[0] + (output[2] - output[0])/2
                    bbox_center_height = output[3] + (output[1] - output[3])/8.3
                    
                    ud = bbox_center_height - slope*bbox_center_widht - y_intercept
                    
                    if ud+3 < 0:
                        if id in below_sub_list and id not in count_list:
                            count_list.add(id)
                            # 모델에 people_count, date, time 삽입
                            today_traffic_instance = Foot_Traffic.objects.create(
                                person_id = id,
                                date=current_time.date(),
                                time=current_time.time(),
                                people_count= 1,
                            )
                            
                            today_traffic_instance.save()  # 데이터베이스에 저장    
                        if id not in above_sub_list and id not in below_sub_list:
                            above_sub_list.add(id)
                                
                    else:
                        if id in above_sub_list and id not in count_list:
                            count_list.add(id)
                            # 모델에 people_count, date, time 삽입
                            today_traffic_instance = Foot_Traffic.objects.create(
                                date=current_time.date(),
                                time=current_time.time(),
                                people_count= 1,
                            )

                            today_traffic_instance.save()  # 데이터베이스에 저장
                            
                        if id not in below_sub_list and id not in above_sub_list:
                            below_sub_list.add(id)
                                                    
                    if save_txt:
                        # to MOT format
                        bbox_left = output[0]
                        bbox_top = output[1]
                        bbox_w = output[2] - output[0]
                        bbox_h = output[3] - output[1]
                    
                    c = int(cls)  ## integer class
                    label = f''
                    annotator.box_label(bboxes, label, color = colors(c, True))
        
        else:
            deepsort.increment_ages()

        im0 = annotator.result()
        # draw line
        if flag:
            flag = 0 # 한 번만 그리게 초기화
            x1, y1, x2, y2 = draw_line.xy(im0) # line
        
            # 선을 기준으로한 일차 방정식의 기울기, y절편 
            slope = (y1 - y2) / (x1 - x2) 
            y_intercept = y1 - slope * x1
        
        # draw line on image
        cv2.line(im0, (x1, y1), (x2, y2), (0, 255, 0), 2)
        scale = im0.shape[0] / 1080  # 현재 영상의 높이를 기준으로 비율을 계산합니다.
        font_scale = max(1, int(scale * 1.5))  # 영상 비율에 맞게 폰트 스케일을 설정합니다.
        # draw counting result on image 
       
        cv2.putText(im0, 'count : '+str(len(count_list)), (25,50), 0, 1, (0, 255, 0), 2)
        # cv2.putText(im0, 'count : ' + str(len(count_list)), (int(25 * scale), int(50 * scale)), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), 2)
        image_bytes = cv2.imencode('.jpg', im0)[1].tobytes()

        # 강제 종료
        if cv2.waitKey(1) == ord('q'):  # 1 millisecond
            exit()
   
        yield(b'--frame\r\n'
              b'Content-Type: image/jpeg\r\n\r\n'+ image_bytes + b'\r\n')
    
def video_feed(request):
   return StreamingHttpResponse(stream(), content_type='multipart/x-mixed-replace; boundary=frame')



#Crowd estimation Start
from django.shortcuts import render, redirect
from django.http import StreamingHttpResponse, HttpResponse
import cv2, torch, time
from detect.FIDTM.Networks.HR_Net.seg_hrnet import * 
from detect.FIDTM.Networks.HR_Net.default import _C as hr_config
# from detect.FIDTM.video_demo import main
import os
from torchvision import datasets, transforms
import scipy
# forms에서 인구 수 입력 진행
from .forms import PopulationForm

# Create your views here.
crowd_video_path = "D:\yolov5-deepsort-web2\detect\crowd2.mp4"
img_transform = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
tensor_transform = transforms.ToTensor()

# Population density analysis
@xframe_options_exempt
def set_stream_crowd(request):
    if request.method == 'POST':
        form = PopulationForm(request.POST) ## form형식에 내용이 추가됨
        if form.is_valid():
            people_count = form.cleaned_data['count']
            request.session['people_count'] = people_count ## 다른 곳에 값 넣기
            # 여기에서 'count' 값에 대한 작업을 수행합니다.
            return render(request, 'success.html', {'count':people_count})  # 처리 성공 시 리다이렉트할 페이지
    else:
        form = PopulationForm()
    # 다시 입력하려면 기본 form양식으로 제출
    return render(request, 'set_stream_crowd.html', {'form': form})

def video(request):
    # 입력받은 인구 수 값 가져오기
    people_count = request.session.get('people_count')
    people_count = people_count - (people_count//100)*5 ## 인구 수 보정하기 
    return StreamingHttpResponse(Crowd(people_count), content_type='multipart/x-mixed-replace; boundary=frame')

def Crowd(people_count):
    model = get_seg_model()
    model = nn.DataParallel(model, device_ids=[0])
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)

    ## pth 모델 불러오기
    checkpoint = torch.load("D:\yolov5-deepsort-web2\model_best_57.pth", map_location=device)
    model.load_state_dict(checkpoint['state_dict'], strict=False)
    ###

    # 추가
    scount = []
    sc_list = []
    c_list = []

    #fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    
    cap = cv2.VideoCapture(crowd_video_path) ## 영상 파일 실행
    ret, frame = cap.read()
    
    '''out video'''
    width = frame.shape[1] // 2 # output size
    height = frame.shape[0] // 2 # output size
    # out = cv2.VideoWriter('C:\projects\yolov5-deepsort-web\static\assets\test.avi', fourcc, 30, (width, height))
    
    while True:
        try:
            ret, frame = cap.read()
            
            scale_factor = 0.5
            frame = cv2.resize(frame, (0, 0), fx=scale_factor, fy=scale_factor)
            ori_img = frame.copy()
        except:
            print("test end")
            cap.release()
            break
        
        frame = frame.copy()
        image = tensor_transform(frame)
        image = img_transform(image).unsqueeze(0)

        with torch.no_grad():
            d6 = model(image)
            
            count, pred_kpoint = counting(d6)

            res = generate_bounding_boxes(pred_kpoint, frame)
            
            scount.append(count)
            c_list.append(count)
            if len(scount) == 11:
              scount.pop(0)
            count = sum(scount) // len(scount)
            sc_list.append(count)

            # 모델에 people_count, date, time 삽입
            current_time = timezone.now()
            second = current_time.second

            if second == 0:  # 현재 second가 00일 때만 데이터 저장
                population_instance = Population_Record.objects.create(
                    date=current_time.date(),
                    time=current_time.time(),
                    people_count=count,
                )
                population_instance.save()


             ## 제한 인원보다 많을 경우 텍스트로 알림           
            if count > people_count:
                # 화면 정중앙에 출력하기
                text = "Population Warning: " + str(count)
                text_width = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1, 5)[0][0]
                text_height = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1, 5)[0][1]
                x = (width - text_width) // 2
                y = (height - text_height) // 2

                # Draw the text at the center coordinates
                cv2.putText(res, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 4)
            

            cv2.putText(res, "Count:" + str(count), (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            '''write in out_video'''
            # out.write(res)

        print("pred:%.1f" % count)
        
        image_bytes = cv2.imencode('.jpg', frame)[1].tobytes()
        yield(b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n'+ image_bytes + b'\r\n')
        
        ## 모델 불러오기
        # update_population(int(count))

def counting(input):
    input_max = torch.max(input).item()
    keep = nn.functional.max_pool2d(input, (3, 3), stride=1, padding=1)
    keep = (keep == input).float()
    input = keep * input

    input[input < 100.0 / 255.0 * torch.max(input)] = 0
    input[input > 0] = 1
    
    '''negative sample'''
    if input_max<0.1:
        input = input * 0

    count = int(torch.sum(input).item())

    kpoint = input.data.squeeze(0).squeeze(0).cpu().numpy()

    return count, kpoint

def generate_point_map(kpoint):
    rate = 1
    pred_coor = np.nonzero(kpoint)
    point_map = np.zeros((int(kpoint.shape[0] * rate), int(kpoint.shape[1] * rate), 3), dtype="uint8") + 255  # 22
    # count = len(pred_coor[0])
    coord_list = []
    for i in range(0, len(pred_coor[0])):
        h = int(pred_coor[0][i] * rate)
        w = int(pred_coor[1][i] * rate)
        coord_list.append([w, h])
        cv2.circle(point_map, (w, h), 3, (0, 0, 0), -1)

    return point_map

def generate_bounding_boxes(kpoint, Img_data):
    '''generate sigma'''
    pts = np.array(list(zip(np.nonzero(kpoint)[1], np.nonzero(kpoint)[0])))
    leafsize = 2048
        
    if pts.shape[0] > 0: # Check if there is a human presents in the frame
        # build kdtree
        tree = scipy.spatial.KDTree(pts.copy(), leafsize=leafsize)

        distances, locations = tree.query(pts, k=4)
        for index, pt in enumerate(pts):
            pt2d = np.zeros(kpoint.shape, dtype=np.float32)
            pt2d[pt[1], pt[0]] = 1.
            if np.sum(kpoint) > 1:
                sigma = (distances[index][1] + distances[index][2] + distances[index][3]) * 0.1
            else:
                sigma = np.average(np.array(kpoint.shape)) / 2. / 2.  # case: 1 point
            sigma = min(sigma, min(Img_data.shape[0], Img_data.shape[1]) * 0.04)

            if sigma < 6:
                t = 2
            else:
                t = 2
            # Img_data = cv2.rectangle(Img_data, (int(pt[0] - sigma), int(pt[1] - sigma)),
            #                         (int(pt[0] + sigma), int(pt[1] + sigma)), (0, 255, 0), t)
            cv2.line(Img_data,(int(pt[0]),int(pt[1])),(int(pt[0]),int(pt[1])),(0,255,0),3) # 추가

    return Img_data

def show_fidt_func(input):
    input[input < 0] = 0
    input = input[0][0]
    fidt_map1 = input
    fidt_map1 = fidt_map1 / np.max(fidt_map1) * 255
    fidt_map1 = fidt_map1.astype(np.uint8)
    fidt_map1 = cv2.applyColorMap(fidt_map1, 2)
    return fidt_map1 









# DB 연동
from django.http import JsonResponse
from django.views import View
from .models import Foot_Record, Population_Record

def foot_aggregate_people_count_by_hour():
    foot_traffic_records = Foot_Traffic.objects.all()
    aggregated_data = [0] * 24

    for record in foot_traffic_records:
        hour = record.time.hour
        aggregated_data[hour] += record.people_count

    foot_record = Foot_Record.objects.create(
        all_count=sum(aggregated_data),
        time_1=aggregated_data[0],   time_2=aggregated_data[1],   time_3=aggregated_data[2],   time_4=aggregated_data[3],
        time_5=aggregated_data[4],   time_6=aggregated_data[5],   time_7=aggregated_data[6],   time_8=aggregated_data[7],
        time_9=aggregated_data[8],   time_10=aggregated_data[9],  time_11=aggregated_data[10], time_12=aggregated_data[11],
        time_13=aggregated_data[12], time_14=aggregated_data[13], time_15=aggregated_data[14], time_16=aggregated_data[15],
        time_17=aggregated_data[16], time_18=aggregated_data[17], time_19=aggregated_data[18], time_20=aggregated_data[19],
        time_21=aggregated_data[20], time_22=aggregated_data[21], time_23=aggregated_data[22], time_24=aggregated_data[23]
    )
    foot_record.save()

import json
from django.core.exceptions import ObjectDoesNotExist

# 차트 그리기
@xframe_options_exempt
def line_chart(request):
    try:
        foot_record = Foot_Record.objects.latest('count_date')
        times = foot_record.get_values()
    except Foot_Record.DoesNotExist:
        # Foot_Record 객체가 없을 때 times default값 세팅
        times = [0] 
    
    context = {'times': json.dumps(times)}
    return render(request, 'foot_line_chart.html', context)

# 월별 집계
@xframe_options_exempt
def month_line_chart(request):
    try:
        foot_record = Foot_Record.objects.latest('count_date')
        times = foot_record.get_values()
    except Foot_Record.DoesNotExist:
        times = [0]

    monthly_data = aggregate_population_by_month()

    context = {'times': json.dumps(times), 'monthly_data': json.dumps(list(monthly_data))}
    return render(request, 'foot_line_chart_month.html', context)


from django.db.models import Sum

def aggregate_population_by_month():
    monthly_data = Foot_Traffic.objects.values('date__month').annotate(people_count_sum=Sum('people_count')).order_by('date__month')
    return monthly_data

from django.db.models import Avg
import json
# 시간대별 인구 수
@xframe_options_exempt
def crowd_line_chart(request):
    hourly_data = Population_Record.objects.values('time__hour').annotate(people_count_avg=Avg('people_count'))

    context = {'hourly_data': json.dumps(list(hourly_data))}
    return render(request, 'crowd_line_chart.html', context)

# 분대별 인구 수
def get_data(request):
    data = Population_Record.objects.all()
    formatted_data = []

    for record in data:
        time = record.time.strftime('%H:%M')
        people_count = record.people_count
        formatted_data.append({'time': time, 'people_count': people_count})

    return JsonResponse(formatted_data, safe=False)
@xframe_options_exempt
def crowd_line_chart_minute(request):
    return render(request, 'crowd_line_chart_minute.html')

from django.http import HttpResponseRedirect
# 데이터베이스 초기화
# Yolo line chart button
class ResetRecordView(View):
    def get(self, request):
        Foot_Record.objects.all().delete()
        return HttpResponseRedirect('/foot_line_chart/')
# Foot_Traffic 초기화
class ResetTrafficView(View):
    def get(self, request):
        Foot_Traffic.objects.all().delete()
        return HttpResponseRedirect('/foot_line_chart/')
# 차트 갱신
class StoreAggregatedDataView(View):
    def get(self, request):
        foot_aggregate_people_count_by_hour()
        return HttpResponseRedirect('/foot_line_chart/')
    
# Crowd line chart button
# Population_Record 초기화
class ResetPopulationRecordView(View):
    def get(self, request):
        Population_Record.objects.all().delete()
        return HttpResponseRedirect('/crowd_line_chart/')

# 차트 갱신
class CrowdStoreAggregatedDataView(View):
    def get(self, request):
        return HttpResponseRedirect('/crowd_line_chart/')
    

def index(request):
    return render(request, 'index.html')

def stream_traffic(request):
    return render(request,'stream_traffic.html')

def stream_crowd(request):
    return render(request,'stream_crowd.html')

def home_foot_traffic(request):
    return render(request, 'home_foot_traffic.html')

def home_crowd_estimation(request):
    return render(request, 'home_crowd_estimation.html')

def crowd_analysis(request):
    return render(request, 'crowd_analysis.html')

def foot_analysis(request):
    return render(request, 'foot_analysis.html')
#===========================================================================================