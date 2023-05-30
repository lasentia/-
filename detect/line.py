import cv2

# 좌표 임시 저장
clicked_points = [] 

# 최종 선의 좌표
a = [] 

class draw_line:

    def MouseLeftClick(event, x, y, flags, param):
        # 왼쪽 마우스가 클릭되면 (x, y) 좌표를 저장한다.
        global clicked_points
        global a
        
        temp = param.copy()
        if event == cv2.EVENT_LBUTTONDOWN:
            clicked_points.append((x, y))
            if len(clicked_points) == 2:
                param = temp.copy()
                cv2.line(param, (clicked_points[0][0], clicked_points[0][1]), (clicked_points[1][0], clicked_points[1][1]), (0, 255, 0), 2)
                cv2.imshow("draw", param)
                a = clicked_points.copy()
                print(a)
                clicked_points = []

    def xy(img):
        global a
        cv2.imshow("draw", img)
        cv2.setMouseCallback("draw", draw_line.MouseLeftClick, img)
        cv2.waitKey()
        return a[0][0], a[0][1], a[1][0], a[1][1]
        