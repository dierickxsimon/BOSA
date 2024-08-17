import cv2
import pandas as pd
import numpy as np
import math

video = "Exchange 1.mp4"


output = {
    #'non_zero':[],
    'x':[],
    'y':[],
    'w':[],
    'h':[],
    'frame_id':[]
}

output_pixels = {
    'x':[],
    'y':[],
    'frame_id':[]
}


lines = pd.read_csv('output.csv')

raw_mask = cv2.imread('cycling_masks.png')
mask = cv2.cvtColor(raw_mask, cv2.COLOR_BGR2GRAY) 
_, mask = cv2.threshold(mask, 128, 255, cv2.THRESH_BINARY)

# Convert the image to grayscale
gray_image = cv2.cvtColor(raw_mask, cv2.COLOR_BGR2GRAY)

# Create a 3-channel black and white image
raw_image = cv2.merge([gray_image, gray_image, gray_image])



backSub = cv2.createBackgroundSubtractorMOG2()


cv2.namedWindow("Frame", cv2.WINDOW_NORMAL)
cv2.namedWindow("FG Mask", cv2.WINDOW_NORMAL)



frame_id = 0
cropped_id = 0

capture = cv2.VideoCapture(cv2.samples.findFileOrKeep(video))





if not capture.isOpened():
    print('Unable to open: ' + video)
    exit(0)
while True:
    ret, frame = capture.read()
    if frame is None:
        break
    
    
    
    fgMask = backSub.apply(frame)
    final_img = cv2.bitwise_and(fgMask, mask)
    final_img = final_img[:final_img.shape[0]//2,:]
    #frame = cv2.bitwise_and(frame, raw_mask)
    frame = frame[:frame.shape[0]//2,:]
    
    
    
    #cv2.rectangle(frame, (10, 2), (100,20), (255,255,255), -1)
    #cv2.putText(frame, str(capture.get(cv2.CAP_PROP_POS_FRAMES)), (15, 15),
     #          cv2.FONT_HERSHEY_SIMPLEX, 0.5 , (0,0,0))
    
    #adding blobs to the fgmask
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(final_img, connectivity=4)
    num_cycling_blobs = 0
    cycling_blobs = {
        'x':[],
        'y':[],
        'w':[],
        'h':[],
    }
    
    
    
    # loop over all blobs and decide which ones are bikes and which are not based on size
    cycling_blobs_tmp = []
    for i in range(1, num_labels):
        x = stats[i, cv2.CC_STAT_LEFT]
        y = stats[i, cv2.CC_STAT_TOP]
        w = stats[i, cv2.CC_STAT_WIDTH]
        h = stats[i, cv2.CC_STAT_HEIGHT]

        # filter the balls that have the right size using thresholds
        if (w > 50 and w < 240 and h > 30 and x > 0 and x < len(final_img[0]) and y > 0 and y < len(final_img)):
            
            #cv2.circle(frame, (x,y),10, color=(0, 255, 0))
            num_cycling_blobs += 1
            cycling_blobs['x'].append(x)
            cycling_blobs['y'].append(y)
            cycling_blobs['w'].append(w)
            cycling_blobs['h'].append(min(h, 75))
            cycling_blobs_tmp.append({'x':x, 'y':y,'w':w,'h':min(h, 75)})
            
            output['x'].append(x)
            output['y'].append(y)
            output['w'].append(w)
            output['h'].append(h)
            output['frame_id'].append(frame_id)
    
    
    blob_to_find = []
    for blob_1 in cycling_blobs_tmp:
        max_area = 0
        blobs_to_merge = None
        blob = None
        for blob_2 in cycling_blobs_tmp:
            if blob_1 == blob_2:
                continue
               # Get coordinates of intersection rectangle
            k = 1.1
            x_left = max(blob_1['x'], blob_2['x'])
            y_top = max(blob_1['y'], blob_2['y']) 
            #x_right = min(blob_1['x'] + blob_1['w'], blob_2['x'] + blob_2['w']) * k
            x_right = min(blob_1['x'] + 25, blob_2['x'] + 25) 

            y_bottom = min(blob_1['y'] + blob_1['h'], blob_2['y'] + blob_2['h']) *k
    
            # Calculate width and height of intersection rectangle
            width = max(0, x_right - x_left)
            height = max(0, y_bottom - y_top) 
    
   
            area = width * height
            
            if area > max_area:
                x_left = min(blob_1['x'], blob_2['x'])
                y_top = min(blob_1['y'], blob_2['y'])
                x_right = max(blob_1['x'] + blob_1['w'], blob_2['x'] + blob_2['w']) 
                y_bottom = max(blob_1['y'] + blob_1['h'], blob_2['y'] + blob_2['h']) 
    
                
                max_area = area
                #blob = (x_left, y_top, x_right, y_bottom)
                blob = {'x': int(x_left), 'y': int(y_top), 'w': int(x_right-x_left), 'h':int(y_bottom-y_top), 'color': (0, 255, 0)}
                blobs_to_merge = (blob_1,blob_2)
                
            
        print(max_area, blobs_to_merge)
        blob_to_find.append({'blob':blob, 'blobs_to_merge': blobs_to_merge, 'max_area': max_area})
    
    for blob_1 in blob_to_find:    
        for i, blob_2 in enumerate(cycling_blobs_tmp):
            if blob_1['max_area']<1:
                continue
            if blob_1['blobs_to_merge'][0]== blob_2:
                cycling_blobs_tmp[i] = blob_1['blob']
                print('iets')
                
            if blob_1['blobs_to_merge'][1] == blob_2:
                cycling_blobs_tmp[i] = blob_1['blob']
                print('blob_2 replaced')
    
    cycling_blobs = {
        'x':[],
        'y':[],
        'w':[],
        'h':[],
    }
           
    for blob in cycling_blobs_tmp:
        #cycling_blobs['x'].append(blob['x'])
        #cycling_blobs['y'].append(blob['y'])
        #cycling_blobs['w'].append(blob['w'])
        #cycling_blobs['h'].append(blob['h'])
        print(blob)
        cv2.rectangle(frame, (blob['x'], blob['y']), (blob['x'] + blob['w'], blob['y'] + blob['h']), blob['color'] if 'color' in blob else (255, 0, 0), 10)
            
        
       
    #if len(output['x']) >2:       
        #for x, y in zip(output['x'], output['y']):
            #cv2.circle(frame, (x, y), radius=4, color=(0, 0, 255), thickness=-1)   
         
            
    cropped_frame = False
    pre_frame = False
    if len(cycling_blobs['x']) > 2:
        
        
                    
        
        
        x_min = max(min(cycling_blobs['x']) - 50, 0)
        y_min = min(cycling_blobs['y']) - 30
        x_max = min(max(cycling_blobs['x']) + max(cycling_blobs['w']) + 30, frame.shape[1])
        y_max = max(cycling_blobs['y'] + max(cycling_blobs['h']) + 30)
        
        if (x_max - x_min) >= 640:
            x_diff = (x_max - x_min) - 640
            x_max -= x_diff 
            x_min += math.ceil(x_diff / 2)

        # Adjust height if it exceeds 640 pixels
        if (y_max - y_min) >= 640:
            y_diff = (y_max - y_min) - 640
            y_max -= math.floor(y_diff / 2)
            y_min += math.ceil(y_diff / 2)
        
        cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 0, 255), thickness=15)

        
        
        '''
        x_min = int(np.average(cycling_blobs['x'])) - 320
        y_min = int(np.average(cycling_blobs['y'])) - 320
        x_max = int(np.average(cycling_blobs['x'])) + 320
        y_max = int(np.average(cycling_blobs['y'])) + 320
        
        if x_min <0:
            x_max = abs(x_min) + x_max 
            x_min = 0
        if y_min<0:
            y_max = abs(y_min) + y_max
            y_min = 0 
            
        '''
       
        
    
        cropped_frame = frame[y_min:y_max, x_min:x_max]
        output_pixels['x'].append(x_min)
        output_pixels['y'].append(y_min)
        output_pixels['frame_id'].append(cropped_id)
        
        
        
        #cv2.imwrite(f'output/{cropped_id}.png', cropped_frame)
    
        
        cropped_id +=1
        
        
    
    elif pre_frame:
        cropped_frame = pre_frame
    
    else:
        cropped_frame = frame
        pre_frame = frame
        
 
    cv2.imshow('Frame', frame)
    cv2.imshow('FG Mask', cropped_frame)

    #non_zero = cv2.countNonZero(final_img)
    #output['non_zero'].append(non_zero)
    
    
    

    
    


    keyboard = cv2.waitKey(30)
    if keyboard == 'q' or keyboard == 27:
        break
    if keyboard == 115:
        cv2.imwrite("input.png", frame)
        cv2.imwrite("mask.png", final_img)

    frame_id += 1


# save data

df = pd.DataFrame(output)
#df.to_csv("output.csv")
cv2.destroyAllWindows() 
   
print("The video was successfully saved") 


df = pd.DataFrame(output_pixels)
df.to_csv('output_coordinates.csv')