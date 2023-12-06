import cv2
import sys
sys.path.append('../main/')
import pandas as pd 
import torch
import torchvision.transforms as T
import torch.nn.functional as F
import os
import mediapipe as mp
import numpy as np
from queue import Queue
import pathlib
import argparse
from inf_utils import *
from model.model import *


def main(model_choice, weights):
    temp = pathlib.PosixPath
    pathlib.PosixPath = pathlib.WindowsPath
    device = 'cuda' if torch.cuda.is_available() else "cpu"
    print(f"Device : {device}")

    model = model_choice["model"]()
    input_type = model_choice["input_type"]

    try:
        model = model.to(device)
        checkpoint = torch.load(weights, map_location=torch.device('cpu'))
        state_dict = checkpoint['state_dict']
        model.load_state_dict(state_dict)
    except Exception as e:
        print(e)
        return


    mp_drawing = mp.solutions.drawing_utils
    mp_hands = mp.solutions.hands

    cap = cv2.VideoCapture(0)
    isCapturing = False

    keypoints_queue = Queue() 
    clip_q = Queue() 

    clip = []
    words = []
    threshold = 0.2 
    # initialize dictionary of dummy keypoint values as previous keypoint values
    prev_dict = init_hands() 
    lookup = pd.read_csv('dictionary.csv')

    with mp_hands.Hands(min_detection_confidence=0.8, min_tracking_confidence=0.5) as hands: 
        try:
            while cap.isOpened():
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                transforms = prepare_transforms(height)

                ret, frame = cap.read()

                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image = cv2.flip(image, 1)

                image.flags.writeable = False
                results = hands.process(image)

                # Set flag to true
                image.flags.writeable = True
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

                # Rendering results
                if results.multi_hand_landmarks:
                    landmarks_dict = {'Right': None, 'Left': None}
                    for num, hand in enumerate(results.multi_hand_landmarks):
                        hand_type = results.multi_handedness[num].classification[0].label

                        mp_drawing.draw_landmarks(image, hand, mp_hands.HAND_CONNECTIONS, 
                                                mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=4),
                                                mp_drawing.DrawingSpec(color=(250, 44, 250), thickness=2, circle_radius=2),
                                                )
                        
                        landmarks_dict[hand_type] = processKeypoints(hand)
                        prev_dict[hand_type] = landmarks_dict[hand_type]

                    if not keypoints_queue.empty():
                        total_disp = calculate_displacements(landmarks_dict, prev_dict, keypoints_queue)
                        if total_disp > threshold: 
                            isCapturing = True

                        cv2.putText(image, f'{total_disp:.3f}', (0,100), 1, 2, (255,0,0), 1, cv2.LINE_AA)
                    keypoints_queue.put(landmarks_dict) 
                    
                if isCapturing:
                    for hand in ['Right', 'Left']:
                        if landmarks_dict[hand] is None:
                            landmarks_dict[hand] = prev_dict[hand]

                    if input_type == "RGB":
                        clip.append(transforms(Image.fromarray(frame)))
                    elif input_type == "keypoint":                
                        clip.append(landmarks_dict)

                    cv2.putText(image, f'CAPTURING: {len(clip):02}', (0,50), 1, 2, (0,0,255), 3, cv2.LINE_AA)

                if len(clip) > 63:  
                    clip_tensor = process_clip(clip, device, input_type)
                    outputs = get_model_outputs(model, clip_tensor)
                    
                    word, words, values, indices = process_outputs(lookup, outputs)
                    
                    print("Top prediction:", word)
                    print("Top 5:", words)
                    print("====Result====")
                    print(values, indices)
                    
                    clip_q.put(clip)
                    clip = []
                    isCapturing = False

                if words:
                    result = f'{words[0]} ({values[0][0].item()*100:.2f}%)'
                    image = displayResult(result, image, 400)

                cv2.imshow('Sign Language Translator', image)
                if cv2.waitKey(10) & 0xFF == ord('q'):
                    break
                
        except Exception as e:
            print(e)
            cap.release()
            cv2.destroyAllWindows()
            
    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Real-time Demo')
    parser.add_argument('-m', '--model', default="Conv1D-LSTM", type=str, help='model to use')
    parser.add_argument('-w', '--weight', default="best.pth", type=str, help='path to  weight to test (default: None)')
    args = parser.parse_args()
    
    model_choice = {
        "I3D": {"model": I3D, "input_type": "RGB"},
        "P3D": {"model": P3D, "input_type": "RGB"},
        "Conv1D-LSTM": {"model": Conv1d_LSTM, "input_type": "keypoint"},
        # "CNN-LSTM": CNNLSTM
    }
    
    assert args.model in model_choice, "Unsupported model specified."
    model = model_choice[args.model]
    assert os.path.exists(args.weight), "path to weights does not exist!"
    main(model, args.weight)
