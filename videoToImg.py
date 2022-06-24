import cv2
import sys
import os

def main(bin_fn, dest_fn):
    os.makedirs(dest_fn, exist_ok=True)
    vidcap = cv2.VideoCapture(bin_fn)
    fps = vidcap.get(cv2.CAP_PROP_FPS)
    print("fps :"+str(fps))
    i=0
    while(vidcap.isOpened()):
        ret, frame = vidcap.read()
        if ret == False:
            break
        frame = cv2.resize(frame,(2000,1000))
        print(i)
        output = os.path.join(dest_fn, str(i).zfill(5)+'.jpg');
        cv2.imwrite(output,frame)
        i+=1

    
if __name__ == "__main__":
    argv = sys.argv
    if len(argv) < 3:
        print("Extract image sequence from video")
        print("Usage: ")
        print("    python video.py [video path] [output destination]")
    bin_fn = argv[1]
    dest_fn = argv[2]

    main(bin_fn,dest_fn)
