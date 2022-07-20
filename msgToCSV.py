

import os
import sys
import msgpack
import numpy as np

from scipy.spatial.transform import Rotation as R
def main(bin_fn, dest_fn):

    output = os.path.join(dest_fn, "slamData.csv");
    print("output file :" + output)

    # Read file as binary and unpack data using MessagePack library
    with open(bin_fn, "rb") as f:
        data = msgpack.unpackb(f.read(), use_list=False, raw=False)

    # The point data is tagged "landmarks"
    key_frames = data["keyframes"]
    print("keyframes has {} frames.".format(len(key_frames)))

    key_frame = {int(k): v for k, v in key_frames.items()}

    with open(output, "w") as f:
        f.write(",key_frame_id,pos_x,pos_y,pos_z,rot_x,rot_y,rot_z,rot_w,timestamp\n")
        count = 0;
        for key in sorted(key_frame.keys()):
            count +=1
            point = key_frame[key]
            #print(point[])

            rot_cw = np.asarray(point["rot_cw"])
            ts = point["ts"] 

            # get conversion from camera to world
            trans_cw = np.matrix(point["trans_cw"]).T
            rot = R.from_quat(point["rot_cw"]).as_matrix()
            
            # compute conversion from world to camera
            rot_wc = rot.T
            trans_wc = - rot_wc * trans_cw

            f.write("{},{},{},{},{},{},{},{},{},{}\n".format(count,key,trans_wc[0, 0], trans_wc[1, 0],-trans_wc[2, 0],rot_cw[0],rot_cw[1],rot_cw[2],rot_cw[3],ts),)
    print("Done")

    output = os.path.join(dest_fn, "landmarks.csv");
    key_frames = data["landmarks"]
    print("keyframes has {} frames.".format(len(key_frames)))
    key_frame = {int(k): v for k, v in key_frames.items()}

    with open(output, "w") as f:
        f.write("1st_keyfrm,n_fnd,n_vis,pos_x,pos_y,pos_z,ref_keyfrm\n")
        for key in sorted(key_frame.keys()):
            point = key_frame[key]
            pos_w = point['pos_w']
            f.write("{},{},{},{},{},{},{}\n".format(point["1st_keyfrm"],point["n_fnd"],point["n_vis"],pos_w[0],pos_w[1],pos_w[2],point["ref_keyfrm"]))



if __name__ == "__main__":
    argv = sys.argv

    if len(argv) < 3:
        print("Unpack all slam in the map file and dump into a csv file")
        print("Usage: ")
        print("    python msgToCSV.py [map file] [csv destination]")

    else:
        bin_fn = argv[1]
        dest_fn = argv[2]
        main(bin_fn, dest_fn)