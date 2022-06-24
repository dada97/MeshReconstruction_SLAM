import os
import sys
import msgpack
import json
import io
import numpy as np


from autolab_core import RigidTransform

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
        f.write("key_frame_id,pos_x,pos_y,pos_z,rot_x,rot_y,rot_z,rot_w,timestamp\n")
        for key in sorted(key_frame.keys()):
            
            point = key_frame[key]
            #print(point[])
            trans_cw = np.asarray(point["trans_cw"])
            rot_cw = np.asarray(point["rot_cw"])
            ts = point["ts"] 
            rigid_cw = RigidTransform(rot_cw, trans_cw)
            pos = np.matmul(rigid_cw.rotation, trans_cw)

            f.write("{},{},{},{},{},{},{},{},{}\n".format(key,pos[0], pos[1], pos[2],rot_cw[0],rot_cw[1],rot_cw[2],rot_cw[3],ts),)
    print("Done")

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