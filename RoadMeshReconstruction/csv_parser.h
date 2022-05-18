#include<iostream>
#include <fstream>
#include <vector>
#include <sstream>
using namespace std;

struct Position {
	float x;
	float y;
	float z;
};

struct Quarternion {
	float x;
	float y;
	float z;
	float w;
};

struct Slam_data {
	int key_frame_id;
	Position position;
	Quarternion quaternion;
	float timestamp;
};

struct Landmark_data {
	int ref_keyframe;
	Position position;
};

class CSV_parser {

public:
	CSV_parser() {};
	CSV_parser(string slampath,string landmarkspath);
	vector<Slam_data>  get_slamdata();
	vector<Landmark_data> get_landmarksdata();

private:
	void read_slam(string path);
	void read_landmarks(string path);

	vector<Slam_data> slam_data;
	vector<Landmark_data> landmarks_data;

};