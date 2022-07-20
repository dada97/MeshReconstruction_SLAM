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
	CSV_parser(string slampath);
	vector<Slam_data>  get_slamdata();

	void read_landmarks(string path);
	vector<Landmark_data> get_landmarksdata();

private:
	void read_slam(string path);		//parse slam data from csv file
	

	vector<Slam_data> slam_data;	
	vector<Landmark_data> landmarks_data;

};