#include"csv_parser.h"


CSV_parser::CSV_parser(string slampath) {
	read_slam(slampath);
}

void CSV_parser::read_slam(string path) {
	ifstream fin(path);
	string output;
	fin >> output;
	while (true) {
		fin >> output;
		if (fin.eof()) break;
		std::stringstream ss(output);
		vector<float> vect;
		for (float i; ss >> i;) {
			vect.push_back(i);
			if (ss.peek() == ',')
				ss.ignore();
		}

		Slam_data data;
		data.key_frame_id = vect[1];
		data.position.x = vect[2];
		data.position.y = vect[3];
		data.position.z = vect[4];

		data.quaternion.x = vect[5];
		data.quaternion.y = vect[6];
		data.quaternion.z = vect[7];
		data.quaternion.w = vect[8];

		data.timestamp = vect[9];

		slam_data.push_back(data);
	}
}

void CSV_parser::read_landmarks(string path) {
	ifstream fin(path);
	string output;
	fin >> output;
	while (true) {
		fin >> output;
		if (fin.eof()) break;
		std::stringstream ss(output);
		vector<float> vect;
		for (float i; ss >> i;) {
			vect.push_back(i);
			if (ss.peek() == ',')
				ss.ignore();
		}
		Landmark_data data;
		data.ref_keyframe = vect[6];
		data.position.x = vect[3];
		data.position.y = vect[4];
		data.position.z = vect[5];

		landmarks_data.push_back(data);
	}
}

vector<Slam_data> CSV_parser::get_slamdata() {
	return slam_data;
}
vector<Landmark_data> CSV_parser::get_landmarksdata() {
	return landmarks_data;
}
