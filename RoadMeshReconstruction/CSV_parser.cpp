#include"csv_parser.h"


CSV_parser::CSV_parser(string slampath,string landmarkspath) {
	read_slam(slampath);
	read_landmarks(landmarkspath);
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
		data.key_frame_id = vect[0];
		data.position.x = vect[1];
		data.position.y = vect[2];
		data.position.z = vect[3];

		data.quaternion.x = vect[4];
		data.quaternion.y = vect[5];
		data.quaternion.z = vect[6];
		data.quaternion.w = vect[7];

		data.timestamp = vect[8];

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
