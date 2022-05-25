#include "RoadMeshReconstruction.h"


void RoadMeshReconstruction::init() {
	string inputDir = "D:/project/Stitching/dataset/insta360/0523_2";
	
	input_dir = inputDir;
	int count = 0;
	for (const auto& entry : fs::directory_iterator(input_dir + "/rgb")) {
		count++;
		string filename = entry.path().filename().string();
		size_t lastindex = filename.find_last_of(".");
		filename = filename.substr(0, lastindex);
		frame_name.push_back(filename);
	}

	total_frame = count;
	cout << "Total number frame: " << count << endl;

	string outputDir = "./output/";
	cv::utils::fs::createDirectory(outputDir);

	output_dir = cv::utils::fs::join(outputDir, "case2_mesh");
	cv::utils::fs::createDirectory(output_dir);

	string slam_path = input_dir + "/openslam.csv";
	string landmarks_path = input_dir + "/landmarks.csv";
	string file_contents;
	std::map<int, std::vector<string>> csv_contents;
	char delimiter = ',';

	CSV_parser csv_parser(slam_path,landmarks_path);

	slam_data = csv_parser.get_slamdata();
	landmarks_data = csv_parser.get_landmarksdata();

}

void RoadMeshReconstruction::readData(int index) {
	cout << "read Data : " << frame_name[index] << endl;
	string depth_path = input_dir + "/depth/" + frame_name[index] + "_disp.npy";
	data.depth_arry = cnpy::npy_load(depth_path);
	data.height = data.depth_arry.shape[2];
	data.width = data.depth_arry.shape[3];

	string seg_path = input_dir + "/seg/" + frame_name[index] + "_prediction.png";
	cv::Mat seg = cv::imread(seg_path, CV_LOAD_IMAGE_COLOR);
	data.seg_ori = seg;

	string rgb_path = input_dir + "/rgb/" + frame_name[index] + ".jpg";
	cv::Mat rgb = cv::imread(rgb_path, CV_LOAD_IMAGE_UNCHANGED);
	data.rgb_ori = rgb;
	rgb = rgb(cv::Rect(0, int(rgb.rows / 4), rgb.cols, (rgb.rows / 2)));


	cv::resize(seg, seg, cv::Size(data.width, data.height));
	cv::resize(rgb, rgb, cv::Size(data.width, data.height));

	data.seg = seg;
	data.rgb = rgb;
}

void RoadMeshReconstruction::findRoadContours() {


	//roud contours
	cv::Mat road;
	cv::Mat roadline;
	cv::Mat other;

	cv::inRange(data.seg_ori, cv::Vec3b(128, 64, 128), cv::Vec3b(128, 64, 128), road);
	cv::inRange(data.seg_ori, cv::Vec3b(255, 255, 255), cv::Vec3b(255, 255, 255), roadline);
	cv::inRange(data.seg_ori, cv::Vec3b(96, 96, 96), cv::Vec3b(96, 96, 96), other);
	roadline.copyTo(road, roadline);
	other.copyTo(road, other);

	cv::Mat element = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(10, 10));
	cv::dilate(road, road, element);
	cv::erode(road, road, element);

	vector<vector<cv::Point> > contours;
	vector<cv::Vec4i> hierarchy;

	cv::findContours(road, contours, hierarchy, cv::RETR_TREE, cv::CHAIN_APPROX_SIMPLE);

	cv::Mat filled = cv::Mat::zeros(data.seg_ori.size(), CV_8UC1);


	int id = 0;
	float max_area = 0;
	for (size_t i = 0; i < contours.size(); i++)
	{
		float area = contourArea(contours[i], false);
		if (area > max_area) {
			id = i;
			max_area = area;

		}
	}

	cv::drawContours(filled, contours, id, 255, -1, cv::LINE_8, hierarchy, CV_FILLED);

	roadMask = filled;
	data.road_fill = filled;
}

void RoadMeshReconstruction::calculateCameraHeight() {
	cout << "build pointcloud!" << endl;
	float* depth_data = data.depth_arry.data<float>();
	std::vector<Point_3> points_3;

	cv::Mat mask = roadMask;
	cv::resize(mask, mask, cv::Size(data.width, data.height));


	for (int y = 0; y < data.height; y++) {
		for (int x = 0; x < data.width; x++) {

			//int segColor = roadMask.at<uchar>(y, x);
		
			if ((int)mask.at<uchar>(y, x) == 255) {

				int idx = y * data.rgb.cols + x;


				float depth = 1 / (depth_data[idx]);
				float a = ((float)x / (float)(data.rgb.cols - 1)) * (2 * M_PI);
				float b = ((float)y / (float)(data.rgb.rows - 1)) * (M_PI) * 0.5 - (0.25 * M_PI);

				float px = -depth * cos(a) * cos(b);
				float py = depth * sin(a) * cos(b);
				float pz = -depth * sin(b);

				cv::Vec3b color = data.rgb.at<cv::Vec3b>(y, x);

				points_3.push_back(Point_3(px,py,pz));

			}
		}
	}
	//plane_coefficients = fitPlane();
	Plane plane;
	Point_3 centeroid;
	linear_least_squares_fitting_3(points_3.begin(), points_3.end(),plane, centeroid, CGAL::Dimension_tag<0>());

	if (current_frame == 0) {
		prevHeight = plane.d();
		camHeight = abs(plane.d());
	}
	else {
		camHeight = prevHeight;
		/*camHeight = plane.d();
		if (abs(camHeight - prevHeight) < 0.5) {
			camHeight = prevHeight;
		}
		else {
	
			if (camHeight-prevHeight<0) {
				camHeight = prevHeight - 0.05;
			}
			else{
				camHeight = prevHeight + 0.05;
			}

			prevHeight = camHeight;
		}*/
	}


	cout << "camHeight: " <<camHeight << endl;
}

std::pair<float, float>  RoadMeshReconstruction::projectToUV(float x, float y, float z) {
	float u, v;
	float phi = atan(sqrt(x * x + y * y) / z);
	u = phi / M_PI;


	float alt;
	if (x > 0) {
		alt = atan(y / x);
	}
	else if (x < 0 && y >= 0) {
		alt = atan(y / x) + M_PI;
	}
	else if (x < 0 && y < 0) {
		alt = atan(y / x) - M_PI;
	}
	else if (x == 0 && y > 0) {
		alt = M_PI / 2;
	}
	else if (x == 0 && y < 0) {
		alt = -M_PI / 2;
	}



	v = -alt / (2 * M_PI);
	if (u < 0) {
		u = 1 + u;
	}

	v = v + 0.5;
	if (v > 1) {
		v = v - 1;
	}

	return std::make_pair(u, v);
}

void RoadMeshReconstruction::outputOBJ() {

	//dt.insert(vertices.begin(), vertices.end());

	string output_path = cv::utils::fs::join(output_dir, "outputMesh.obj");

	cout << output_path << endl;

	ofstream myfile;
	myfile.open(output_path);
	std::map<Mesh::Vertex_index, unsigned> vertex_map;
	unsigned count = 1;
	
	cout << dt.number_of_vertices() << endl;

	// Loop over vertices
	for (Mesh::Vertex_index vi : m.vertices()) {
		K::Point_3 pt = m.point(vi);
		vertex_map[vi] = count;
		++count;
		myfile << "v " << pt << "\n";
	}
	
	
	myfile << "\n";
	// Map over facets. Each facet is a cell of the underlying
	// Delaunay triangulation, and the vertex that is not part of
	// this facet. We iterate all vertices of the cell except the one
	// that is opposite.
	for (Mesh::Face_index face_index : m.faces()) {
		myfile << "f";
		Mesh::Halfedge_index hf = m.halfedge(face_index);
		for (Mesh::Halfedge_index hi : halfedges_around_face(hf, m))
		{
			Mesh::Vertex_index vi = target(hi, m);
			myfile << " " << vertex_map[vi];
		}
		myfile << std::endl;
	}

	myfile.close();
}

void RoadMeshReconstruction::outputDelaunay() {
	ofstream ofmesh;
	ofmesh.open("delaunay.obj");
	ofmesh << "\n";
	std::map < CDT::Vertex_handle, unsigned > vm;
	unsigned ccount = 1;


	cout << dt.number_of_vertices() << endl;

	// Loop over vertices
	for (auto it = dt.vertices_begin(); it != dt.vertices_end(); it++) {
		vm[it] = ccount;
		ofmesh << "v " << it->point() << " 0 " << std::endl;
		++ccount;
	}

	// Map over facets. Each facet is a cell of the underlying
	// Delaunay triangulation, and the vertex that is not part of
	// this facet. We iterate all vertices of the cell except the one
	// that is opposite.
	for (auto it = dt.faces_begin(); it != dt.faces_end(); it++) {
		ofmesh << "f";
		ofmesh << " " << vm[it->vertex(0)] << " " << vm[it->vertex(1)] << " " << vm[it->vertex(2)];
		ofmesh << std::endl;
	}
	ofmesh.close();

}

std::vector<std::vector<cv::Point>> RoadMeshReconstruction::findPointCloudContours(int min_x, int max_x, int min_y, int max_y) {
	float w = max_x - min_x;
	float h = max_y - min_y;
	cnt_w = w;
	cnt_h = h;
	cout << w << " " << h << endl;


	//Find contours
	cv::Mat test = cv::Mat::zeros(cv::Size(w * 15 + 10, h * 15 + 10), CV_8UC1);
	cout << test.size() << endl;
	for (int i = 0; i < pc.size(); i++) {
		Point_3 pt = pc[i];
		float u = (pt.hx() - (min_x)) / (w);
		float v = (pt.hy() - (min_y)) / (h);

		if (u >= 0 && u < 1.0 && v >= 0 && v < 1.0)
			test.at<uchar>(v * (h * 15 + 5), u * (w * 15 + 5)) = 255;

	}

	cout << "contours" << endl;
	cv::imwrite("before.jpg", test);
	cv::Mat tmp;
	test.copyTo(tmp);
	cv::GaussianBlur(test, test, cv::Size(5, 5), 0);
	cv::morphologyEx(test, test, cv::MORPH_OPEN, cv::getStructuringElement(cv::MORPH_RECT, cv::Size(10, 10.0)));
	cv::inRange(test, 150, 255, test);
	/*for (int i = 0; i < 7; i++) {
		cv::Mat element = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));
		cv::dilate(test, test, element);


		cv::erode(test, test, element);
		cv::inRange(test, 150, 255, test);
	}*/


	test(cv::Rect(5, 5, w * 15, h * 15)).copyTo(test);


	vector<vector<cv::Point> > contours;
	vector<cv::Vec4i> hierarchy;


	cv::findContours(test, contours, hierarchy, cv::RETR_CCOMP, cv::CHAIN_APPROX_SIMPLE);

	int id = 0;
	float max_area = 0;
	cout << "size : " << contours.size() << endl;
	cv::Mat drawing = cv::Mat::zeros(test.size(), CV_8UC1);; //=


	for (size_t i = 0; i < contours.size(); i++)
	{
		float area = contourArea(contours[i], false);
		cout << hierarchy[i] << endl;
		if (area > max_area) {
			id = i;
			max_area = area;

		}
		drawContours(drawing, contours, i, (255), 1, cv::LINE_8, hierarchy, 0);
	}
	//cv::Mat smoothCont = cv::Mat::zeros(cv::Size(w * 40, h * 40), CV_8UC1);
	//cv::Mat cnt = cv::Mat::zeros(cv::Size(w * 15, h * 15), CV_8UC1);


	cv::imwrite("testimage.jpg", test);
	cv::imwrite("drawing.jpg", drawing);
	return contours;
}

void RoadMeshReconstruction::analyzeLandmarks() {

	//myfile.open("test.obj");
	std::map<int, int> keyframe_map;
	for (int i = 0; i < slam_data.size(); i++) {
		int keyframe = slam_data[i].key_frame_id;
		keyframe_map[keyframe] = i;

	}
	// Loop over vertices
	int currentid = -1;

	cv::Mat seg_img;
	cv::Mat road;
	cv::Mat roadline;
	cv::Mat rgb;
	ofstream myfile;
	myfile.open("testoutput.obj");
	for (int i = 0; i < landmarks_data.size(); i++) {
		float x = landmarks_data[i].position.x;
		float y = landmarks_data[i].position.y;
		float z = landmarks_data[i].position.z;

		auto iter = keyframe_map.find(landmarks_data[i].ref_keyframe);

		if (y >= -0.5 && y < 0.5) {
			testPoints.push_back(Point_3(x, z, -y));
		}

	}
	// Loop over vertices

	Point_3 centeroid;
	linear_least_squares_fitting_3(testPoints.begin(), testPoints.end(), testplane, centeroid, CGAL::Dimension_tag<0>());
	cout << "plane :" << testplane.a() <<" "<< testplane.b() << " " << testplane.c() << " " << testplane.d();

	for (int i = 0; i < testPoints.size();i++) {
		Point_3 pt = testPoints[i];

		myfile << "v " << pt << "\n";
	}
	myfile.close();

	ofstream myfile2;
	myfile2.open("testplane.obj");
	for (int x = 0; x < 100; x++) {
		for (int y = 0; y < 100; y++) {

			int min = -4;
			int max = 4;

			float idx = ((float)x / 100)*(max - min)+min;
			float idy = ((float)y / 100)*(max - min)+min;

			//ax+by+cz+d=0;
			float a = testplane.a();
			float b = testplane.b();
			float c = testplane.c();
			float d = testplane.d();
			float z = (-d - a * idx - b * idy) / c;
			myfile2 << "v " << idx<<" "<<idy<<" "<<z << "\n";
		}
	}
	myfile2.close();

		//if (iter != keyframe_map.end()) {
		//	Slam_data slamdata = slam_data[iter->second];
		//	int frame = round(slamdata.timestamp * fps);

	
		//	cout << frame << endl;
		//	cout << "read Data : " << frame_name[frame] << endl;
		//	if (currentid != frame) {
		//		string rgbpath = input_dir + "/rgb/" + frame_name[frame] + ".jpg";

		//		//rgb = cv::imread(rgbpath, CV_LOAD_IMAGE_COLOR);
		//		

		//		string seg_path = input_dir + "/seg/" + frame_name[frame] + "_prediction.png";
		//		cout << seg_path << endl;
		//		seg_img = cv::imread(seg_path, CV_LOAD_IMAGE_COLOR);
		//		currentid = frame;
		//		cv::inRange(seg_img, cv::Vec3b(128, 64, 128), cv::Vec3b(128, 64, 128), road);
		//		cv::inRange(seg_img, cv::Vec3b(255, 255, 255), cv::Vec3b(255, 255, 255), roadline);

		//		roadline.copyTo(road, roadline);
		//	}
		//	cout << "here" << endl;
		//	Eigen::Quaterniond quaternion(slamdata.quaternion.w,slamdata.quaternion.x,slamdata.quaternion.y,slamdata.quaternion.z);


		//	float local_x = landmarks_data[i].position.x - slamdata.position.x;
		//	float local_y = landmarks_data[i].position.y - slamdata.position.y;
		//	float local_z = landmarks_data[i].position.z - slamdata.position.z;
		//	Eigen::Vector3d point(local_x, local_y, local_z);
		//	point = quaternion.inverse() * point;

		//	std::pair<float, float>uv = projectToUV(point.x(), point.z(), point.y());

		//		int u_y = (uv.first - 0.25) / 0.5 * (seg_img.rows);
		//		int v_x = uv.second * (seg_img.cols);

		//		cout << "here" << endl;

		//		//uchar mask = road.at<uchar>(u_y, v_x);
		//		cv::Vec3d color;
		//		cout << uv.first << endl;
		//		if (uv.first >= 0.25 && uv.first < 0.75) {
		//			cout << u_y << " " << v_x << endl;

		//			color = seg_img.at<cv::Vec3b>(u_y, v_x);
		//		}
		//		else {

		//			color = cv::Vec3b(255,255,255);
		//		}
		//		cout << color << endl;
		//		//if (y>=-2||y<=2)
		//		myfile << x << " " << y << " " << z << " " << color[2] << " " << color[1] << " " << color[0] << "\n";
		//
		//	/*if (mask != 0 ) {
		//		testPoints.push_back(Point_3(x,y,z));
		//	}*/
		//}
			

		
		

	//}
	
}

void RoadMeshReconstruction::buildPointCloud() {

	cout << "\npointcloud" << endl;

	float angle=0;

	ofstream myfile3;
	myfile3.open("testpc2.ply");

	for (int keyframe = 0; keyframe< slam_data.size(); keyframe++) {

		current_frame = keyframe;

		//Calculate Camera Transformation
		Eigen::Quaterniond q(slam_data[keyframe].quaternion.w, slam_data[keyframe].quaternion.x, -slam_data[keyframe].quaternion.z, slam_data[keyframe].quaternion.y);
		cur_quat = q;
		Eigen::Vector3d euler = q.toRotationMatrix().eulerAngles(2, 1, 0);

		// roll (x-axis rotation)
		double sinr_cosp = +2.0 * (q.w() * q.x() + q.y() * q.z());
		double cosr_cosp = +1.0 - 2.0 * (q.x() * q.x() + q.y() * q.y());
		double roll = atan2(sinr_cosp, cosr_cosp);

		// pitch (y-axis rotation)
		double sinp = +2.0 * (q.w() * q.y() - q.z() * q.x());
		double pitch;
		if (fabs(sinp) >= 1)
			pitch = copysign(M_PI / 2, sinp); // use 90 degrees if out of range
		else
			pitch = asin(sinp);

		double siny_cosp = +2.0 * (q.w() * q.z() + q.x() * q.y());
		double cosy_cosp = +1.0 - 2.0 * (q.y() * q.y() + q.z() * q.z());
		double yaw = atan2(siny_cosp, cosy_cosp);
		angle = yaw;

		cout <<"yaw :" <<yaw/M_PI*180 <<" pitch: "<< pitch /M_PI*180 <<"roll: "<< roll /M_PI*180 << endl;

		cout << keyframe << endl;

		cout << slam_data[keyframe].position.x<< endl;
		int frame = round(slam_data[keyframe].timestamp * fps);
		cout << frame << endl;

		readData(frame);

		cv::Mat seg = data.seg_ori;
		cv::Mat rgb = data.rgb_ori;


		//roadMask = filled;
		findRoadContours();
		calculateCameraHeight();

		//build pointcloud
		int size_n = 100;
		for (int idx = 0; idx < size_n; idx++) {
			for (int idy = 0; idy < size_n; idy++) {

				float max =2;
				float min = -2.0;
				float x = (max - min) * idx / (size_n - 1) + min;
				float y = (max - min) * idy / (size_n - 1) + min;

				float dis = sqrt(x * x + y * y);
				

				if (dis <= 2.0 ) {

					float a = 0;
					float b = 0;
					float c = 1;

					float d = camHeight * 0.5;

					//float z = -camHeight;
					float z = (-d - (a * x) - (b * y)) / c;
					pair<float, float> uv = projectToUV(x, y, z);
					uv.second += 0.25;
					if (uv.second > 1) {
						uv.second -= 1;
					}

					float scale = 1.0;
					if (uv.first >= 0.25 && uv.first < 0.75) {
						int u_y = (uv.first - 0.25) / 0.5 * (roadMask.rows);
						int v_x = uv.second * (roadMask.cols);

						//	test2.at<uchar>(u_y, v_x) = 255;


						//	//cv::Vec3b segColor = seg.at<cv::Vec3b>(u_y, v_x);

						if ((int)roadMask.at<uchar>(u_y, v_x) == 255) {
							
							Eigen::Vector3d pt(x,y,z);
							pt = cur_quat * pt;


							//float newx = x * cos(angle) - y * sin(angle);
							//float newy = x * sin(angle) + y * cos(angle);
							//float newz = (-d - (a * newx) - (b * newy)) / c;
					
							//float wx = (newx * scale + slam_data[keyframe].position.x);
							//float wy = (newy * scale + -slam_data[keyframe].position.z) ;
							//float wz = (newz + - slam_data[keyframe].position.y);

							float wx = pt.x() + slam_data[keyframe].position.x;
							float wy = pt.y() - slam_data[keyframe].position.z;
							float wz = pt.z(); //- slam_data[keyframe].position.y;
							
							cv::Vec3b color = rgb.at<cv::Vec3b>((int)(uv.first * rgb.rows), (int)(uv.second * rgb.cols));
							
							myfile3 << wx<<" "<<wy<<" "<<wz<< " "<<(int)color[2]<< " "<< (int)color[1]<< " "<< (int)color[0] << "\n";

							if (wx > pt_maxx) {
								pt_maxx = wx;
							}
							if (wx<pt_minx) {
								pt_minx = wx;
							}
							if (wy > pt_maxy) {
								pt_maxy = wy;
							}
							if (wy < pt_miny) {
								pt_miny = wy;
							}
						
							pc.push_back(Point_3(wx, wy, wz));
							//pc.push_back(Point_3(wx, wy, z));
						}
					}
					else {
	
					/*	float newx = x * cos(angle) - y * sin(angle);
						float newy = x * sin(angle) + y * cos(angle);
						float newz = (-d - (a * newx) - (b * newy)) / c;

						float wx = (newx * scale + slam_data[keyframe].position.x);
						float wy = (newy * scale + -slam_data[keyframe].position.z);
						float wz = (newz + -slam_data[keyframe].position.y);*/
						Eigen::Vector3d pt(x, y, z);
						pt = cur_quat * pt; 
						float wx = pt.x() + slam_data[keyframe].position.x;
						float wy = pt.y() - slam_data[keyframe].position.z;
						float wz = pt.z();// - slam_data[keyframe].position.y;
						pc.push_back(Point_3(wx,wy, wz));


						cv::Vec3b color = rgb.at<cv::Vec3b>((int)(uv.first * rgb.rows), (int)(uv.second * rgb.cols));

						myfile3 << wx << " " << wy << " " << wz << " " << (int)color[2] << " " << (int)color[1] << " " << (int)color[0] << "\n";

						if (wx > pt_maxx) {
							pt_maxx = wx;
						}
						if (wx < pt_minx) {
							pt_minx = wx;
						}
						if (wy > pt_maxy) {
							pt_maxy = wy;
						}
						if (wz < pt_miny) {
							pt_miny = wy;
						}

					}
	
				}
			}
		}
	
	}


	myfile3.close();
	cout << pt_minx << " " << pt_maxx << endl;
	cout << pt_miny << " " << pt_maxy << endl;
}

void RoadMeshReconstruction::delaunayTriangulation() {
	vector<vector<cv::Point> > contours = findPointCloudContours(pt_minx, pt_maxx, pt_miny, pt_maxy);

	//build Polygon 
	vector<Polygon_2> allPoly;
	cout << "contours " << contours.size() << endl;
	for (size_t i = 0; i < contours.size(); i++)
	{
		Polygon_2 polygon;
		for (int j = 0; j < contours[i].size(); j++) {
			int current_id = j;
			int next_id = (j + 1) % contours[i].size();
			cv::Point pt = contours[i][current_id];
			cv::Point pt_next = contours[i][next_id];

			float x = (pt_maxx - pt_minx) * pt.x / ((cnt_w * 15)) + pt_minx;
			float y = (pt_maxy - pt_miny) * pt.y / ((cnt_h * 15)) + pt_miny;

			float next_x = (pt_maxx - pt_minx) * pt_next.x / ((cnt_w * 15)) + pt_minx;
			float next_y = (pt_maxy - pt_miny) * pt_next.y / ((cnt_h * 15)) + pt_miny;

			polygon.push_back(Point_2(x, y));


		}
		testpolyon = polygon;
		allPoly.push_back(polygon);
	}

	//constraint delaunay triangulation。
	for (int i = 0; i < allPoly.size(); i++) {
		dt.insert_constraint(allPoly[i].vertices_begin(), allPoly[i].vertices_end(), true);
	}

	//outputDelaunay();

	std::list<Point_2> list_of_seeds;
	list_of_seeds.push_back(Point_2(0, 0));

	std::cout << "Number of vertices: " << dt.number_of_vertices() << std::endl;
	std::cout << "Number of finite faces: " << dt.number_of_faces() << std::endl;

	CGAL::refine_Delaunay_mesh_2(dt, list_of_seeds.begin(), list_of_seeds.end(), Criteria(0.0001, 1.0), true);

	std::cout << "Number of vertices: " << dt.number_of_vertices() << std::endl;
	std::cout << "Number of finite faces: " << dt.number_of_faces() << std::endl;

	//build wall
	int face_count = 0;
	for (CDT::Finite_faces_iterator fit = dt.finite_faces_begin(); fit != dt.finite_faces_end(); ++fit)
	{
		if (fit->is_in_domain()) {
			float a = 0;
			float b = 0;
			float c = 1;
			float d = camHeight * 0.5;

			float z1 = (-d - (a * fit->vertex(0)->point().hx()) - (b * fit->vertex(0)->point().hy())) / c;
			float z2 = (-d - (a * fit->vertex(1)->point().hx()) - (b * fit->vertex(1)->point().hy())) / c;
			float z3 = (-d - (a * fit->vertex(2)->point().hx()) - (b * fit->vertex(2)->point().hy())) / c;

			Point_3 v1 = Point_3(fit->vertex(0)->point().hx(), fit->vertex(0)->point().hy(), z1);
			Point_3 v2 = Point_3(fit->vertex(1)->point().hx(), fit->vertex(1)->point().hy(), z2);
			Point_3 v3 = Point_3(fit->vertex(2)->point().hx(), fit->vertex(2)->point().hy(), z3);

			Mesh::Vertex_index u = m.add_vertex(v1);
			Mesh::Vertex_index v = m.add_vertex(v2);
			Mesh::Vertex_index w = m.add_vertex(v3);
			m.add_face(u, v, w);

			for (int j = 0; j < 3; j++) {
				if (fit->is_constrained(j)) {

					CDT::Vertex_handle vh1 = fit->vertex((j + 2) % 3);
					CDT::Vertex_handle vh2 = fit->vertex((j + 1) % 3);

					Point_2 pt1 = vh1->point();
					Point_2 pt2 = vh2->point();

					float z1 = (-d - (a * pt1.hx()) - (b * pt1.hy())) / c;
					float z2 = (-d - (a * pt2.hx()) - (b * pt2.hy())) / c;

					Point_3 v1 = Point_3(pt1.hx(), pt1.hy(), z1);
					Point_3 v2 = Point_3(pt2.hx(), pt2.hy(), z2);
					Point_3 v3 = Point_3(pt2.hx(), pt2.hy(), z2 + 4);


					Mesh::Vertex_index u = m.add_vertex(v1);
					Mesh::Vertex_index v = m.add_vertex(v2);
					Mesh::Vertex_index w = m.add_vertex(v3);
					m.add_face(u, v, w);

					v1 = Point_3(pt2.hx(), pt2.hy(), z2 + 4);
					v2 = Point_3(pt1.hx(), pt1.hy(), z1 + 4);
					v3 = Point_3(pt1.hx(), pt1.hy(), z1);

					u = m.add_vertex(v1);
					v = m.add_vertex(v2);
					w = m.add_vertex(v3);
					m.add_face(u, v, w);
				}
			}
		}
	}
	cout << "size: " << pc.size() << endl;
}

void RoadMeshReconstruction::startReconstruction() {
	analyzeLandmarks();
	buildPointCloud();
	delaunayTriangulation();
	outputOBJ();
}