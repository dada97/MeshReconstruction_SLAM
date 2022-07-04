#include "RoadMeshReconstruction.h"

RoadMeshReconstruction::RoadMeshReconstruction(string input, string output, bool debug) {
	input_dir = input;
	output_dir = output;
	debugMode = debug;
}

void RoadMeshReconstruction::init() {
	//initial data
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

	cv::utils::fs::createDirectory(output_dir);

	if (debugMode == true) {
		cv::utils::fs::createDirectory(debug_dir);
	}

	//parse slam data
	string slam_path = input_dir + "/slam.csv";
	string landmarks_path = input_dir + "/landmarks.csv";
	string file_contents;

	std::map<int, std::vector<string>> csv_contents;
	char delimiter = ',';
	CSV_parser csv_parser(slam_path);
	slam_data = csv_parser.get_slamdata();
}

void RoadMeshReconstruction::readData(int index) {
	cout << "read Data : " << frame_name[index] << endl;

	//read depth data
	string depth_path = input_dir + "/depth/" + frame_name[index] + "_disp.npy";
	data.depth_arry = cnpy::npy_load(depth_path);
	data.height = data.depth_arry.shape[2];
	data.width = data.depth_arry.shape[3];

	//read rgb img
	string rgb_path = input_dir + "/rgb/" + frame_name[index] + ".jpg";
	cv::Mat rgb = cv::imread(rgb_path, CV_LOAD_IMAGE_UNCHANGED);
	data.rgb_ori = rgb;
	rgb = rgb(cv::Rect(0, int(rgb.rows / 4), rgb.cols, (rgb.rows / 2))); //crop rgb

	//read segmentation image
	string seg_path = input_dir + "/seg/" + frame_name[index] + "_prediction.png";
	cv::Mat seg = cv::imread(seg_path, CV_LOAD_IMAGE_COLOR);
	data.seg_ori = seg;

	//resize rgb and segmentation as depth_data size
	cv::resize(seg, seg, cv::Size(data.width, data.height));
	cv::resize(rgb, rgb, cv::Size(data.width, data.height));

	data.seg = seg;
	data.rgb = rgb;
}

void RoadMeshReconstruction::findRoadMask() {

	cv::Mat road;
	cv::Mat roadline;
	cv::Mat other;

	//filter label 
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
	data.roadMask = filled;
}

void RoadMeshReconstruction::calculateCameraHeight() {

	float* depth_data = data.depth_arry.data<float>();
	std::vector<Point_3> points_3;

	cv::Mat mask = data.roadMask;
	cv::resize(mask, mask, cv::Size(data.width, data.height));

	for (int y = 0; y < data.height; y++) {
		for (int x = 0; x < data.width; x++) {

			//if pixel label is road
			if ((int)mask.at<uchar>(y, x) == 255) {

				int idx = y * data.rgb.cols + x;

				//pixel actual depth
				float depth = 1 / (depth_data[idx]);

				//calculate 3d point from 2d pixel point; 
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
	
	// least square method find plane equation
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

	if (debugMode) {
		cout << "camHeight: " << camHeight << endl;
	}
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



std::vector<std::vector<cv::Point>> RoadMeshReconstruction::findPointCloudContours(float min_x, float max_x, float min_y,float max_y) {
	float w = max_x - min_x;
	float h = max_y - min_y;
	
	//Find contours from pointcloud
	cv::Mat pcImg = cv::Mat::zeros(cv::Size(w * PCIMGSCALE +10, h * PCIMGSCALE +10 ), CV_8UC1);

	cout << pcImg.size() << endl;
	for (int i = 0; i < pc.size(); i++) {
		Point_3 pt = pc[i];
		float u = (pt.hx() - (min_x)) / (w);
		float v = (pt.hy() - (min_y)) / (h);

		if (u >= 0 && u < 1.0 && v >= 0 && v < 1.0)
			pcImg.at<uchar>(v * (pcImg.rows-10)+5, u * (pcImg.cols-10)+5) = 255;
	}

	if (debugMode == true) {
		cv::imwrite(debug_dir + "pointcloudMask_Original.jpg", pcImg);
	}

	cv::Mat tmp;
	pcImg.copyTo(tmp);

	for (int i = 0; i < 7; i++) {
		cv::Mat element = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(5, 5));
		cv::dilate(pcImg, pcImg, element);
		cv::GaussianBlur(pcImg, pcImg, cv::Size(5, 5), 0);
		cv::erode(pcImg, pcImg, element);
		cv::inRange(pcImg, 150, 255, pcImg);
	}
	pcImg(cv::Rect(5, 5, pcImg.cols-10, pcImg.rows-10)).copyTo(pcImg);

	cnt_w = pcImg.cols;
	cnt_h = pcImg.rows;

	if (debugMode == true) {
		cv::imwrite(debug_dir + "pointcloudMask_Simplify.jpg", pcImg);
	}

	vector<vector<cv::Point> > contours;
	vector<cv::Vec4i> hierarchy;

	cv::findContours(pcImg, contours, hierarchy, cv::RETR_CCOMP, cv::CHAIN_APPROX_SIMPLE);

	int id = 0;
	float max_area = 0;
	cv::Mat cntImg = cv::Mat::zeros(pcImg.size(), CV_8UC1);

	for (size_t i = 0; i < contours.size(); i++)
	{
		float area = contourArea(contours[i], false);
		if (area > max_area) {
			id = i;
			max_area = area;

		}
		drawContours(cntImg, contours, i, (255), 1, cv::LINE_8, hierarchy, 0);
	}

	if (debugMode == true) {
		cv::imwrite(debug_dir + "pointcloudContours.jpg", cntImg);
	}

	return contours;
}

void RoadMeshReconstruction::buildPointCloud() {
	float angle=0;
	for (int keyframe = 0; keyframe< slam_data.size(); keyframe++) {

		current_frame = keyframe;

		//Calculate Camera Rotation 

		//Quarternion convert to Rotation
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

		if (debugMode) {
			cout << "yaw :" << yaw / M_PI * 180 << " pitch: " << pitch / M_PI * 180 << " roll: " << roll / M_PI * 180 << endl;
		}
		
		int frame = round(slam_data[keyframe].timestamp * FPS);
		readData(frame);
		findRoadMask();

		calculateCameraHeight();

		cv::Mat seg = data.seg_ori;
		cv::Mat rgb = data.rgb_ori;
		cv::Mat mask = data.roadMask;
	
		//build pointcloud
		int size_n = PCSIZE;
		for (int idx = 0; idx < size_n; idx++) {
			for (int idy = 0; idy < size_n; idy++) {

				float max = PCMAXRANGE;
				float min = PCMINRANGE;
				float x = (max - min) * idx / (size_n - 1) + min;
				float y = (max - min) * idy / (size_n - 1) + min;

				float dis = sqrt(x * x + y * y);
				

				if (dis <= 2.0 ) {

					float a = 0;
					float b = 0;
					float c = 1;

					float d = camHeight * CAMERAHEIGHTSCALE;

					float z = (-d - (a * x) - (b * y)) / c;
					pair<float, float> uv = projectToUV(x, y, z);
					uv.second += 0.25;
					if (uv.second > 1) {
						uv.second -= 1;
					}

					float scale = 1.0;
					if (uv.first >= 0.25 && uv.first < 0.75) {
						int u_y = (uv.first - 0.25) / 0.5 * (mask.rows);
						int v_x = uv.second * (mask.cols);

						if ((int)mask.at<uchar>(u_y, v_x) == 255) {
							
							Eigen::Vector3d pt(x,y,z);
							pt = cur_quat * pt;

							float wx = pt.x() + slam_data[keyframe].position.x;
							float wy = pt.y() - slam_data[keyframe].position.z;
							float wz = pt.z();
	
							cv::Vec3b color = rgb.at<cv::Vec3b>((int)(uv.first * rgb.rows), (int)(uv.second * rgb.cols));
							pc.push_back(Point_3(wx, wy, wz));
							pc_color.push_back(color);
							
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
						

						}
					}
					else {
	
						Eigen::Vector3d pt(x, y, z);
						pt = cur_quat * pt; 
						float wx = pt.x() + slam_data[keyframe].position.x;
						float wy = pt.y() - slam_data[keyframe].position.z;
						float wz = pt.z();
				
						cv::Vec3b color = rgb.at<cv::Vec3b>((int)(uv.first * rgb.rows), (int)(uv.second * rgb.cols));
						pc.push_back(Point_3(wx, wy, wz));
						pc_color.push_back(color);

						if (wx > pt_maxx) {
							pt_maxx = wx;
						}
						if (wx < pt_minx) {
							pt_minx = wx;
						}
						if (wy > pt_maxy) {
							pt_maxy = wy;
						}
						if (wy < pt_miny) {
							pt_miny = wy;
						}
					}
				}
			}
		}
	}

	if (debugMode == true) {
		outputPointCloud();
	}
}

void RoadMeshReconstruction::delaunayTriangulation() {

	vector<vector<cv::Point> > contours = findPointCloudContours(pt_minx, pt_maxx, pt_miny, pt_maxy);
	//build 2d Polygon from contours edge
	vector<Polygon_2> allPoly;
	for (size_t i = 0; i < contours.size(); i++)
	{
		Polygon_2 polygon;
		for (int j = 0; j < contours[i].size(); j++) {
			int current_id = j;
			cv::Point pt = contours[i][current_id];
	
			float x = ((float)pt.x / (float)(cnt_w)) * (pt_maxx - pt_minx) + pt_minx;
			float y = ((float)pt.y / (float)(cnt_h)) * (pt_maxy - pt_miny) + pt_miny;
			polygon.push_back(Point_2(x, y));
		}
		allPoly.push_back(polygon);
	}

	//constraint delaunay triangulation。
	for (int i = 0; i < allPoly.size(); i++) {
		dt.insert_constraint(allPoly[i].vertices_begin(), allPoly[i].vertices_end(), true);
	}

	if (debugMode == true) {
		string output_path = cv::utils::fs::join(debug_dir, "Delaunay.obj");
		outputDelaunay(output_path);
	}

	std::list<Point_2> list_of_seeds;
	list_of_seeds.push_back(Point_2(0, 0));

	//refine Delaunay 
	CGAL::refine_Delaunay_mesh_2(dt, list_of_seeds.begin(), list_of_seeds.end(), Criteria(0.0001, 1.0), true);

	if (debugMode == true) {
		string output_path = cv::utils::fs::join(debug_dir, "DelaunayRefinement.obj");
		outputDelaunay(output_path);
	}

	//build wall along the mesh edge
	int face_count = 0;

	std::map<Point_3, Mesh::Vertex_index> vertex_map;

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

			addMeshFace(v1, v2, v3);


			Mesh::Vertex_index u = m.add_vertex(v1);
			Mesh::Vertex_index v = m.add_vertex(v2);
			Mesh::Vertex_index w = m.add_vertex(v3);
			//m.add_face(u, v, w);

			u = m_floor.add_vertex(v1);
			v = m_floor.add_vertex(v2);
			w = m_floor.add_vertex(v3);
			m_floor.add_face(u, v, w);

			Point_3 v4 = Point_3(fit->vertex(0)->point().hx(), fit->vertex(0)->point().hy(), z1 + CEILINGHEIGHT);
			Point_3 v5 = Point_3(fit->vertex(1)->point().hx(), fit->vertex(1)->point().hy(), z2 + CEILINGHEIGHT);
			Point_3 v6 = Point_3(fit->vertex(2)->point().hx(), fit->vertex(2)->point().hy(), z3 + CEILINGHEIGHT);

			addMeshFace(v6, v5, v4);

			/*u = m.add_vertex(v6);
			v = m.add_vertex(v5);
			w = m.add_vertex(v4);
			m.add_face(u, v, w);*/


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
					Point_3 v3 = Point_3(pt2.hx(), pt2.hy(), z2 + CEILINGHEIGHT);

					addMeshFace(v1, v2, v3);
					/*Mesh::Vertex_index u = m.add_vertex(v1);
					Mesh::Vertex_index v = m.add_vertex(v2);
					Mesh::Vertex_index w = m.add_vertex(v3);
					m.add_face(u, v, w);*/

					v1 = Point_3(pt2.hx(), pt2.hy(), z2 + CEILINGHEIGHT);
					v2 = Point_3(pt1.hx(), pt1.hy(), z1 + CEILINGHEIGHT);
					v3 = Point_3(pt1.hx(), pt1.hy(), z1);
					addMeshFace(v1, v2, v3);

				/*	u = m.add_vertex(v1);
					v = m.add_vertex(v2);
					w = m.add_vertex(v3);
					m.add_face(u, v, w);*/
				}
			}
		}
	}

	if (debugMode == true) {
		string output_path = cv::utils::fs::join(debug_dir, "mesh_Floor.obj");
		outputOBJ(m_floor, output_path);
	}
}

void RoadMeshReconstruction::addMeshFace(Point_3 v1, Point_3 v2, Point_3 v3) {
	Mesh::Vertex_index f1;
	Mesh::Vertex_index f2;
	Mesh::Vertex_index f3;

	if (vt_map.find(v1) == vt_map.end()) {
		f1 = m.add_vertex(v1);
		vt_map[v1] = f1;
	}
	else {
		f1 = vt_map[v1];
	}

	if (vt_map.find(v2) == vt_map.end()) {
		f2 = m.add_vertex(v2);
		vt_map[v2] = f2;
	}
	else {
		f2 = vt_map[v2];
	}

	if (vt_map.find(v3) == vt_map.end()) {
		f3 = m.add_vertex(v3);
		vt_map[v3] = f3;
	}
	else {
		f3 = vt_map[v3];
	}
	m.add_face(f1, f2, f3);

};

void RoadMeshReconstruction::outputPointCloud() {
	cout << "write pointcloud" << endl;
	ofstream of;
	string output_path = cv::utils::fs::join(debug_dir, "pointcloud.ply");
	of.open(output_path);
	of << "ply\n"
		<< "format ascii 1.0\n"
		<< "element vertex " + std::to_string(pc.size()) << "\n"
		<< "property float x\n"
		<< "property float y\n"
		<< "property float z\n"
		<< "property uint8 red\n"
		<< "property uint8 green\n"
		<< "property uint8 blue\n"
		<< "end_header\n";
	for (int i = 0; i < pc.size(); i++) {
		cv::Vec3b color = pc_color[i];
		of << pc[i].hx()<<" "<< pc[i].hy()<<" "<< pc[i].hz() << " " << (int)color[2] << " " << (int)color[1] << " " << (int)color[0] << "\n";
	}
	of.close();
}

void RoadMeshReconstruction::outputDelaunay(string path) {
	cout << "write delaunay" << endl;
	ofstream ofmesh;

	ofmesh.open(path);
	ofmesh << "\n";
	std::map < CDT::Vertex_handle, unsigned > vm;
	unsigned ccount = 1;

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

void RoadMeshReconstruction::outputOBJ(Mesh mesh,string path) {
	cout << "write mesh" << endl;

	ofstream of;
	of.open(path);
	std::map<Mesh::Vertex_index, unsigned> vertex_map;
	unsigned count = 1;

	// Loop over vertices
	for (Mesh::Vertex_index vi : mesh.vertices()) {
		K::Point_3 pt = mesh.point(vi);
		vertex_map[vi] = count;
		++count;
		of << "v " << pt << "\n";
	}

	of << "\n";
	// Map over facets. Each facet is a cell of the underlying
	// Delaunay triangulation, and the vertex that is not part of
	// this facet. We iterate all vertices of the cell except the one
	// that is opposite.
	for (Mesh::Face_index face_index : mesh.faces()) {
		of << "f";
		Mesh::Halfedge_index hf = mesh.halfedge(face_index);
		for (Mesh::Halfedge_index hi : halfedges_around_face(hf, mesh))
		{
			Mesh::Vertex_index vi = target(hi, mesh);
			of << " " << vertex_map[vi];
		}
		of << std::endl;
	}

	of.close();
}


void RoadMeshReconstruction::startReconstruction() {
	buildPointCloud();
	delaunayTriangulation();
	
	//write Result
	string output_path = cv::utils::fs::join(output_dir, "outputMesh.obj");
	outputOBJ(m,output_path);
}