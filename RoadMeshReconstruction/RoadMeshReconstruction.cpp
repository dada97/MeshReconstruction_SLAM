#include "RoadMeshReconstruction.h"

void RoadMeshReconstruction::init() {
	string inputDir = "./input/case1";

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



	//for (int i = 0; i < slam_data.size(); i++) {
	//	cout << "id: " << i << endl;
	//	cout << "position :" << slam_data[i].position.x << " " << slam_data[i].position.y << " " << slam_data[i].position.z << endl;
	//	cout << "quarternion :" << slam_data[i].quaternion.x << " " << slam_data[i].quaternion.y << " " << slam_data[i].quaternion.z <<" "<< slam_data[i].quaternion.w << endl;
	//	cout << "timestamp :" << slam_data[i].timestamp << endl;
	//}

}

void RoadMeshReconstruction::readData() {
	cout << "read Data : " << frame_name[current_frame] << endl;
	string depth_path = input_dir + "/depth/" + frame_name[current_frame] + "_disp.npy";
	data.depth_arry = cnpy::npy_load(depth_path);
	data.height = data.depth_arry.shape[2];
	data.width = data.depth_arry.shape[3];

	string seg_path = input_dir + "/seg/" + frame_name[current_frame] + "_prediction.png";
	cv::Mat seg = cv::imread(seg_path, CV_LOAD_IMAGE_COLOR);
	data.seg_ori = seg;

	string rgb_path = input_dir + "/rgb/" + frame_name[current_frame] + ".jpg";
	cv::Mat rgb = cv::imread(rgb_path, CV_LOAD_IMAGE_UNCHANGED);
	rgb = rgb(cv::Rect(0, int(rgb.rows / 4), rgb.cols, (rgb.rows / 2)));

	cv::resize(seg, seg, cv::Size(data.width, data.height));
	cv::resize(rgb, rgb, cv::Size(data.width, data.height));
	data.seg = seg;
	data.rgb = rgb;
}

void RoadMeshReconstruction::findRoadContours() {
	cv::Mat road;
	cv::Mat roadline;

	cv::inRange(data.seg_ori, cv::Vec3b(128, 64, 128), cv::Vec3b(128, 64, 128), road);
	cv::inRange(data.seg_ori, cv::Vec3b(255, 255, 255), cv::Vec3b(255, 255, 255), roadline);

	roadline.copyTo(road, roadline);
	cv::resize(road, road, cv::Size(data.width, data.height));

	cv::Mat element = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(10, 10));
	cv::dilate(road, road, element);
	cv::erode(road, road, element);

	vector<vector<cv::Point> > contours;
	vector<cv::Vec4i> hierarchy;

	cv::findContours(road, contours, hierarchy, cv::RETR_TREE, cv::CHAIN_APPROX_SIMPLE);
	cv::Mat drawing = cv::Mat::zeros(data.seg.size(), CV_8UC1);
	cv::Mat filled = cv::Mat::zeros(data.seg.size(), CV_8UC1);
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

	drawContours(drawing, contours, (int)id, (255), 1, cv::LINE_8, hierarchy, 0);
	drawContours(filled, contours, id, 255, -1, cv::LINE_8, hierarchy, CV_FILLED);

	//cv::imshow("seg", drawing);
	//cv::imshow("filled", filled);
	//cv::waitKey (0);

	data.road_contour = drawing;
	data.road_fill = filled;

}

void RoadMeshReconstruction::buildPointCloudfromDepth() {
	cout << "build pointcloud!" << endl;
	float* depth_data = data.depth_arry.data<float>();
	std::vector<Point_3> points_3;
	for (int y = 0; y < data.height; y++) {
		for (int x = 0; x < data.width; x++) {

			cv::Vec3b segColor = data.seg.at<cv::Vec3b>(y, x);
			if (segColor == cv::Vec3b(255, 255, 255) || segColor == cv::Vec3b(128, 64, 128)) {

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
	/*	camHeight = plane.d();
		if (abs(camHeight - prevHeight) < 0.1) {
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

bool isLeft(Point_2 a, Point_2 b, Point_2 c) {
	return ((b.hx() - a.hx()) * (c.hy() - a.hy()) - (b.hy() - a.hy()) * (c.hx() - a.hx())) > 0;
}


void RoadMeshReconstruction::buildPlanePointCloud() {
	int size_n = 400;
	cv::Mat test = cv::Mat::zeros(cv::Size(size_n, size_n), CV_8UC1);
	cv::Mat test2 = cv::Mat::zeros(cv::Size(size_n, size_n), CV_8UC1);
	cv::Mat test3 = cv::Mat::zeros(cv::Size(size_n, size_n), CV_8UC1);
	for (int i = 0; i < size_n; i++) {
		for (int j = 0; j < size_n; j++) {
			float max = 4.5;
			float min = -4.5;
			float x = (max - min) * i / (size_n - 1) + min;
			float y = (max - min) * j / (size_n - 1) + min;

			float dis = sqrt(x * x + y * y);

			if (dis <= 4.5) {

				float a = 0;
				float b = 0;
				float c = 1;
				float d = camHeight;

				//float z = (-d - (a * x) - (b * y)) / c;

				float z = -camHeight;

				pair<float, float> uv = projectToUV(x, y, z);
				
				if (uv.first >= 0.25 && uv.first < 0.75) {
					int u_y = (uv.first - 0.25) / 0.5 * (data.height);
					int v_x = uv.second * (data.width);
					
					test2.at<uchar>(u_y, v_x) = 255;


					cv::Vec3b segColor = data.seg.at<cv::Vec3b>(u_y, v_x);
					//cout <<(int) data.road_fill.at<uchar>(u_y, v_x) << endl;
					if ((int)data.road_fill.at<uchar>(u_y, v_x) == 255) {
						
						vertices.push_back(Point_2(x, y));
						/*cv::Vec3b imgcolor = data.rgb.at<Vec3b>(u_y, v_x);
						plane_pc.push_back(PointXYZRGB(x, y, z, 255, 255, 255));

						repc.push_back(PointXYZ(x, y, z));

						repc_normal.push_back(pcl::Normal(0, 0, 1));*/
						test.at<uchar>(i, j) = 255;
					}

				}
				else {
					vertices.push_back(Point_2(x, y));
					test.at<uchar>(i, j) = 255;
					//plane_pc.push_back(PointXYZRGB(x, y, z, 255, 255, 255));
					//repc.push_back(PointXYZ(x, y, z));
				}
			}
		}
	}

	vector<vector<cv::Point> > contours;
	vector<cv::Vec4i> hierarchy;

	cv::findContours(test, contours, hierarchy, cv::RETR_TREE, cv::CHAIN_APPROX_SIMPLE);
	cv::Mat drawing = cv::Mat::zeros(size_n, size_n, CV_8UC1);
	cv::Mat filled = cv::Mat::zeros(size_n, size_n, CV_8UC1);
	int id = 0;
	float max_area = 0;

	for (size_t i = 0; i < contours.size(); i++)
	{
		float area = contourArea(contours[i], false);
		drawContours(drawing, contours, (int)i, (255), 1, cv::LINE_8, hierarchy, 0);
		if (area > max_area) {
			id = i;
			max_area = area;

		}
	}



	//cv::imshow("test2", test2);
	//cv::imshow("test", data.road_fill);
	//cv::waitKey(0);
	vector<cv::Point>  approx1;
	approx1 = contours[id];
	cout << "appro" << approx1.size() << endl;

	//cv::approxPolyDP(contours[id], approx1,1.5, true);
	//cout << "appro" << approx1.size() << endl;

	/*dt.insert_constraint(Point_2(0, 0), Point_2(-1, 2));
	dt.insert_constraint(Point_2(-1, 2), Point_2(1, 3));
	dt.insert_constraint(Point_2(1, 3), Point_2(2, 2));
	dt.insert_constraint(Point_2(2, 2), Point_2(1, 0));
	dt.insert_constraint(Point_2(1, 0), Point_2(0, 0));*/
	Polygon_2 polygon;
	for (int i = 0; i < approx1.size(); i++) {
		int current_id = i;
		int next_id = (i + 1) % approx1.size();
		cv::Point pt = approx1[current_id];
		cv::Point pt_next = approx1[next_id];

		float max = 4.5;
		float min = -4.5;
		float x = (-1)*((max - min) * pt.x / (size_n - 1) + min);
		float y = (max - min) * pt.y / (size_n - 1) + min;

		float next_x = (max - min) * pt_next.x / (size_n - 1) + min;
		float next_y = (max - min) * pt_next.y / (size_n - 1) + min;

		polygon.push_back(Point_2(x, y));


	}
	cout << "simplify" << endl;
	Cost cost;
	polygon = PS::simplify(polygon, cost, Stop(0.03));
	cout << "simplify" << endl;
	testpolyon = polygon;
	dt.insert_constraint(polygon.vertices_begin(), polygon.vertices_end(), true);
	//dt.insert(polygon.vertices_begin(), polygon.vertices_end());
	//CGAL::make_conforming_Delaunay_2(dt);
	std::list<Point_2> list_of_seeds;
	list_of_seeds.push_back(Point_2(0, 0));

	std::cout << "Number of vertices: " << dt.number_of_vertices() << std::endl;
	std::cout << "Number of finite faces: " << dt.number_of_faces() << std::endl;
	//CGAL::make_conforming_Delaunay_2(dt);
	CGAL::refine_Delaunay_mesh_2(dt, list_of_seeds.begin(),list_of_seeds.end(),Criteria(0.0001,1.0),true);
	//CGAL::lloyd_optimize_mesh_2(dt,
	//	CGAL::parameters::max_iteration_number = 10);
	//std::cout << " done." << std::endl;

	//CGAL::refine_Delaunay_mesh_2(dt, Criteria(),true);
	std::cout << "Number of vertices: " << dt.number_of_vertices() << std::endl;
	std::cout << "Number of finite faces: " << dt.number_of_faces() << std::endl;

	//MesherEngine mesher(dt);
	//mesher.set_criteria(Criteria());
	//mesher.refine_mesh();
	//mesher.

	//engine.set_criteria(Criteria(0.125, 0.5));
	//engine.refine_mesh();
	int face_count = 0;
	for (CDT::Finite_faces_iterator fit = dt.finite_faces_begin(); fit != dt.finite_faces_end(); ++fit)
	{
		//face_count++;
		//cout << face_count << endl;
		if (fit->is_in_domain()) {
			//cout << "true" <<endl; 
			//cout << (fit->is_constrained(0)) << " " << (fit->is_constrained(1)) << (fit->is_constrained(2));

			Point_3 v1 = Point_3(fit->vertex(0)->point().hx(), fit->vertex(0)->point().hy(), -camHeight);
			Point_3 v2 = Point_3(fit->vertex(1)->point().hx(), fit->vertex(1)->point().hy(), -camHeight);
			Point_3 v3 = Point_3(fit->vertex(2)->point().hx(), fit->vertex(2)->point().hy(), -camHeight);
			Mesh::Vertex_index u = m.add_vertex(v1);
			Mesh::Vertex_index v = m.add_vertex(v2);
			Mesh::Vertex_index w = m.add_vertex(v3);
			m.add_face(u, v, w);
			//dt.delete_face(fit);

			for (int j = 0; j < 3; j++) {
				if (fit->is_constrained(j)) {

					CDT::Vertex_handle vh1 = fit->vertex((j + 2)% 3);
					CDT::Vertex_handle vh2 = fit->vertex((j + 1) % 3);

					Point_2 pt1 = vh1->point();
					Point_2 pt2 = vh2->point();



					Point_3 v1 = Point_3(pt1.hx(), pt1.hy(), -camHeight);
					Point_3 v2 = Point_3(pt2.hx(), pt2.hy(), -camHeight);
					Point_3 v3 = Point_3(pt2.hx(), pt2.hy(), -camHeight + 4);

					Mesh::Vertex_index u = m.add_vertex(v1);
					Mesh::Vertex_index v = m.add_vertex(v2);
					Mesh::Vertex_index w = m.add_vertex(v3);
					m.add_face(u, v, w);

					v1 = Point_3(pt2.hx(), pt2.hy(), -camHeight + 4);
					v2 = Point_3(pt1.hx(), pt1.hy(), -camHeight + 4);
					v3 = Point_3(pt1.hx(), pt1.hy(), -camHeight);

					u = m.add_vertex(v1);
					v = m.add_vertex(v2);
					w = m.add_vertex(v3);
					m.add_face(u, v, w);
				}
			}
		}
		else {
		/*	Point_3 v1 = Point_3(fit->vertex(0)->point().hx(), fit->vertex(0)->point().hy(), -camHeight);
			Point_3 v2 = Point_3(fit->vertex(1)->point().hx(), fit->vertex(1)->point().hy(), -camHeight);
			Point_3 v3 = Point_3(fit->vertex(2)->point().hx(), fit->vertex(2)->point().hy(), -camHeight);
			Mesh::Vertex_index u = m.add_vertex(v1);
			Mesh::Vertex_index v = m.add_vertex(v2);
			Mesh::Vertex_index w = m.add_vertex(v3);
			m.add_face(u, v, w);*/
			//cout << "is not in domain" << endl;
			//dt.delete_face(fit);
		}
	}
	
	Point_2 ptp1;
	Point_2 ptp2;
	Point_2 ptp3;
	Point_2 ptp4;
	int count = 0;
	std::map<CDT::Vertex_handle, unsigned> vertex_map;

	//for (CDT::Constrained_edges_iterator fit = dt.constrained_edges_begin(); fit != dt.constrained_edges_end(); ++fit) {
	//
	//
	//	CDT::Segment seg = dt.segment(*fit);
	//	CDT::Face& f = *(fit->first);

	//	//int id = fit->second;
	//	
	//	//Point_2 pt1 = fit->first->vertex((fit->second+1)%3)->point();
	//	//Point_2 pt2 = fit->first->vertex((fit->second + 2) % 3)->point();
	//	
	//	CDT::Vertex_handle vh1 = fit->first->vertex((fit->second + 1) % 3);
	//	CDT::Vertex_handle vh2 = fit->first->vertex((fit->second + 2) % 3);

	//	if (vertex_map.find(vh1) == vertex_map.end()) {
	//		vertex_map[vh1] = count;
	//		count++;
	//	}
	//	if (vertex_map.find(vh2) == vertex_map.end()) {
	//		vertex_map[vh2] = count;
	//		count++;
	//	}
	//	int id = vertex_map[vh1];
	//	int id2 = vertex_map[vh2];
	//	Point_2 pt3 = fit->first->vertex(0)->point();
	//	Point_2 pt1 = vh1->point();
	//	Point_2 pt2 = vh2->point();

	//	



	//	


	//	//CDT::Constrained_edges_iterator = fit-1*((CDT::Constrained_edges_iterator);


	//	//Point_2 ptp1 = fit->first->vertex((fit->second + 1) % 3)->point();
	//	//Point_2 ptp2 = fit->first->vertex((fit->second + 2) % 3)->point();
	//	//CDT::Constrained_edges_iterator fit = dt.constrained_edges_begin() + 1 * sizeof(CDT::Constrained_edges_iterator);
	//	//if(dt,constrained)
	//	////Point_3 v1 = Point_3(vt.hx(), vt.hy(), camHeight);
	//	//Point_3 v2 = Point_3(pt1.hx(), pt1.hy(), camHeight);
	//	//Point_3 v3 = Point_3(pt2.hx(), pt2.hy(), camHeight);
	//	/*Mesh::Vertex_index u = m.add_vertex(v1);
	//	Mesh::Vertex_index v = m.add_vertex(v2);
	//	Mesh::Vertex_index w = m.add_vertex(v3); 
	//	m.add_face(u, v, w)*/;
	//	
	//	
	//	
	//	//cout << endl;
	//	//cout << vt.hx() <<" "<< vt.hy() << endl;
	//	//cout << pt1.hx() << " " << pt1.hy() << endl;
	//	//cout << pt2.hx() << " " << pt2.hy() << endl;

	//	/*Point_3 v1 = Point_3(pt1.hx(), pt1.hy(), camHeight);
	//	Point_3 v2 = Point_3(pt2.hx(), pt2.hy(), camHeight);
	//	Point_3 v3 = Point_3(pt2.hx(), pt2.hy(), camHeight+3);

	//	Mesh::Vertex_index u = m.add_vertex(v1);
	//	Mesh::Vertex_index v = m.add_vertex(v2);
	//	Mesh::Vertex_index w = m.add_vertex(v3);
	//	m.add_face(u, v, w);

	//	v1 = Point_3(pt2.hx(), pt2.hy(), camHeight+3);
	//	v2 = Point_3(pt1.hx(), pt1.hy(), camHeight+3);
	//	v3 = Point_3(pt1.hx(), pt1.hy(), camHeight );

	//	u = m.add_vertex(v1);
	//	v = m.add_vertex(v2);
	//	w = m.add_vertex(v3);
	//	m.add_face(u, v, w);*/
	//}

	//for (CDT::Edge_iterator fit = dt.edges_begin(); fit != dt.edges_end(); ++fit)
	//{
	//	if (dt.is_constrained(*fit)) {
	//		CDT::Face_handle& fh = fit->first;
	//
	//		CDT::Edge e = *fit;
	//		CDT::Segment s = dt.segment(fit);
	//		
	//		Point_2 pt1 = dt.
	//		Point_2 pt2 = s->vertex(2)->point();
	//		

	//		Point_3 v1 = Point_3(pt1.hx(), pt1.hy(), camHeight);
	//		Point_3 v2 = Point_3(pt2.hx(), pt2.hy(), camHeight);
	//		Point_3 v3 = Point_3(pt2.hx(), pt2.hy(), camHeight+3);

	//		Mesh::Vertex_index u = m.add_vertex(v1);
	//		Mesh::Vertex_index v = m.add_vertex(v2);
	//		Mesh::Vertex_index w = m.add_vertex(v3);
	//		m.add_face(u, v, w);

	//		v1 = Point_3(pt2.hx(), pt2.hy(), camHeight+3);
	//		v2 = Point_3(pt1.hx(), pt1.hy(), camHeight+3);
	//		v3 = Point_3(pt1.hx(), pt1.hy(), camHeight );

	//		u = m.add_vertex(v1);
	//		v = m.add_vertex(v2);
	//		w = m.add_vertex(v3);
	//		m.add_face(u, v, w);
	//	}
	//	else {

	//	}
	//}
	//
	vector<vector<cv::Point>> tmp;
	tmp.push_back(approx1);
	drawContours(drawing, tmp, (int)0, (255), 1, cv::LINE_8, hierarchy, 0);
	cv::imwrite("test3.jpg", drawing);


	

	//drawContours(filled, contours, id, 255, -1, cv::LINE_8, hierarchy, CV_FILLED);

	//dt.insert(vertices.begin(), vertices.end());

	//for (auto f = dt.faces_begin(); f != dt.faces_end(); ++f) {

	//	float face_x, face_y = 0;
	//	face_x = f->vertex(0)->point().x()+ f->vertex(1)->point().x()+ f->vertex(2)->point().x();
	//	face_x = face_x / 3;

	//	face_y = f->vertex(0)->point().y() + f->vertex(1)->point().y() + f->vertex(2)->point().y();
	//	face_y= face_y / 3;

	//	/*for (unsigned j = 0; j < 3; ++j) {
	//		face_x += f->vertex(j)->point().x();
	//		face_y += f->vertex(j)->point().y();
	//	}*/
	//
	//	if (face_x < -4.5 || face_x>4.5) {
	//		cout << face_x << endl;
	//	}
	//	if (face_y < -4.5 || face_y>4.5) {
// 
	//		cout << face_y << endl;
	//	}
	//
	//	int u, v = 0;
	//	float max = 4.5;
	//	float min = -4.5;

	//	u = (face_x - min) /(max - min)*(200 );
	//	v = (face_y - min) / (max - min) * (200 )  ;
	//	//cout << face_x << " "<< face_y<< endl;
	//	//cout << u << " "<<v << endl;
	//	if (u >= 0 && u < 200 && v >= 0 && v < 200) {
	//		if (test.at<uchar>(u, v) != 255) {
	//			test2.at<uchar>(u, v) = 255;
	//			dt.delete_face(f);
	//		}
	//	}
	//	
	//}
	cout << "done" << endl;


	
	//cv::imwrite("test2.jpg", test2);
	//	Alpha_shape_2 A(vertices.begin(), vertices.end(),FT(10000),Alpha_shape_2::GENERAL);
	//	for (Alpha_shape_2::Alpha_shape_vertices_iterator it = A.Alpha_shape_vertices_begin(); it != A.Alpha_shape_vertices_end(); ++it)
	//	{
	//		int xalpha = (*it)->point().x();
	//		int yalpha = (*it)->point().y();
	//		vertices2.push_back
	//	}
	//
	//}
}

void RoadMeshReconstruction::outputOBJ() {

	cout << "output" << endl;
	//dt.insert(vertices.begin(), vertices.end());

	string output_path = cv::utils::fs::join(output_dir, frame_name[current_frame] + ".obj");
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
	
	// write vertex
		
		
	
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
	/*cout << dt.number_of_faces() << endl;
	for (auto f = dt.faces_begin();f != dt.faces_end();++f) {
		myfile << "f";
		for (unsigned j = 0; j<3; ++j) {
			myfile << " " << vertex_map[f->vertex(j)];
		}
		myfile << std::endl;
	}*/



	myfile.close();

	

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
	
	buildPointCloud();

}

void RoadMeshReconstruction::buildPointCloud() {
	vector<Point_3> pc;
	cout << "\npointcloud" << endl;

	float angle=0;

	float pt_minx = FLT_MAX;
	float pt_maxx = FLT_MIN;

	float pt_miny = FLT_MAX;
	float pt_maxy = FLT_MIN;
	ofstream myfile3;
	myfile3.open("testpc2.ply");

	for (int keyframe = 0; keyframe <slam_data.size(); keyframe++) {


		//int keyframe = slam_data[i].key_frame_id;

		if (keyframe > 0) {
			Eigen::Vector2d v2(0, 1);
			Eigen::Vector2d v1(slam_data[keyframe].position.x- slam_data[keyframe - 1].position.x , -slam_data[keyframe].position.z-(-slam_data[keyframe-1].position.z));
			

			

			angle = -atan2(v1.x() * v2.y() - v1.y() * v2.x(), v1.x() * v2.x() + v1.y() * v2.y());
	

			//angle += atan2(v1.cross(v2).norm(), v1.dot(v2));
			cout <<"angle :" << angle/M_PI*180 << endl;
		}
		cout << keyframe << endl;
		
		//double yaw = atan2(2.0 * (q.y * q.z + q.w * q.x), q.w * q.w - q.x * q.x - q.y * q.y + q.z * q.z);
		//double pitch = asin(-2.0 * (q.x * q.z - q.w * q.y));
		//double roll = atan2(2.0 * (q.x * q.y + q.w * q.z), q.w * q.w + q.x * q.x - q.y * q.y - q.z * q.z);
		//cout << "yaw :" << euler[0] / M_PI * 180 << " pitch :" << euler[1] << " roll :" << euler[2] << endl;


		cout << slam_data[keyframe].position.x<< endl;
		int frame = round(slam_data[keyframe].timestamp * fps);
		cout << frame << endl;
		string seg_path = input_dir + "/seg/" + frame_name[frame] + "_prediction.png";
		cv::Mat seg = cv::imread(seg_path, CV_LOAD_IMAGE_COLOR);
		string rgb_path = input_dir + "/rgb/" + frame_name[frame] + ".jpg";
		cv::Mat rgb = cv::imread(rgb_path, CV_LOAD_IMAGE_COLOR);
		cv::Mat road;
		cv::Mat roadline;

		cv::inRange(seg, cv::Vec3b(128, 64, 128), cv::Vec3b(128, 64, 128), road);
		cv::inRange(seg, cv::Vec3b(255, 255, 255), cv::Vec3b(255, 255, 255), roadline);
		roadline.copyTo(road, roadline);


		cv::Mat element = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(10, 10));
		cv::dilate(road, road, element);
		cv::erode(road, road, element);

		vector<vector<cv::Point> > contours;
		vector<cv::Vec4i> hierarchy;

		cv::findContours(road, contours, hierarchy, cv::RETR_TREE, cv::CHAIN_APPROX_SIMPLE);

		cv::Mat filled = cv::Mat::zeros(seg.size(), CV_8UC1);
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
	

		int size_n = 100;


		for (int idx = 0; idx < size_n; idx++) {
			for (int idy = 0; idy < size_n; idy++) {

				float max = 1.5;
				float min = -1.5;
				float x = (max - min) * idx / (size_n - 1) + min;
				float y = (max - min) * idy / (size_n - 1) + min;

				float dis = sqrt(x * x + y * y);
				

				if (dis <= 2.0 && (keyframe==0||y>=0)) {

					float a = testplane.a();
					float b = testplane.b();
					float c = testplane.c();
					float d = testplane.d();

					//float z = -camHeight;
					float z = (-d - (a * x) - (b * y)) / c;
					pair<float, float> uv = projectToUV(x, y, z);
					uv.second += 0.25;
					if (uv.second > 1) {
						uv.second -= 1;
					}

					float scale = 2.0;
					if (uv.first >= 0.25 && uv.first < 0.75) {
						int u_y = (uv.first - 0.25) / 0.5 * (filled.rows);
						int v_x = uv.second * (filled.cols);

						//	test2.at<uchar>(u_y, v_x) = 255;


						//	//cv::Vec3b segColor = seg.at<cv::Vec3b>(u_y, v_x);

						if ((int)filled.at<uchar>(u_y, v_x) == 255) {
							//testPoints.push_back(Point_3(x, z, -y));

							//Eigen::Quaterniond q(slam_data[keyframe].quaternion.w, slam_data[keyframe].quaternion.x, -slam_data[keyframe].quaternion.y, slam_data[keyframe].quaternion.z);
							//Eigen::Vector3d pt(x, z, -y);
							//Eigen::Vector3d pntRot =  q*pt;

							float newx = x * cos(angle) - y * sin(angle);
							float newy = x * sin(angle) + y * cos(angle);
							float newz = (-d - (a * newx) - (b * newy)) / c;
					
							float wx = (newx * scale + slam_data[keyframe].position.x);
							float wy = (newy * scale + -slam_data[keyframe].position.z) ;
							float wz = (newz + - slam_data[keyframe].position.y);
							
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
							if (wz < pt_miny) {
								pt_miny = wy;
							}
						
							pc.push_back(Point_3(wx, wy, wz));
							//pc.push_back(Point_3(wx, wy, z));
						}
					}
					else {
	
						float newx = x * cos(angle) - y * sin(angle);
						float newy = x * sin(angle) + y * cos(angle);
						float newz = (-d - (a * newx) - (b * newy)) / c;

						float wx = (newx * scale + slam_data[keyframe].position.x) ;
						float wy = (newy * scale + -slam_data[keyframe].position.z) ;
						float wz = (newz + -slam_data[keyframe].position.y);
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


	float w = pt_maxx - pt_minx;
	float h = pt_maxy - pt_miny;
	cout << w << " " << h << endl;

	

	cv::Mat test = cv::Mat::zeros(cv::Size(w* 15,h* 15), CV_8UC1);
	
	cout << test.size() << endl;
	for (int i = 0; i < pc.size(); i++) {
		Point_3 pt = pc[i];
		float u = (pt.hx() - (pt_minx))/(w);
		float v = (pt.hy() - (pt_miny))/(h);

		if(u>=0&&u<1.0&&v>=0&&v<1.0)
			test.at<uchar>(v*(h* 15) ,u*(w* 15)) = 255;

	} 

	cv::waitKey(0);

	cout << "contours" << endl;
	cv::imwrite("before.jpg", test);

	//for (int i = 0; i < 10; i++) {
	//	cv::Mat element = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(5, 5));
	//	cv::dilate(test, test, element);

	//	cv::GaussianBlur(test, test, cv::Size(9, 9), 0);
	//	cv::erode(test, test, element);
	//}
	cv::inRange(test, 100,255, test);

	vector<vector<cv::Point> > contours;
	vector<cv::Vec4i> hierarchy;


	cv::findContours(test, contours, hierarchy, cv::RETR_CCOMP, cv::CHAIN_APPROX_SIMPLE);

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
	//cv::Mat smoothCont = cv::Mat::zeros(cv::Size(w * 40, h * 40), CV_8UC1);
	cv::Mat cnt = cv::Mat::zeros(cv::Size(w * 15, h * 15), CV_8UC1);

	cout << contours[id].size() << endl;
	cv::Mat drawing = cv::Mat::zeros(cv::Size(w * 15, h * 15), CV_8UC1);; //=



	//cv::drawContours(drawing, v, 0, cv::Scalar(255, 0, 0), 2, CV_AA);

	//cv::imshow("test2", test2);
	//cv::imshow("test", data.road_fill);
	//cv::waitKey(0);
	vector<vector<cv::Point>> a;
	vector<cv::Point>  approx1;
	approx1 = contours[id];
	//cv::approxPolyDP(contours[id], approx1,2, true);


	//for (int i = 1; i < approx1.size(); i++) {
	//	Eigen::Vector2d v2(approx1[i].x - approx1[i-1].x, approx1[i].y - approx1[i - 1].y);
	//
	//	Eigen::Vector2d v1(approx1[(i+1)% approx1.size()].x - approx1[i].x, approx1[(i+1)% approx1.size()].y - approx1[i].y);

	//	angle = atan2(v1.x() * v2.y() - v1.y() * v2.x(), v1.x() * v2.x() + v1.y() * v2.y());
	//	cout << angle/180*M_PI << endl;
	//	if (abs(angle) < 30 / 180 * M_PI) {

	//		Eigen::Vector2d v3(approx1[(i + 1) % approx1.size()].x - approx1[i-1].x, approx1[(i + 1) % approx1.size()].y - approx1[i - 1].y);
	//		float dis2 = sqrt(v3.x() * v3.x() + v3.y() * v3.y());
	//		float dis1 = sqrt(v2.x() * v2.x() + v2.y() * v2.y());

	//		Eigen::Vector2d v4(v3.x() / dis2*dis1, v3.y() / dis2*dis1);
	//		approx1[i] = cv::Point(v4.x(), v4.y());
	//	
	//	}
	//	cnt.at<uchar>(approx1[i].y, approx1[i].x) = 255;

	//}

	a.push_back(approx1);

	//cv::drawContours(cnt, a, id, cv::Scalar(255, 0, 0), 2, CV_AA);
	cv::imwrite("cnt.jpg", cnt);
	
	//cv::GaussianBlur(smoothCont, smoothCont, cv::Size(5,5), 0);

	drawContours(drawing, a, 0, (255), 1, cv::LINE_8, hierarchy, 0);
	cv::imwrite("testimage.jpg", test);
	cv::imwrite("drawing.jpg", drawing);


	cout << "appro" << approx1.size() << endl;

	Polygon_2 polygon;
	for (int i = 0; i < approx1.size(); i++) {
		int current_id = i;
		int next_id = (i + 1) % approx1.size();
		cv::Point pt = approx1[current_id];
		cv::Point pt_next = approx1[next_id];

		float x = (pt_maxx - pt_minx) * pt.x  / (w * 15) + pt_minx;
		float y = (pt_maxy - pt_miny) * pt.y  / (h * 15) + pt_miny;

		float next_x = (pt_maxx - pt_minx) * pt_next.x / (w * 15) + pt_minx;
		float next_y = (pt_maxy - pt_miny) * pt_next.y / (h * 15) + pt_miny;

		polygon.push_back(Point_2(x, y));


	}
	cout << "simplify" << endl;
	Cost cost;
	//polygon = PS::simplify(polygon, cost, Stop(0.3));
	cout << "simplify" << endl;
	testpolyon = polygon;
	dt.insert_constraint(polygon.vertices_begin(), polygon.vertices_end(), true);

	
	std::list<Point_2> list_of_seeds;
	list_of_seeds.push_back(Point_2(0, 0));

	std::cout << "Number of vertices: " << dt.number_of_vertices() << std::endl;
	std::cout << "Number of finite faces: " << dt.number_of_faces() << std::endl;
	
	CGAL::refine_Delaunay_mesh_2(dt, list_of_seeds.begin(), list_of_seeds.end(), Criteria(0.0001, 1.0), true);

	std::cout << "Number of vertices: " << dt.number_of_vertices() << std::endl;
	std::cout << "Number of finite faces: " << dt.number_of_faces() << std::endl;

	int face_count = 0;
	for (CDT::Finite_faces_iterator fit = dt.finite_faces_begin(); fit != dt.finite_faces_end(); ++fit)
	{
		//face_count++;
		//cout << face_count << endl;
		if (fit->is_in_domain()) {
			//cout << "true" <<endl; 
			//cout << (fit->is_constrained(0)) << " " << (fit->is_constrained(1)) << (fit->is_constrained(2));
			float a = testplane.a();
			float b= testplane.b();
			float c= testplane.c();
			float d= testplane.d();

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
			//dt.delete_face(fit);

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
					Point_3 v3 = Point_3(pt2.hx(), pt2.hy(), z2+ 4);
				

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
	ofstream myfile;
	myfile.open("testpc2.obj");
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

	// write vertex



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
	
}

void RoadMeshReconstruction::startReconstruction() {
	for (current_frame = 0; current_frame < total_frame; current_frame+=1) {
	//current_frame = 1020;
		readData();
		findRoadContours();
		buildPointCloudfromDepth();
		buildPlanePointCloud();
		outputOBJ();
		vertices.clear();
		vertices2.clear();
		dt.clear();
		testpolyon.clear();
		m.clear();
	//reconstruction();

	}
}