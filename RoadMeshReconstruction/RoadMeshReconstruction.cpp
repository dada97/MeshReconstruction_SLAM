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
	polygon = PS::simplify(polygon, cost, Stop(0.05));
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