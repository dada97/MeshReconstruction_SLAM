#include "cnpy.h"
#include <filesystem>
#include "csv_parser.h"
#include "Config.h"

#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/core/utils/filesystem.hpp>

#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Delaunay_triangulation_2.h>
#include <CGAL/Triangulation_2.h>

#include <CGAL/linear_least_squares_fitting_3.h>
#include <CGAL/Simple_cartesian.h>


#include <CGAL/Regular_triangulation_2.h>

#include <CGAL/Constrained_Delaunay_triangulation_2.h>
#include <CGAL/Dimension.h>
#include <CGAL/Triangulation_conformer_2.h>
#include <CGAL/Delaunay_mesh_face_base_2.h>
#include <CGAL/Delaunay_mesh_vertex_base_2.h>
#include <CGAL/Delaunay_mesher_2.h>
#include <CGAL/Delaunay_mesh_size_criteria_2.h>
#include <CGAL/Polygon_2.h>
#include <CGAL/Polyline_simplification_2/simplify.h>
#include <CGAL/Polyline_simplification_2/Squared_distance_cost.h>
#include <CGAL/algorithm.h>
#include <CGAL/assertions.h>
#include <CGAL/Surface_mesh.h>

# define M_PI  3.14159265358979323846 
namespace fs = std::filesystem;
using namespace std;

typedef CGAL::Exact_predicates_inexact_constructions_kernel K;
typedef K::FT			FT;
typedef K::Plane_3		Plane;
typedef K::Point_3		Point_3;
typedef K::Point_2		Point_2;
typedef K::Segment_2	Segment;

typedef CGAL::Triangulation_vertex_base_2<K> Vb;
typedef CGAL::Delaunay_mesh_face_base_2<K> Fb;

typedef CGAL::Triangulation_data_structure_2<Vb, Fb>        Tds;
typedef CGAL::Constrained_Delaunay_triangulation_2<K, Tds>	CDT;

typedef CGAL::Surface_mesh<Point_3> Mesh;
typedef Mesh::Vertex_index m_vd;
typedef Mesh::Face_index m_fd;

typedef CGAL::Delaunay_mesh_size_criteria_2<CDT>	Criteria;
typedef CDT::Finite_vertices_iterator				FiniteVerticesIterator;
typedef CDT::Finite_faces_iterator					FiniteFacesIterator;
typedef CDT::Finite_edges_iterator					FiniteEdgesIterator;
typedef CDT::Face_circulator						FacesCirculator;
typedef CDT::Point_iterator							PointIterator;
typedef CDT::Vertex									Vertex;
typedef CDT::Vertex_handle							VertexHandle;
typedef CDT::Edge									Edge;
typedef CDT::Face									Face;
typedef CDT::Face_handle							FaceHandle;
typedef CDT::Triangle								Triangle;
typedef CGAL::Polygon_2<K>							Polygon_2;
typedef CGAL::Delaunay_mesher_2<CDT, Criteria>		MesherEngine;

namespace PS = CGAL::Polyline_simplification_2;

typedef PS::Stop_below_count_ratio_threshold Stop;
typedef PS::Squared_distance_cost            Cost;
typedef CGAL::Parallel_if_available_tag Concurrency_tag;

struct InputData {
	cv::Mat rgb;
	cv::Mat rgb_ori;
	cv::Mat seg;
	cv::Mat seg_ori;
	cv::Mat road_contour;
	cv::Mat roadMask;
	cnpy::NpyArray depth_arry;
	int width;
	int height;
};


class RoadMeshReconstruction {
public:
	InputData data;
	vector<Slam_data> slam_data;
	RoadMeshReconstruction(string input, string output,bool debug);
	void init();
	void startReconstruction();

	int current_frame = 0;
	int total_frame = 0;
private:
	
	void readData(int index);
	void findRoadMask();				//find roud contours from segmentation image 
	void calculateCameraHeight(); 
	void buildPointCloud();
	std::vector<std::vector<cv::Point>> findPointCloudContours(float min_x, float max_x, float min_y, float max_y);
	void delaunayTriangulation();
	void addMeshFace(Point_3 v1,Point_3 v2,Point_3 v3);

	void outputPointCloud();					//output Pointcloud
	void outputDelaunay(string path);			//output Delaunay result
	void outputOBJ(Mesh mesh,string path);		//output mesj

	std::pair<float, float> projectToUV(float x, float y, float z); //project 3d point to UV space

	string input_dir;
	string output_dir="./output/";
	string debug_dir="./debug/";

	vector<string> frame_name;

	bool debugMode = false;

	float camHeight = 0;
	float prevHeight = 0;
	
	std::vector<Point_3> pc;
	std::vector<cv::Vec3b> pc_color;
	Mesh m;		
	Mesh m_floor;

	float cnt_w = 0; //pointcloud contour width
	float cnt_h = 0; //pointcloud contour height
	
	Eigen::Quaterniond cur_quat;
	

	float pt_minx = FLT_MAX;
	float pt_maxx = FLT_MIN;

	float pt_miny = FLT_MAX;
	float pt_maxy = FLT_MIN;

	CDT dt; //constraint delaunay triangulation 

	std::map<Point_3, Mesh::Vertex_index> vt_map; //Point map to mesh vertex index
};