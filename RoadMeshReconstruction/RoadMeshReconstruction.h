#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Delaunay_triangulation_2.h>
#include <CGAL/Triangulation_2.h>
#include "cnpy.h"
#include <CGAL/linear_least_squares_fitting_3.h>
#include <filesystem>
#include <CGAL/Simple_cartesian.h>

#include <opencv2/core/utils/filesystem.hpp>

#include <CGAL/Regular_triangulation_2.h>
#include <CGAL/Alpha_shape_2.h>
#include <CGAL/Alpha_shape_vertex_base_2.h>
#include <CGAL/Alpha_shape_face_base_2.h>
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
#include <CGAL/lloyd_optimize_mesh_2.h>
#include "csv_parser.h"
#include <CGAL/compute_average_spacing.h>
#include <CGAL/remove_outliers.h>

# define M_PI  3.14159265358979323846 
namespace fs = std::filesystem;
using namespace std;

typedef CGAL::Exact_predicates_inexact_constructions_kernel K;
typedef K::FT FT;
typedef K::Plane_3											Plane;
typedef K::Point_3											Point_3;
typedef K::Point_2		Point_2;
typedef K::Segment_2 Segment;



typedef CGAL::Triangulation_vertex_base_2<K> Vb;
typedef CGAL::Delaunay_mesh_face_base_2<K> Fb;
//typedef CGAL::Triangulation_data_structure_2<Vb, Fb> Tds;
// Vertex type

//typedef CGAL::Alpha_shape_vertex_base_2<K, Vb>                 AsVb;
//typedef CGAL::Alpha_shape_face_base_2<K, Fb>                 AsFb;

typedef CGAL::Triangulation_data_structure_2<Vb, Fb>        Tds;
typedef CGAL::Constrained_Delaunay_triangulation_2<K, Tds> CDT;
//typedef CGAL::Alpha_shape_2<CDT>                 Alpha_shape_2;

typedef CGAL::Surface_mesh<Point_3> Mesh;
typedef Mesh::Vertex_index m_vd;
typedef Mesh::Face_index m_fd;
//typedef Alpha_shape_2::Alpha_shape_edges_iterator            Alpha_shape_edges_iterator;
//typedef Alpha_shape_2::Alpha_shape_vertices_iterator         Alpha_shape_vertices_iterator;

typedef CGAL::Delaunay_mesh_size_criteria_2<CDT> Criteria;
typedef CDT::Finite_vertices_iterator FiniteVerticesIterator;
typedef CDT::Finite_faces_iterator FiniteFacesIterator;
typedef CDT::Finite_edges_iterator FiniteEdgesIterator;
typedef CDT::Face_circulator FacesCirculator;
typedef CDT::Point_iterator PointIterator;
typedef CDT::Vertex Vertex;
typedef CDT::Vertex_handle VertexHandle;
typedef CDT::Edge Edge;
typedef CDT::Face Face;
typedef CDT::Face_handle FaceHandle;
typedef CDT::Triangle Triangle;
typedef CGAL::Polygon_2<K>   Polygon_2;
typedef CGAL::Delaunay_mesher_2<CDT, Criteria> MesherEngine;

namespace PS = CGAL::Polyline_simplification_2;

typedef PS::Stop_below_count_ratio_threshold Stop;
typedef PS::Squared_distance_cost            Cost;
typedef CGAL::Parallel_if_available_tag Concurrency_tag;



//typedef CGAL::Delaunay_triangulation_2<K>					Delaunay;
//typedef CGAL::Triangulation_2<K>                            Triangulation;
//typedef CGAL::Constrained_Delaunay_triangulation_2<K>		CDT;
//typedef CGAL::Delaunay_mesh_vertex_base_2<K>                Vb;
//typedef CGAL::Delaunay_mesh_size_criteria_2<CDT> Criteria;
//typedef CGAL::Delaunay_mesher_2<CDT, Criteria> MesherEngine;
//typedef CGAL::Delaunay_mesh_face_base_2<K> Fb;


struct InputData {
	cv::Mat rgb;
	cv::Mat seg;
	cv::Mat seg_ori;
	cv::Mat road_contour;
	cv::Mat road_fill;
	cnpy::NpyArray depth_arry;
	int width;
	int height;
};

class RoadMeshReconstruction {
public:
	InputData data;
	void init();
	void startReconstruction();
	void analyzeLandmarks();
	int current_frame = 0;

	int total_frame = 0;
private:
	
	void readData();
	void findRoadContours();
	void buildPointCloudfromDepth();
	void buildPlanePointCloud();

	void buildPointCloud();

	std::pair<float, float> projectToUV(float x, float y, float z);

	void outputOBJ();
	std::vector<Point_2> vertices;
	std::vector<Point_2> vertices2;
	string input_dir;
	string output_dir;
	float camHeight = 0;
	float prevHeight = 0;
	vector<string> frame_name;
	Polygon_2 testpolyon;
	Mesh m;
	CDT dt;
	vector<Slam_data> slam_data;
	vector<Landmark_data> landmarks_data;

	std::vector<Point_3> testPoints;
	Plane testplane;
	int fps = 24;
};