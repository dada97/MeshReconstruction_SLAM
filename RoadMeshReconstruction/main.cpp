#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Triangulation_2.h>
#include <CGAL/draw_triangulation_2.h>
#include <CGAL/Delaunay_triangulation_2.h>
#include "RoadMeshReconstruction.h"
#include <fstream>
#include <GL/glut.h>

using namespace std;

int main(int argc, char* argv[]) {

	RoadMeshReconstruction roadMeshReconstruction;
	roadMeshReconstruction.init();
	roadMeshReconstruction.analyzeLandmarks();

}