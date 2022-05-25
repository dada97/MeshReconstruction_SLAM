#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Triangulation_2.h>
#include <CGAL/draw_triangulation_2.h>
#include <CGAL/Delaunay_triangulation_2.h>
#include "RoadMeshReconstruction.h"
#include <fstream>
#include <GL/glut.h>
using namespace std::chrono;

using namespace std;
using namespace std::chrono;

int main(int argc, char* argv[]) {
	auto start = high_resolution_clock::now();
	RoadMeshReconstruction roadMeshReconstruction;
	roadMeshReconstruction.init();
	roadMeshReconstruction.startReconstruction();
	auto stop = high_resolution_clock::now();
	auto duration = duration_cast<microseconds>(stop - start);

	cout << "time :" << duration.count() << endl;
}