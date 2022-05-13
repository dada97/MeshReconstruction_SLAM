#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Triangulation_2.h>
#include <CGAL/draw_triangulation_2.h>
#include <CGAL/Delaunay_triangulation_2.h>
#include "RoadMeshReconstruction.h"
#include <fstream>
#include <GL/glut.h>

using namespace std;

//typedef CGAL::Exact_predicates_inexact_constructions_kernel K;
//typedef CGAL::Triangulation_2<K>                            Triangulation;
//typedef Triangulation::Point                                Point;
//typedef CGAL::Delaunay_triangulation_2<K>					Delaunay;
//typedef Delaunay::Vertex_handle Vertex_handle;
//typedef K::Point_2 Point;

//std::vector<Point> vertices, mypts;
//int global_h;
//
//void points_draw() {
//	glPushMatrix();
//	glClear(GL_COLOR_BUFFER_BIT);
//	glPushMatrix();
//
//	std::vector<Point>::iterator iter;
//	glColor3f(1.0, 1.0, 1.0);
//	glPointSize(8);
//	glBegin(GL_POINTS);
//	for (iter = vertices.begin(); iter != vertices.end(); iter++) {
//		glVertex2i(iter->hx(), iter->hy());
//
//	}
//	glEnd();
//	glPopMatrix();
//	glutSwapBuffers();
//
//
//}
//
//
//void points_add_point(int x, int y) {
//	vertices.push_back(Point(x, global_h-y));
//}
//
//
//void points_clear() {
//	glClear(GL_COLOR_BUFFER_BIT);
//	glPushMatrix();
//	glPopMatrix();
//	glutSwapBuffers();
//	vertices.clear();
//}
//
//void points_triangulation() {
//	
//	Delaunay dt;
//	dt.insert(vertices.begin(), vertices.end());
//	glPushMatrix();
//	Delaunay::Finite_faces_iterator fit;
//	glColor3f(0, 0, 1);
//	for (fit = dt.finite_faces_begin(); fit != dt.finite_faces_end(); fit++) {
//		glBegin(GL_LINE_LOOP);
//		glVertex2i(fit->vertex(0)->point().hx(), fit->vertex(0)->point().hy());
//		glVertex2i(fit->vertex(1)->point().hx(), fit->vertex(1)->point().hy());
//		glVertex2i(fit->vertex(2)->point().hx(), fit->vertex(2)->point().hy());
//		glEnd();
//	}
//	
//}
//
//void init(void) {
//	glClearColor(0, 0, 0, 0);
//	glEnable(GL_DEPTH_TEST);
//}
//
//void reshape(int w, int h) {
//	cout << "reshape" << endl;
//	glViewport(0, 0, w, h);
//	glMatrixMode(GL_PROJECTION);
//	glLoadIdentity();
//	gluPerspective(45.0, (double)w / (double)h, 1.0, 200.0);
//	
//	/*glViewport(0, 0, w, h);
//	glMatrixMode(GL_PROJECTION);
//	glLoadIdentity();
//
//	glOrtho(0, w, 0, h, -1, 1.0);
//	glMatrixMode(GL_MODELVIEW);
//	glLoadIdentity();*/
//
//	//points_draw();
//
//}
//
//void mouse(int button, int state, int x, int y) {
//
//	if (button == GLUT_LEFT_BUTTON) {
//		cout << "mouse" << " " << x << " " << y << endl;
//		points_add_point(x, y);
//		points_draw();
//	}
//	if (button == GLUT_RIGHT_BUTTON) {
//		points_triangulation();
//	}
//}
//
//void display(void) {
//	//glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
//	//glMatrixMode(GL_MODELVIEW);
//	//glLoadIdentity();
//	//glRotatef(0, 0.0f, 1.0f, 0.0f); //rotate object by 30 degree with respect to y-axis
//	//glTranslatef(0.0f, 0.0f, -10.0f);
//
//	//glPushMatrix();
//	//glTranslatef(5.0f, -1.0f, 0.0f);
//	//glScalef(2.0f, 2.0f, 2.0f);
//	//glRotatef(0, 1.0f, 3.0f, 2.0f); //rotating object continuously by 2 degree
//	//glBegin(GL_QUADS);
//
//	//glVertex3f(-0.7f, 0.0f, 0.0);
//	//glVertex3f(0.7f, 0.0f, 0.0);
//	//glVertex3f(0.5f, 2.0f, 0.0);
//	//glVertex3f(-0.5f, 2.0f, 0.0);
//
//	//glEnd();
//
//	//glPopMatrix();
//	//glutSwapBuffers();
//
//}
//
//void write_to_obj(Delaunay dt) {
//
//	ofstream myfile;
//	myfile.open("output.obj");
//	std::map<Delaunay::Vertex_handle, unsigned> vertex_map;
//	unsigned count = 1;
//
//	// Loop over vertices
//	for (auto v = dt.vertices_begin(); v != dt.vertices_end(); ++v) {
//		vertex_map[v] = count;
//		++count;
//		auto point = v->point();
//		myfile << "v " << point << " 0.0\n";
//	}
//
//	// write vertex
//	
//	
//
//	myfile << "\n";
//	// Map over facets. Each facet is a cell of the underlying
//	// Delaunay triangulation, and the vertex that is not part of
//	// this facet. We iterate all vertices of the cell except the one
//	// that is opposite.
//	for (auto f = dt.faces_begin();f != dt.faces_end();++f) {
//		myfile << "f";
//		for (unsigned j = 0; j<3; ++j) {
//			myfile << " " << vertex_map[f->vertex(j)];
//		}
//		myfile << std::endl;
//	}
//	myfile.close();
//}

int main(int argc, char* argv[]) {
	//glutInit(&argc, argv);
	//glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB);
	//glutInitWindowSize(800, 600);
	////glutInitWindowPosition(0, 0);

	//glutCreateWindow(argv[0]);

	//init();
	//glutDisplayFunc(display);
	//glutReshapeFunc(reshape);
	//glutMouseFunc(mouse);

	//glutMainLoop();
	
	//std::ifstream in((argc > 1) ? argv[1] : "data/triangulation_prog1.cin");
	//std::istream_iterator<Point> begin(in);
	//std::istream_iterator<Point> end;
	//Triangulation t;
	//t.insert(begin, end);
	//points_triangulation(t);
	////CGAL::draw(t);
	//return EXIT_SUCCESS;

	/*vertices.push_back(Point(0, 0));
	vertices.push_back(Point(1, 1));
	vertices.push_back(Point(0, 1));
	vertices.push_back(Point(1, 0));

	Delaunay dt;
	dt.insert(vertices.begin(), vertices.end());
	Delaunay::Face_iterator i;
	Delaunay::Face face;
	cout << "Number of face : " << dt.number_of_faces() << std::endl;
	for (i = dt.faces_begin(); i != dt.faces_end(); i++) {
		face = *i;
	}
	write_to_obj(dt);*/

	RoadMeshReconstruction roadMeshReconstruction;
	roadMeshReconstruction.init();
	roadMeshReconstruction.startReconstruction();
	/*MeshReconstruction meshReconstruction;
	meshReconstruction.init();
	meshReconstruction.startReconsturction();*/
}