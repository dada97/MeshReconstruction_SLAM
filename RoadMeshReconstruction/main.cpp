#include "RoadMeshReconstruction.h"
#include <fstream>

using namespace std::chrono;
using namespace std;

int main(int argc, char* argv[]) {
 
    std::vector<std::string> cmdLineArgs(argv, argv + argc);
    cout << argc << endl;
    bool debug_mode = false;
    string input_dir, output_dir;

    for (auto i = cmdLineArgs.begin(); i != cmdLineArgs.end(); ++i) {
        if (*i == "-h" || *i == "--help") {
            cout << "Command Example: RoadMeshReconstruction.exe -i [input] -o [output]" << endl;
            cout << "parameter\tvalue" << endl;
            cout << "-i/--input\tinput directory"<<endl;
            cout << "-o/--output\toutput directory" << endl;
            cout << "-d/--debug\tdebug mode(optional,default = false)" << endl;
            return 0;
        }
        else if (*i == "-i"||*i=="--input") {
            input_dir = *++i;
        }
        else if (*i == "-o"|| *i == "--output") {
            output_dir = *++i;
        }
        else if (*i == "-d"||* i == "--debug") {
            string needDebug = *++i;
            if (needDebug == "false") {
                debug_mode = false;
            }
            else if(needDebug == "true") {
                debug_mode = true;
            }
        }
    }

    if (argc != 3) {
        cout << "command error!" << endl;
        std::cout << "command : RoadMeshReconstruction [input path] [output path]" << endl;
    }

	auto start = high_resolution_clock::now();
	RoadMeshReconstruction roadMeshReconstruction(input_dir,output_dir,debug_mode);
	roadMeshReconstruction.init();
	roadMeshReconstruction.startReconstruction();
	auto stop = high_resolution_clock::now();
	auto duration = duration_cast<seconds>(stop - start);

	cout << "time :" << duration.count() << " seconds" << endl;
}