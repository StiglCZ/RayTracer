#include <CL/cl.h>
#include <CL/opencl.hpp>
#include <cmath>
#include <error.h>
#include <fstream>
#include <ios>
#include <iostream>
#include <string>
#include "program2.hh"

const int
    W = 3840,
    H = 2160,
    C = 3;

const Vector3
    Vector3::UnitX = {1, 0, 0},
    Vector3::UnitY = {0, 1, 0},
    Vector3::UnitZ = {0, 0, 1},
    Vector3::Zero  = {0, 0, 0},
    Vector3::One   = {1, 1, 1};

CLData::CLData(int SceneLen, std::string FileName) {
    Device = cl::Device::getDefault(&Err);
    if(Err != CL_SUCCESS) {
        Succeeded = -1;
        return;
    }
    
    Reader = std::ifstream(FileName, std::ios::ate);
    if(!Reader.is_open()) {
        Succeeded = -2;
        return;
    }
    
    ProgramSize = Reader.tellg();
    Reader.seekg(0, std::ios::beg);
    Source = std::string(ProgramSize, '\0');
    Reader.read(Source.data(), ProgramSize);
    Reader.close();

    Global = cl::NDRange(W, H);
    Local  = cl::NDRange(1, 1);
    Offset = cl::NullRange;
    
    Sources.push_back({Source.c_str(), Source.length() + 1});
    Context = cl::Context(Device);
    Program = cl::Program(Context, Sources);
    
    if(Program.build() != CL_BUILD_SUCCESS) {
        Succeeded = -3;
        BuildLogs = Program.getBuildInfo<CL_PROGRAM_BUILD_LOG>()[0].second;
        return;
    }
    BuildLogs = Program.getBuildInfo<CL_PROGRAM_BUILD_LOG>()[0].second;
    Input = cl::Buffer(Context, CL_MEM_READ_ONLY  | CL_MEM_HOST_WRITE_ONLY, sizeof(Triangle) * SceneLen);
    Output = cl::Buffer(Context, CL_MEM_WRITE_ONLY |  CL_MEM_HOST_READ_ONLY, W * H * C * sizeof(u8));
    Kernel = cl::Kernel(Program, "Main");
    Queue = cl::CommandQueue(Context, Device);
    Succeeded = 0;
}

Model LoadModel(std::string ModelFile) {
    std::string line;
    std::ifstream ifs(ModelFile);
    std::vector<Vector3> verts;
    Model result;
    while(std::getline(ifs, line)) {
        if(line.empty()) continue;
        char* str = (char*)line.c_str() + 2;
        if(line[0] == 'v') {
            Vector3 vec = {strtof(str, &str), strtof(str, &str), strtof(str, &str)};
            verts.push_back(vec);
        } else if(line[0] == 'f')
            result.push_back((Triangle){
                    verts[(int)strtol(str, &str, 0) -1],
                    verts[(int)strtol(str, &str, 0) -1],
                    verts[(int)strtol(str, &str, 0) -1],
                });
    } return result;
}

void Export(u8* ImageData) {
    std::ofstream out("output.ppm");
    out << "P6\n"
        << std::to_string(W) << ' ' << std::to_string(H) << "\n"
        << "255\n";
    
    if(C == 3) out.write((char*)ImageData, W * H * C);
    else for(int i =0; i < W * H; i++)
             out.write((char*)ImageData + i * C, 3);
    out.close();
}

int main() {
    Model Scene = LoadModel("Untitled.obj");
    CLData ClData(Scene.size(), "Tracer.c");

    if(ClData.Succeeded)
        error(1, 0, "OpenCL failed with\n\tCode: %d\n\tCLCode: %d\n\tBuild Log:%s\n",
              ClData.Succeeded, ClData.Err, ClData.BuildLogs.c_str());    
    float FOV = M_PI_2f,
        Step = FOV / W;

    ClData.Kernel.setArg(0, ClData.Input);
    ClData.Kernel.setArg(1, ClData.Output);
    ClData.Kernel.setArg(2, (Vector2){Step, Step});
    ClData.Kernel.setArg(3, (int)Scene.size());
    ClData.Kernel.setArg(4, (Vector3){0, -3, -10}); // Origin

    u8* Buffer = new u8[W * H * C];
    ClData.Queue.enqueueWriteBuffer(ClData.Input, CL_TRUE, 0, sizeof(Triangle) * Scene.size(), Scene.data());
    ClData.Queue.enqueueNDRangeKernel(ClData.Kernel, ClData.Offset, ClData.Global, ClData.Local);
    ClData.Queue.enqueueReadBuffer(ClData.Output, CL_TRUE, 0, W * H * C, Buffer);
    Export(Buffer);
    delete[] Buffer;
}
