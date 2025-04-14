#ifndef PROGRAM2_HH
#define PROGRAM2_HH

#include <CL/opencl.hpp>
#include <fstream>
#include <vector>

typedef unsigned char u8;
struct __attribute__((packed)) Vector2 { float X, Y; };
struct Vector3 {
    float X, Y, Z;
    friend Vector3 operator+(const Vector3& lhs, const Vector3& rhs) {
        return { lhs.X + rhs.X, lhs.Y + rhs.Y, lhs.Z + rhs.Z };
    }
    friend Vector3 operator-(const Vector3& lhs, const Vector3& rhs) {
        return { lhs.X - rhs.X, lhs.Y - rhs.Y, lhs.Z - rhs.Z };
    }
    friend Vector3 operator*(const Vector3& lhs, const Vector3& rhs) {
        return { lhs.X * rhs.X, lhs.Y * rhs.Y, lhs.Z * rhs.Z };
    }
    friend Vector3 operator/(const Vector3& lhs, const Vector3& rhs) {
        return { lhs.X / rhs.X, lhs.Y / rhs.Y, lhs.Z / rhs.Z };
    }
    friend bool operator==(const Vector3& lhs, const Vector3& rhs) {
        return lhs.X == rhs.X && lhs.Y == rhs.Y && lhs.Z == rhs.Z;
    }
    friend bool operator!=(const Vector3& lhs, const Vector3& rhs) {
        return lhs.X != rhs.X && lhs.Y != rhs.Y && lhs.Z != rhs.Z;
    }

    static const Vector3 UnitX,
        UnitY,
        UnitZ,
        Zero,
        One;
} __attribute__ ((packed));

struct Triangle {
    Vector3 t1, t2, t3;
} __attribute__((packed));

struct CLData {
    public:
    int Err;
    std::ifstream Reader;
    std::string Source;
    cl::Program::Sources Sources;
    cl::Program Program;
    cl::Kernel Kernel;
    
    cl::CommandQueue Queue;
    cl::NDRange Global, Local, Offset;
    cl::Context Context;
    cl::Device Device;
    std::string BuildLogs;
    int Succeeded, ProgramSize;
    cl::Buffer Input, Output;
    CLData(int SceneLen, std::string SourceFile);
};

typedef std::vector<Triangle> Model;
#endif
