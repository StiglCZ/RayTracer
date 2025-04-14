#ifndef __OPENCL_VERSION__
#define __kernel
#define __global
#define get_global_id(x) x
#define get_global_size(x) x
#include <math.h>
#include <stdio.h>
#define FLT_EPSILON 1.000f
#endif

typedef unsigned char u8;
typedef struct Vector2 Vector2;
typedef struct Vector3 Vector3;
typedef struct Triangle Triangle;

struct __attribute__((packed)) Vector2 { float X, Y; };
struct __attribute__((packed)) Vector3 { float X, Y, Z; };
struct __attribute__((packed)) Triangle { Vector3 t1, t2, t3; };

Vector3 sub(Vector3 a, Vector3 b) { return (Vector3){ a.X - b.X, a.Y - b.Y, a.Z - b.Z, }; }
float dot(Vector3 a, Vector3 b) { return a.X * b.X + a.Y * b.Y + a.Z * b.Z; }

Vector3 cross(Vector3 a, Vector3 b) {
    return (Vector3) {
        a.Y * b.Z - a.Z * b.Y,
        a.Z * b.X - a.X * b.Z,
        a.X * b.Y - a.Y * b.X,
    };
}

__kernel void Main(__global const Triangle *SceneInput,
                   __global u8 *ScreenOutput,
                   Vector2 Step, int Triangles) {
    const int
        X = get_global_id(0),
        Y = get_global_id(1),
        W = get_global_size(0),
        H = get_global_size(1);
    
    Vector2 Angle = {
        Step.X * (X - W / 2.0f),
        Step.Y * (Y - H / 2.0f),
    };
    
    Vector3 Ray = {
        cos(Angle.Y) * sin(Angle.X),
        sin(Angle.Y) * cos(Angle.X),
        cos(Angle.Y) * cos(Angle.X),
    };
    
    u8 r = 0, g = 0, b = 0;
    float Min_D = 10000;
    for(int i = 0; i < Triangles; i++) {
        Triangle t = SceneInput[i];

        Vector3
            edge1 = sub(t.t2, t.t1),
            edge2 = sub(t.t3, t.t1);
        
        Vector3 
            s     = sub((Vector3){0, 0, 0}, t.t1),
            cross1= cross(s,   edge1),
            cross2= cross(Ray, edge2);
        float det = dot(edge1, cross2);
        if(det < 0) continue;
        
        float
            u = dot(s, cross2) / det,
            v = dot(Ray, cross1) / det,
            d = dot(edge2, cross1) / det;
        
        if(u < 0 || v < 0 || u + v > 1) continue;
        if(d < 0.0001 || d > Min_D) continue;
        Min_D = d;
        r = (u8)(5131283356366124 * i + i * 12093);
        g = (u8)(51334645136645 * i + i * 12093);
        b = (u8)(51256333146631 * i + i * 12093);
    }
    float value = (1.5f - log(Min_D)) * (1.5 - log(Min_D));
    ScreenOutput[(X + Y * W) * 3 + 0] = r * value;
    ScreenOutput[(X + Y * W) * 3 + 1] = g * value;
    ScreenOutput[(X + Y * W) * 3 + 2] = b * value;
}
