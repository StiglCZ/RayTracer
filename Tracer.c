#ifndef __OPENCL_VERSION__
#define __kernel
#define __global
#define __local
#define get_global_id(x) x
#define get_global_size(x) x
#include <math.h>
#include <stdio.h>
#define FLT_EPSILON 1.000f
#endif

typedef unsigned char u8;
typedef unsigned short u16;
typedef struct Ray Ray;
typedef struct Color Color;
typedef struct Intersection Intersection;
typedef struct Vector2 Vector2;
typedef struct Vector3 Vector3;
typedef struct Triangle Triangle;

struct __attribute__((packed)) Vector2 { float X, Y; };
struct __attribute__((packed)) Vector3 { float X, Y, Z; };
struct __attribute__((packed)) Triangle { Vector3 t1, t2, t3; };

Vector3 sub(Vector3 a, Vector3 b) { return (Vector3){ a.X - b.X, a.Y - b.Y, a.Z - b.Z, }; }
float dot(Vector3 a, Vector3 b) { return a.X * b.X + a.Y * b.Y + a.Z * b.Z; }
long Random(int seed) { return (seed * 0x5DEECE66DL + 0xBL) & ((1L << 48) - 1); }
float fRandom(int seed) {
    long r = Random(seed) % 0xFFFFFFFFFFFF;
    return (double)r / (double)0xFFFFFFFFFFFF;
}

Vector3 cross(Vector3 a, Vector3 b) {
    return (Vector3) {
        a.Y * b.Z - a.Z * b.Y,
        a.Z * b.X - a.X * b.Z,
        a.X * b.Y - a.Y * b.X,
    };
}

struct Color { u16 r, g, b, a; };
struct Ray {
    Vector3 src, dir;
    Color c;
    int Previous;
};
struct Intersection {
    int Triangle;
    float Distance;
};

float Magnitude(Vector3 src) {
    return sqrt(src.X * src.X + src.Y * src.Y + src.Z * src.Z);
}
Vector3 Normalize(Vector3 src) {
    float Mag = Magnitude(src);
    return (Vector3){
        src.X / Mag,
        src.Y / Mag,
        src.Z / Mag,
    };
}

Intersection Trace(__global const Triangle *SceneInput, int Triangles, Ray ray) {
    float Min_D = 10000;
    Vector3 dir = ray.dir;
    Intersection result;
    result.Triangle = -1;
    for(int i = 0; i < Triangles; i++) {
        if(ray.Previous == i) continue;
        Triangle t = SceneInput[i];
        
        Vector3
            edge1 = sub(t.t2, t.t1),
            edge2 = sub(t.t3, t.t1);
        
        Vector3 
            s     = sub(ray.src, t.t1),
            cross1= cross(s,   edge1),
            cross2= cross(dir, edge2);
        
        float det = dot(edge1, cross2);
        // if(det < 0) In theory, this should exist
        
        float
            u = dot(s, cross2) / det,
            v = dot(dir, cross1) / det,
            d = dot(edge2, cross1) / det;
        
        if(u < 0 || v < 0 || u + v > 1) continue;
        if(d < 0.0001 || d > Min_D) continue;
        Min_D = d;
        result.Distance = Min_D;
        result.Triangle = i;
    }
    return result;
}

Color GetTriangleColor(int TriangleIndex, Color result) {
    if(TriangleIndex != -1) {
        result.r += fRandom(Random(3189 * TriangleIndex * 3 + 1)) * 200 + 50;
        result.g += fRandom(Random(2209 * TriangleIndex * 3 + 2)) * 200 + 50;
        result.b += fRandom(Random(3913 * TriangleIndex * 3 + 3)) * 200 + 50;
        result.a ++;
    } return result;
}

Vector3 Reflection(Ray r, Triangle t, Intersection i) {
    Vector3 ab = sub(t.t2, t.t1), ac = sub(t.t3, t.t1);
    Vector3 abac = cross(ab, ac);
    Vector3 normal = Normalize(abac);
    float Dot = dot(r.dir, normal);
    return Normalize((Vector3) {
            r.dir.X - 2 * Dot * normal.X,
            r.dir.Y - 2 * Dot * normal.Y,
            r.dir.Z - 2 * Dot * normal.Z,
        });
}

inline Color TraceLoop(Ray ray, __global const Triangle *SceneInput, int Triangles) {
    int Counter = 0;
    Intersection intersect;
    while(Counter++ < 20 && (intersect = Trace(SceneInput, Triangles, ray)).Triangle != -1) {
        Ray new;
        new.src = (Vector3) {
            intersect.Distance * ray.dir.X + ray.src.X,
            intersect.Distance * ray.dir.Y + ray.src.Y,
            intersect.Distance * ray.dir.Z + ray.src.Z,
        };
        new.c = GetTriangleColor(intersect.Triangle, ray.c);
        new.dir = Reflection(ray, SceneInput[intersect.Triangle], intersect);
        new.Previous = intersect.Triangle;
        
        ray = new;
    }
    return ray.c;
}

__kernel void Main(__global const Triangle *SceneInput,
                   __global u8 *ScreenOutput,
                   Vector2 Step, int Triangles,
                  Vector3 origin) {
    const int
        X = get_global_id(0),
        Y = get_global_id(1),
        W = get_global_size(0),
        H = get_global_size(1);
    
    Vector2 Angle = { Step.X * (X - W / 2.0f), Step.Y * (Y - H / 2.0f) };
    Vector3 RayDirection = {
        cos(Angle.Y) * sin(Angle.X),
        sin(Angle.Y) * cos(Angle.X),
        cos(Angle.Y) * cos(Angle.X),
    };
    origin = (Vector3){
        origin.X + Angle.X / 1000.0f,
        origin.Y + Angle.Y / 1000.0f,
        origin.Z,
    };
    Ray ray = {origin, RayDirection, {0, 0, 0, 0}, -1};
    Color c = TraceLoop(ray, SceneInput, Triangles);
    u8 r = 0, g = 0, b = 0;
    
    if(c.a) {
        r = c.r / c.a;
        g = c.g / c.a;
        b = c.b / c.a;
    }
    
    ScreenOutput[(X + Y * W) * 3 + 0] = r;
    ScreenOutput[(X + Y * W) * 3 + 1] = g;
    ScreenOutput[(X + Y * W) * 3 + 2] = b;
}
