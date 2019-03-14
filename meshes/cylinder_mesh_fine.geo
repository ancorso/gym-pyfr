// ---- 2D Circular Cylinder Gmsh Tutorial ----
// 2D_cylinder_tutorial.geo
// Creates a mesh with an inner structured-quad region and 
// an outer unstructured tri region
//
// Created 11/26/2014 by Jacob Crabill
// Aerospace Computing Lab, Stanford University
// --------------------------------------------

// Gmsh allows variables; these will be used to set desired
// element sizes at various Points
cl1 = 6;
cl2 = .02;
cl3 = 5;
cl4 = .2;

nRadial = 13;
nTagent = 20;

// Interior box of mesh
Point(1) = {-2, -2, 0, cl4};
Point(2) = { 9, -2, 0, cl4};
Point(3) = { 9,  2, 0, cl4};
Point(4) = {-2,  2, 0, cl4};
Line(1) = {1, 2};
Line(2) = {2, 3};
Line(3) = {3, 4};
Line(4) = {4, 1};

// Circle & surrounding structured-quad region
Point(5) = {0,   0, 0, cl2};
Point(6) = {0,  .5, 0, cl2};
Point(7) = {0, -.5, 0, cl2};
Point(8) = {0,  1.25, 0, cl2};
Point(9) = {0, -1.25, 0, cl2};
Circle(5) = {7, 5, 6};
Circle(6) = {6, 5, 7};
Circle(7) = {8, 5, 9};
Circle(8) = {9, 5, 8};
Line(9)  = {6, 8};
Line(10) = {7, 9};
Transfinite Line {5,6,7,8} = nTagent; // We want 40 points along each of these lines
Transfinite Line {9,10} = nRadial Using Progression 1.2;    // And 10 points along each of these lines

// Exterior (bounding box) of mesh
Point(10) = {-30, -30, 0, cl1};
Point(11) = { 50, -30, 0, cl3};
Point(12) = { 50,  30, 0, cl3};
Point(13) = {-30,  30, 0, cl1};
Line(11) = {10, 11};
Line(12) = {11, 12};
Line(13) = {12, 13};
Line(14) = {13, 10};

// Each region which to be independantly meshed must have a line loop
// Regions which will be mesh with Transfinite Surface must have 4 lines
// and be labeled in CCW order, with the correct orientation of each edge
Line Loop(1) = {1, 2, 3, 4, 7, 8}; // Inner Exterior
Line Loop(2) = {10, 8, -9, -5}; // RH side of quad region - note ordering
Line Loop(3) = {7, -10, -6, 9}; // LH side of quad region - note ordering
Line Loop(4) = {11,12,13,14};   // Outer Exterior

Plane Surface(1) = {1}; // Outer unstructured region
Plane Surface(2) = {2}; // RH inner structured region
Plane Surface(3) = {3}; // LH inner structured region
Plane Surface(4) = {4,-1}; // Outer outer region

// Mesh these surfaces in a structured manner
Transfinite Surface{2,3};

// Turn into quads (optional, but Transfinite Surface looks best with quads)
Recombine Surface {2,3};
// Turn outer region into unstructured quads (optional)
Recombine Surface {1,4};

// Apply boundary conditions
// Note: Can change names later at top of .msh file
Physical Line("Bottom") = {11};
Physical Line("Right")  = {12};
Physical Line("Top")    = {13};
Physical Line("Left")   = {14};
Physical Line("Circle") = {5,6};
// Alternate version - make all 4 outer bounds part of the same B.C.:
//Physical Line("Char") = {1,2,3,4}; 

// IMPORTANT: "FLUID" MUST contain all fluid surfaces(2D)/volumes(3D)
Physical Surface("FLUID") = {1,2,3,4};
