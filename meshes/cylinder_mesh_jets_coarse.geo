// ---- 2D Circular Cylinder Gmsh Tutorial ----
// 2D_cylinder_tutorial.geo
// Creates a mesh with an inner structured-quad region and 
// an outer unstructured tri region
//
// Created 11/26/2014 by Jacob Crabill
// Aerospace Computing Lab, Stanford University
// --------------------------------------------

alpha = Pi/2;
js = Pi/10;

// Gmsh allows variables; these will be used to set desired
// element sizes at various Points
//cl1 = 6;
cl1 = 10;
//cl2 = .02;
cl2 = .03;
//cl3 = 5;
cl3 = 8;
//cl4 = .2;
cl4 = .3;

//nRadial = 13;
nRadial = 11;
//nTagent = 20;
ppr = 51 / (2*Pi); // Points per radian on the cirlce
np_back = Ceil(ppr*(2*Pi - 2*alpha + js)); // Back half of the cylinder
np_jets = Ceil(ppr*js); // Number of points on the jets
np_front = Ceil(ppr*(2*alpha - js)); // Number of points in front between the jets

// Interior box of mesh
Point(1) = {-2, -2, 0, cl4};
Point(2) = { 9, -2, 0, cl4};
Point(3) = { 9,  2, 0, cl4};
Point(4) = {-2,  2, 0, cl4};
Line(1) = {1, 2};
Line(2) = {2, 3};
Line(3) = {3, 4};
Line(4) = {4, 1};

// Inner Circle
// Top and bottom points
Point(5) = {0,   0, 0, cl2};
// Top Jet
Point(8) = {-0.5*Cos(alpha + js/2), 0.5*Sin(alpha + js/2), 0, cl2};
Point(9) = {-0.5*Cos(alpha - js/2), 0.5*Sin(alpha - js/2), 0, cl2};
// Bottom jet
Point(10) = {-0.5*Cos(alpha - js/2), -0.5*Sin(alpha - js/2), 0, cl2};
Point(11) = {-0.5*Cos(alpha + js/2), -0.5*Sin(alpha + js/2), 0, cl2};

Circle(5) = {11, 5, 8}; // Back half of the cylinder
Circle(7) = {8, 5, 9}; // Top jet
Circle(8) = {9, 5, 10}; // In between jets
Circle(9) = {10, 5, 11}; // bottom jet


// Outer Circle
// Top Jet
Point(14) = {-1.25*Cos(alpha + js/2), 1.25*Sin(alpha + js/2), 0, cl2};
Point(15) = {-1.25*Cos(alpha - js/2), 1.25*Sin(alpha - js/2), 0, cl2};
// Bottom jet
Point(16) = {-1.25*Cos(alpha - js/2), -1.25*Sin(alpha - js/2), 0, cl2};
Point(17) = {-1.25*Cos(alpha + js/2), -1.25*Sin(alpha + js/2), 0, cl2};

Circle(11) = {14, 5, 17}; // Back half of the cylinder
Circle(13) = {14, 5, 15}; // Top jet
Circle(14) = {15, 5, 16}; // In between jets
Circle(15) = {16, 5, 17}; // bottom jet


Line(23) = {8, 14};
Line(24) = {9, 15};
Line(25) = {10, 16};
Line(26) = {11, 17};


Transfinite Line {5, 11} = np_back; // Back of cylinder
Transfinite Line {7,9,13,15} = np_jets; // Jets
Transfinite Line {8, 14} = np_front; // In between jets

Transfinite Line {23,24,25,26} = nRadial Using Progression 1.2;    // And 10 points along each of these lines


// Exterior (bounding box) of mesh
Point(18) = {-30, -30, 0, cl1};
Point(19) = { 50, -30, 0, cl3};
Point(20) = { 50,  30, 0, cl3};
Point(21) = {-30,  30, 0, cl1};
Line(19) = {18, 19};
Line(20) = {19, 20};
Line(21) = {20, 21};
Line(22) = {21, 18};


Line Loop(27) = {14, -25, -8, 24};
Plane Surface(28) = {27};
Line Loop(29) = {13, -24, -7, 23};
Plane Surface(30) = {29};
Line Loop(31) = {-11, 26, -5, -23};
Plane Surface(32) = {31};
Line Loop(33) = {-9, -26, 15, 25};
Plane Surface(34) = {33};

Transfinite Surface {28, 30, 32, 34};
Recombine Surface {28, 30, 32, 34};

Line Loop(35) = {4, 1, 2, 3};
Line Loop(36) = {14, 15, -11, 13};
Plane Surface(37) = {35, 36};
Line Loop(38) = {-21, -22, -19, -20};
Plane Surface(39) = {35, 38};


Recombine Surface {37, 39};

Physical Surface("FLUID") = {39, 37, 32, 30, 28, 34};
Physical Line("wall") = {5, 8};
Physical Line("jet1") = {7};
Physical Line("jet2") = {9};
Physical Line("farfield") = {19, 22, 21, 20};
