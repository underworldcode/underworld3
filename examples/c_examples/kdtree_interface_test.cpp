#include "kdtree_interface.hpp"
#include <iostream>
#include <cmath>

double dist(double* p1, double* p2, int dim)
{
    double distance = 0;
    for (size_t i=0; i<dim; i++)
        distance += pow(p2[i]-p1[i],2);
    return sqrt(distance);
}
int main()
{
    int n = 500000;
    int dim = 3;
    double* pts = (double*) malloc(sizeof(double)*n*dim);
    double coord[3] = {0.5,0.5,0.5};
    double distmin = 9999999;
    int distminp=-1;

	for (size_t i = 0; i < n;i++)
	{
        pts[i*dim + 0] = (double)rand()/RAND_MAX;
        pts[i*dim + 1] = (double)rand()/RAND_MAX;
        pts[i*dim + 2] = (double)rand()/RAND_MAX;
        double distp = dist(&pts[i*dim],(double*)&coord,dim);
        // std::cout << "Point " << i <<  " added "<< pts[i*dim+0] << "," << pts[i*dim+1] << " with distance " << distp << "\n";
        if (distp<distmin) 
        {   
            distmin=distp;
            distminp = i;
        }
	}
    std::cout << "Particle " << distminp << " is closest with distance " << distmin << "\n";
    auto index = KDTree_Interface( pts, n, dim );

	for (size_t i = 0; i < n; i++)
	{
        coord[0] = (double)rand()/RAND_MAX;
        coord[1] = (double)rand()/RAND_MAX;
        coord[2] = (double)rand()/RAND_MAX;

        size_t result = index.find_closest_point(coord, 1);
        // std::cout << "Found index is "<< result <<"\n";
        // std::cout << "Found index coord is "<< pts[result*dim+0] << "," << pts[result*dim+1]<< " with distance " << dist(&pts[result*dim],(double*)&coord,dim) << "\n";
    }
    return 0;
}