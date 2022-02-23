#include "nanoflann.hpp"

struct PointCloudAdaptor
{
	const double* obj;
    int numpoints;
    int dim;

	/// The constructor that sets the data set source
	PointCloudAdaptor(const double* obj, int numpoints, int dim) : obj(obj), numpoints(numpoints), dim(dim) { };

	// Must return the number of data points
	inline size_t kdtree_get_point_count() const { return numpoints; };

	// Returns the dim'th component of the idx'th point in the class:
	// Since this is inlined and the "dim" argument is typically an immediate value, the
	//  "if/else's" are actually solved at compile time.
	inline double kdtree_get_pt(const size_t idx, const size_t dof) const
	{
		return obj[idx*dim + dof];
	}

	// Optional bounding-box computation: return false to default to a standard bbox computation loop.
	//   Return true if the BBOX was already computed by the class and returned in "bb" so it can be avoided to redo it again.
	//   Look at bb.size() to find out the expected dimensionality (e.g. 2 or 3 for point clouds)
	template <class BBOX>
	bool kdtree_get_bbox(BBOX& /*bb*/) const { return false; }

}; // end of PointCloudAdaptor

// construct a kd-tree index
typedef nanoflann::KDTreeSingleIndexAdaptor<
    nanoflann::L2_Simple_Adaptor<double, PointCloudAdaptor > ,
    PointCloudAdaptor,
    2 /* dim */
    > tree_2d;
typedef nanoflann::KDTreeSingleIndexAdaptor<
    nanoflann::L2_Simple_Adaptor<double, PointCloudAdaptor > ,
    PointCloudAdaptor,
    3 /* dim */
    > tree_3d;

class KDTree_Interface
{
    
    int dim=0;
    PointCloudAdaptor* pca=nullptr;
    tree_2d* index2d=nullptr;
    tree_3d* index3d=nullptr;
    public:
        KDTree_Interface( const double* points, int numpoints, int dim ) : dim(dim), pca(new PointCloudAdaptor(points, numpoints, dim)) 
        { 
            if(dim==2)
            {
                index2d = new tree_2d(dim, *pca, nanoflann::KDTreeSingleIndexAdaptorParams(10 /* max leaf */) );
                index2d->buildIndex();
            }
            else
            {
                index3d = new tree_3d(dim, *pca, nanoflann::KDTreeSingleIndexAdaptorParams(10 /* max leaf */) );
                index3d->buildIndex();
            }	
        };
        ~KDTree_Interface()
        {
            delete(index3d);
            index3d=nullptr;
            delete(index2d);
            index3d=nullptr;
            delete(pca);
            pca=nullptr;
        };
        void build_index()
        {
            if(dim==2)
                index2d->buildIndex();
            else
                index3d->buildIndex();
        };
        void find_closest_point( size_t num_coords, const double* coords, long unsigned int* indices, double* out_dist_sqr, bool* found )
        {
            double dist;
            nanoflann::KNNResultSet<double> resultSet(1);
            for (size_t item=0; item<num_coords; item++ )
            {
                resultSet.init( &indices[item], &dist );
                bool founditem;
                if(dim==2)
                    founditem = index2d->findNeighbors(resultSet, &coords[item*dim], nanoflann::SearchParams(10)); // note that I believe the value 10 here is ignored.. i'll retain it as it's used in the examples
                else
                    founditem = index3d->findNeighbors(resultSet, &coords[item*dim], nanoflann::SearchParams(10));  // See line 561 of .hpp, not used but 
                                                                                                                    // if you want to set other args you'll need to be aware of it
                if (out_dist_sqr!=NULL) out_dist_sqr[item] = dist; 
                if (       found!=NULL)        found[item] = founditem; 
            }
        }; 

        size_t knnSearch(const double* query_point, const size_t num_closest, long unsigned int* indices, double* out_dist_sqr ) {
          if( dim == 2 )
           return index2d->knnSearch( query_point, num_closest, indices, out_dist_sqr ); 
          else
           return index3d->knnSearch( query_point, num_closest, indices, out_dist_sqr ); 
        };
 
};


// todo ... add interface to the knnSearch routines to find a neighbour cloud. 
