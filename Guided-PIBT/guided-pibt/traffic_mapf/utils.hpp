
#ifndef utils_hpp
#define utils_hpp

#include "Types.h"

namespace TrafficMAPF{
int get_d(int diff, const SharedEnvironment* env)  ;

int get_id_from_d(int d, int curr_id, const SharedEnvironment* env);

bool validateMove(int loc, int loc2, const SharedEnvironment* env);

int manhattanDistance(int loc, int loc2,const SharedEnvironment* env);

void getNeighbors(const SharedEnvironment* env, std::vector<std::pair<int,int>>& neighbors, int location,int direction) ;

void getNeighbors_nowait(const SharedEnvironment* env, std::vector<std::pair<int,int>>& neighbors, int location,int direction) ;

void getNeighborLocs(const SharedEnvironment* env, std::vector<int>& neighbors, int location) ;

void getNeighborLocs(const SharedEnvironment* env, int neighbors[], int location) ;
}

#endif