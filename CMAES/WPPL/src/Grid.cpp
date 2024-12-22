#include "Grid.h"
#include <boost/tokenizer.hpp>

Grid::Grid(string fname,
           double left_w_weight,
           double right_w_weight)
{
    load_map_from_path(fname, left_w_weight, right_w_weight);
}

Grid::Grid(nlohmann::json map_json,
           double left_w_weight,
           double right_w_weight)
{
    load_map_from_json(map_json, left_w_weight, right_w_weight);
}

void Grid::load_map_from_path(string fname,
                              double left_w_weight,
                              double right_w_weight)
{
    std::string line;
    std::ifstream myfile((fname).c_str());
    if (!myfile.is_open())
    {
        cout << "Map file " << fname << " does not exist. " << std::endl;
        exit(-1);
    }

    cout << "*** Loading map ***" << std::endl;
    clock_t t = std::clock();
    size_t pos = fname.rfind('.');   // position of the file extension
    map_name = fname.substr(0, pos); // get the name without extension
    getline(myfile, line);

    if (line[0] == 't')
    {
        // Benchmark
        boost::char_separator<char> sep(" ");
        getline(myfile, line);
        boost::tokenizer<boost::char_separator<char>> tok(line, sep);
        boost::tokenizer<boost::char_separator<char>>::iterator beg;
        beg = tok.begin();
        beg++;
        rows = atoi((*beg).c_str()); // read number of rows
        getline(myfile, line);
        boost::tokenizer<boost::char_separator<char>> tok2(line, sep);
        beg = tok2.begin();
        beg++;
        cols = atoi((*beg).c_str()); // read number of cols
        getline(myfile, line);       // skip "map"
    }
    else
    {
        boost::char_separator<char> sep(",");
        boost::tokenizer<boost::char_separator<char>> tok(line, sep);
        boost::tokenizer<boost::char_separator<char>>::iterator beg = tok.begin();
        rows = atoi((*beg).c_str()); // read number of rows
        beg++;
        cols = atoi((*beg).c_str()); // read number of cols
    }

    map.resize(cols * rows, 0);

    // DeliverGoal.resize(row*col, false);
    //  read map
    // int ep = 0, ag = 0;
    for (int i = 0; i < rows; i++)
    {
        getline(myfile, line);
        for (int j = 0; j < cols; j++)
        {
            int id = cols * i + j;
            grid_types.push_back(line[j]);
            if (line[j] == '@' || line[j] == 'T') // obstacle
                map[id] = 1;
            else
            { // free space
                map[id] = 0;
                empty_locations.push_back(id);
            }

            if (line[j] == 'e')
            {
                end_points.push_back(id);
            }

            if (line[j] == 'w')
            {
                agent_home_locations.push_back(id);

                // Add weights to workstations s.t. one side of the
                // workstations are more "popular" than the other
                if (j == 0)
                {
                    this->agent_home_loc_weights.push_back(left_w_weight);
                }
                else
                {
                    this->agent_home_loc_weights.push_back(right_w_weight);
                }
            }
        }
    }

    myfile.close();
    double runtime = (std::clock() - t) / CLOCKS_PER_SEC;
    cout << "Map size: " << rows << "x" << cols << ", empty locs: "
         << empty_locations.size() << ", end_points: " << end_points.size()
         << ", agent_homes (workstations): " << agent_home_locations.size()
         << endl;
    cout << "Done! (load time: " << runtime << " s)" << std::endl;
}

void Grid::load_map_from_json(nlohmann::json map_json,
                              double left_w_weight,
                              double right_w_weight)
{
    std::cout << "*** Loading map ***" << std::endl;
    clock_t t = std::clock();

    // Read in n_row, n_col, n_agent_loc, maxtime
    this->rows = map_json["n_row"];
    this->cols = map_json["n_col"];

    this->map_name = map_json["name"];

    this->map.resize(cols * rows, 0);

    std::string line;

    for (int i = 0; i < this->rows; i++)
    {
        line = map_json["layout"][i];
        for (int j = 0; j < this->cols; j++)
        {
            int id = cols * i + j;
            grid_types.push_back(line[j]);
            if (line[j] == '@' || line[j] == 'T') // obstacle
                map[id] = 1;
            else
            { // free space
                map[id] = 0;
                empty_locations.push_back(id);
            }

            if (line[j] == 'e')
            {
                end_points.push_back(id);
            }

            if (line[j] == 'w')
            {
                agent_home_locations.push_back(id);

                // Add weights to workstations s.t. one side of the
                // workstations are more "popular" than the other
                if (j == 0)
                {
                    this->agent_home_loc_weights.push_back(left_w_weight);
                }
                else
                {
                    this->agent_home_loc_weights.push_back(right_w_weight);
                }
            }
        }
    }

    double runtime = (std::clock() - t) / CLOCKS_PER_SEC;
    cout << "Map size: " << rows << "x" << cols << ", empty locs: "
         << empty_locations.size() << ", end_points: " << end_points.size()
         << ", agent_homes (workstations): " << agent_home_locations.size()
         << endl;
    cout << "Done! (load time: " << runtime << " s)" << std::endl;
}
