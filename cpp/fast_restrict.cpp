#include <boost/python.hpp>
#include <boost/python/list.hpp>
#include <vector>
#include <iostream> 
#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/connected_components.hpp>
#include <chrono>

// np_to_vec
std::vector<int> np_to_vec(const long ptr, const int len_arr) {
    long* element_ptr = (long*) ptr;
    std::vector<int> cpp_vector;
    cpp_vector.reserve(len_arr);
    for (int i = 0; i < len_arr; ++i){
        cpp_vector.push_back(*element_ptr);
        // std::cout << *element_ptr << " ";
        ++element_ptr;
    }
    // std::cout << std::endl;
    return cpp_vector;
}


// list_to_vec
std::vector<int> list_to_vec(boost::python::list py_list)
{
    std::vector<int> cpp_vector;
    for (int i = 0; i < len(py_list); ++i) {
        boost::python::extract<int> element(py_list[i]);
        if (element.check())
            cpp_vector.push_back(element());
        // else { // Handle error if the element is not an integer}
    }
    return cpp_vector;
}

// vec_to_list
boost::python::list vec_to_list(const std::vector<int>& cpp_vector)
{
    boost::python::list py_list;

    for (const auto& element : cpp_vector) {
        py_list.append(element);
    }

    return py_list;
}


typedef boost::adjacency_list<boost::vecS, boost::vecS, boost::undirectedS> Graph;

std::vector<int> find_connected_components(const Graph& graph) {

    std::vector<int> component_map(boost::num_vertices(graph));
    boost::connected_components(graph, &component_map[0]);
    // std::cout << "Number of connected components: " << num_components << std::endl;

    return component_map;
}

bool in_nodes_to_remove(int* check_this_node, const std::vector<int>& nodes_to_remove)
{
    for (int i = 0; i < nodes_to_remove.size(); ++i) {
        if (nodes_to_remove[i] == *check_this_node) {
            return true;
        }
    }
    return false;
}

boost::python::list get_connected_components_from_list(int num_nodes, 
                        boost::python::list py_edge_from, 
                        boost::python::list py_edge_to, 
                        boost::python::list nodes_to_remove)
{
    // greate graph with all nodes
    Graph graph(num_nodes);

    // turn edge_index and nodes to remove to vectors
    std::vector<int> cpp_edge_from = list_to_vec(py_edge_from);
    std::vector<int> cpp_edge_to = list_to_vec(py_edge_to);
    std::vector<int> cpp_nodes_to_remove = list_to_vec(nodes_to_remove);

    // iterate over every edge
    for (int i = 0; i < cpp_edge_from.size(); ++i) {
        int n1 = cpp_edge_from[i];
        int n2 = cpp_edge_to[i];
        bool remove_n1 = in_nodes_to_remove(&n1, cpp_nodes_to_remove);
        bool remove_n2 = in_nodes_to_remove(&n2, cpp_nodes_to_remove);
            // only add edge if node is not in cpp_nodes_to_remove
            if (!remove_n1 && !remove_n2) {
                boost::add_edge(n1, n2, graph);
                // std::cout << "Added edge: " << n1 << " <---> " << n2 << std::endl;
            }
            else {
                // std::cout << "No edge:    " << n1 << " <-X-> " << n2 << std::endl;
        }
    }

    std::vector<int> component_map = find_connected_components(graph);
    // for (int i = 0; i < component_map.size(); ++i) {
    //     std::cout << "Node " << i << " belongs to component " << component_map[i] << std::endl;
    // }

    return vec_to_list(component_map);
}

boost::python::list get_connected_components(const int num_nodes, 
                                        const int num_edges,
                                        const long edge_from_ptr, 
                                        const long edge_to_ptr, 
                                        boost::python::list nodes_to_remove)
{
    // greate graph with all nodes
    Graph graph(num_nodes);

    // turn edge_index and nodes to remove to vectors
    std::vector<int> cpp_edge_from = np_to_vec(edge_from_ptr, num_edges);
    std::vector<int> cpp_edge_to = np_to_vec(edge_to_ptr, num_edges);
    std::vector<int> cpp_nodes_to_remove = list_to_vec(nodes_to_remove);

    // iterate over every edge
    for (int i = 0; i < cpp_edge_from.size(); ++i) {
        int n1 = cpp_edge_from[i];
        int n2 = cpp_edge_to[i];
        bool remove_n1 = in_nodes_to_remove(&n1, cpp_nodes_to_remove);
        bool remove_n2 = in_nodes_to_remove(&n2, cpp_nodes_to_remove);
            // only add edge if node is not in cpp_nodes_to_remove
            if (!remove_n1 && !remove_n2) {
                boost::add_edge(n1, n2, graph);
                // std::cout << "Added edge: " << n1 << " <---> " << n2 << std::endl;
            }
            else {
                // std::cout << "No edge:    " << n1 << " <-X-> " << n2 << std::endl;
        }
    }

    std::vector<int> component_map = find_connected_components(graph);
    // for (int i = 0; i < component_map.size(); ++i) {
    //     std::cout << "Node " << i << " belongs to component " << component_map[i] << std::endl;
    // }

    return vec_to_list(component_map);
}



BOOST_PYTHON_MODULE(cpp_graph_divide)
{
    using namespace boost::python;
    def("get_connected_components_from_list", get_connected_components_from_list);
    def("get_connected_components", get_connected_components);
}