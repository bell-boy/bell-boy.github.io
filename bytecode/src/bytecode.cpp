#include "transpiler.h"
#include <iostream>

int main(int argc, char *argv[]) {
    if(argc != 3)
    {
        std::cerr << "not enough arguments!\n"; 
        return 1;
    }

    std::string input_path(argv[1]);
    auto AST = CompressedAST(FileParser(input_path));

    std::string finished_html = "";
    for(auto node : AST) finished_html += buildNode(node) + "\n";  

    std::ifstream template_file("./src/template.html");
    std::vector<std::string> template_vector;
    std::string last_line;
    while (getline(template_file, last_line)) {
        template_vector.push_back(last_line);
    }

    template_vector[27] = finished_html;

    std::string output_path(argv[2]);
    std::ofstream outfile(output_path);

    for(auto s : template_vector) outfile << s << std::endl;
}