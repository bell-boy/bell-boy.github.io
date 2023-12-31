#include "parser.h"
#include <iostream>


std::vector<Node> LineParser(std::string line) {
    std::vector<Node> result;
    if(line == "") return result;

    std::string current_suffix;
    int offset;
    std::vector<TokenMatch> order;

    std::vector<TokenMatch> unchecked_matches;
    unchecked_matches.push_back(TokenMatch(Italic, std::regex("\\*"), 1));
    unchecked_matches.push_back(TokenMatch(Bold, std::regex("\\*{2}"), 2));
    unchecked_matches.push_back(TokenMatch(BoldItalic, std::regex("\\*{3}"), 3));

    for(auto &umatch : unchecked_matches) {
        offset = 0;
        current_suffix = line;
        std::smatch group_match;
        while ( std::regex_search(current_suffix, group_match, umatch.pattern) ) {
            if(umatch.start_position == -1) {
                umatch.start_position = group_match.position(0) + offset;
            } else if (umatch.end_position == -1) {
                umatch.end_position = group_match.position(0) + offset;
                order.push_back(umatch);
            } else break;
            offset += group_match.prefix().length() + umatch.seq_length;
            current_suffix = group_match.suffix().str();
        }
    }

    sort(order.begin(), order.end());
    if(order.empty()) result.push_back(Node(Text, line));
    else {
        TokenMatch top_group = order[0];
        std::vector<Node> prefix_recurse = LineParser(line.substr(0, top_group.start_position));
        result.insert(result.end(), prefix_recurse.begin(), prefix_recurse.end());

        Node group_node(top_group.value);
        std::vector<Node> middle_recurse = LineParser(line.substr(top_group.start_position + top_group.seq_length, top_group.end_position - top_group.start_position - top_group.seq_length));
        group_node.children = middle_recurse;
        
        result.push_back(group_node);
        
        if(top_group.end_position + top_group.seq_length != line.size()) {
            std::vector<Node> suffix_recurse = LineParser(line.substr(top_group.end_position + top_group.seq_length));
            result.insert(result.end(), suffix_recurse.begin(), suffix_recurse.end());
        }
    }
    return result;
}

std::vector<Node> FileParser(std::string path) {
    std::ifstream file(path);

    std::vector<Node> result;
    std::string line;
    while( std::getline(file, line) ) {
        if(line == "") {
            result.push_back(Node(Empty));
            continue;
        }

        std::regex header("^#+\\s");
        std::smatch header_match;
        bool is_header = std::regex_search(line, header_match, header);

        Node LineNode(Paragraph);
        if(is_header)
        {
            int length = header_match.str().size() - 1;
            if(length < 7) {
                LineNode.contents = std::to_string(length);
                LineNode.value = Header;
                line = header_match.suffix().str();
            }
        }

        LineNode.children = LineParser(line);
        result.push_back(LineNode);
    }
    file.close();
    return result;
}

std::vector<Node> CompressedAST(std::vector<Node> raw_ast) {
    auto result = raw_ast;
    int current = 0;
    while(current + 1 != result.size() ) {
        if(result[current].value == result[current + 1].value) {
            for(auto child : result[current + 1].children) result[current].children.push_back(child);
            result.erase(next(result.begin(), current + 1));
        } else current++;
    }
    return result;
}