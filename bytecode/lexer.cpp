#include "lexer.h"
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