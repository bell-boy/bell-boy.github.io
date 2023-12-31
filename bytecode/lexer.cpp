#include "lexer.h"
#include <iostream>

std::vector<Node> LineParser(std::string line) {
    std::vector<Node> result;
    if(line == "") return result;

    std::string current_suffix;
    int offset;
    std::vector<TokenMatch> order;
    
    std::smatch italics_match;
    TokenMatch italics_group(Italic, std::regex("\\*"), 1);
    italics_group.start_position = italics_group.end_position = -1; // -1 signifies that we don't have a position for this token.
    offset = 0;
    current_suffix = line;

    while (std::regex_search(current_suffix, italics_match, italics_group.pattern) ) {
        if(italics_group.start_position == -1) {
            italics_group.start_position = italics_match.position(0) + offset;
        } else if (italics_group.end_position == -1) {
            italics_group.end_position = italics_match.position(0) + offset;
            order.push_back(italics_group);
        } else break;
        offset += italics_match.prefix().length() + italics_group.seq_length;
        current_suffix = italics_match.suffix().str();
    }

    std::smatch bold_match;
    TokenMatch bold_group(Bold, std::regex("\\*{2}"), 2);
    bold_group.start_position = bold_group.end_position = -1; // -1 signifies that we don't have a position for this token.
    offset = 0;
    current_suffix = line;

    while ( std::regex_search(current_suffix, bold_match, bold_group.pattern) ) {
        if(bold_group.start_position == -1) {
            bold_group.start_position = bold_match.position(0) + offset;
        } else if (bold_group.end_position == -1) {
            bold_group.end_position = bold_match.position(0) + offset;
            order.push_back(bold_group);
            break;
        } 
        offset += bold_match.prefix().length() + bold_group.seq_length;
        current_suffix = bold_match.suffix().str();
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