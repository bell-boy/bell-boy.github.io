#include <vector>
#include <string>
#include <regex>
#include <fstream>

enum NodeValue {BoldItalic, Bold, Italic, Header, List, ListItem, Paragraph, Text, Empty};

class Node {
    public:
        NodeValue value;
        std::string contents;
        std::vector<Node> children;

        Node(NodeValue value) : value(value) { }
        Node(NodeValue value, std::string contents) : value(value), contents(contents) { }
};

class TokenMatch {
    public:
        NodeValue value;
        std::regex pattern;
        // -1 signifies that we don't have a position for this token.
        int start_position = -1;
        int end_position = -1;
        int seq_length;
        TokenMatch(NodeValue value, std::regex pattern, int seq_length) : value(value), pattern(pattern), seq_length(seq_length) {}

    bool operator<(TokenMatch other) const {
        if (other.start_position != start_position) return start_position < other.start_position;
        return value < other.value;
    }
};

std::vector<Node> LineParser(std::string line);

std::vector<Node> FileParser(std::string path);

std::vector<Node> CompressedAST(std::vector<Node> raw_ast);