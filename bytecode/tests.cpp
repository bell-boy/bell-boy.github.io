#include "lexer.h"
#include <iostream>

// verifiy that we see the expected graph structure
int italic_test() {
    std::string test_string = "this isn't either*this is in italics*this is not*";
    std::vector<Node> result = LineParser(test_string);
    bool sub_test_4 = result.size() == 3;
    if(!sub_test_4) return 4;

    bool sub_test_1 = (result[0].value == Text) && (result[0].contents == "this isn't either");
    bool sub_test_2 = (result[1].value == Italic) && (result[1].children[0].contents == "this is in italics");
    bool sub_test_3 = (result[2].value == Text) && (result[2].contents == "this is not*");

    if(!sub_test_1) return 1;
    if(!sub_test_2) return 2;
    if(!sub_test_3) return 3;

    return 0;
}

int bold_test() {
    std::string test_string = "**this is bold *this is in italics*** this isn't anything";
    std::vector<Node> result = LineParser(test_string);
    bool sub_test_4 = result.size() == 2;
    if(!sub_test_4) return 4;
    bool sub_test_1 = (result[0].value == Bold) && (result[0].children.size() == 1);
    bool sub_test_2 = (result[0].children[0].value == Text) && (result[0].children[0].contents == "this is bold *this is in italics");
    bool sub_test_3 = (result[1].value == Text) && (result[1].contents == "* this isn't anything");
    

    if(!sub_test_1) return 1;
    if(!sub_test_2) return 2;
    if(!sub_test_3) return 3;

    return 0;
}

int bold_italic_test() {
    std::string test_string = "***this is bold italics*** this is normal **this is bold**";
    std::vector<Node> result = LineParser(test_string);
    bool sub_test_4 = result.size() == 3;
    if(!sub_test_4) return 4;
    
    bool sub_test_1 = (result[0].value == BoldItalic) && (result[0].children.size() == 1);
    bool sub_test_2 = (result[0].children[0].value == Text) && (result[0].children[0].contents == "this is bold italics");
    bool sub_test_3 = (result[1].value == Text) && (result[1].contents == " this is normal ");
    bool sub_test_5 = (result[2].value == Bold) && (result[2].children[0].contents == "this is bold");
    

    if(!sub_test_1) return 1;
    if(!sub_test_2) return 2;
    if(!sub_test_3) return 3;
    if(!sub_test_5) return 5;

    return 0;
}

int file_test() {
    std::vector<Node> result = FileParser("test.md");
    bool sub_test_4 = result.size() == 3;
    if(!sub_test_4) return 4;
    
    bool sub_test_1 = (result[0].value == Header) && (result[0].children.size() == 1) && (result[0].contents == "1");
    bool sub_test_2 = (result[0].children[0].value == Text) && (result[0].children[0].contents == "this is a header");
    bool sub_test_3 = (result[1].value == Paragraph) && (result[1].children[0].contents == "this is not a header");
    bool sub_test_5 = (result[2].value == Header) && (result[2].children[0].contents == "this is an h2 header") && (result[2].contents == "2");
    

    if(!sub_test_1) return 1;
    if(!sub_test_2) return 2;
    if(!sub_test_3) return 3;
    if(!sub_test_5) return 5;

    return 0;
}


int main() {
    std::cout << "italic_test: " << italic_test() << std::endl;
    std::cout << "bold_test: " << bold_test() << std::endl;
    std::cout << "bold_italic_test: " << bold_italic_test() << std::endl; 
    std::cout << "file_test: " << file_test() << std::endl; 
    return 0;
}