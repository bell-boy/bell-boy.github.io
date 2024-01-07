#include "./src/transpiler.h"
#include <iostream>
#include <assert.h>

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
    std::vector<Node> result = FileParser("../testfiles/test.md");
    assert(result.size() == 3);
    
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

int compression_test() {
    std::vector<Node> raw_ast = FileParser("../testfiles/test2.md");
    std::vector<Node> result = CompressedAST(raw_ast);

    int sub_test_1 = result.size() == 3;
    if(!sub_test_1) return 1;

    return 0;
}

void representation_test() {
    std::vector<Node> test_data = FileParser("../testfiles/test3.md");
    std::string result = buildNode(test_data[0]);
    assert(result == "<h1><em>italics</em> header</h1>");
}

void list_test() {
    std::vector<Node> test_data = CompressedAST(FileParser("../testfiles/test4.md"));
    assert(test_data.size() == 1);
    std::string result = buildNode(test_data[0]);
    assert(result == "<ul><li>item 1</li><li>item 2</li></ul>");
}

void esacpe_test() { 
    std::vector<Node> test_data = CompressedAST(FileParser("../testfiles/test5.md"));
    assert(test_data.size() == 1);
    std::string result = buildNode(test_data[0]);
    assert(result == "<p>this is normal *this is also normal* this is again normal</p>");
}

void link_test_graph() { 
    std::vector<Node> test_data = LineParser("[this is a link](to this dot com)");
    assert(test_data.size() == 1);
    assert(test_data[0].value == Link);
    assert(test_data[0].contents == "to this dot com");
    assert(test_data[0].children[0].contents == "this is a link");
}

void link_test_string() {
    std::string test_data = buildNode(LineParser("[this is a link](to this dot com)")[0]);
    assert(test_data == "<a href=\"to this dot com\">this is a link</a>");
}

int main() {
    std::cout << "italic_test: " << italic_test() << std::endl;
    std::cout << "bold_test: " << bold_test() << std::endl;
    std::cout << "bold_italic_test: " << bold_italic_test() << std::endl; 
    std::cout << "file_test: " << file_test() << std::endl; 
    std::cout << "compression_test: " << compression_test() << std::endl;
    representation_test();
    list_test();
    esacpe_test();
    link_test_graph();
    link_test_string();
    return 0;
}