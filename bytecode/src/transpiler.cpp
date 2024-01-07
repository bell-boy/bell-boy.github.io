#include "transpiler.h"

std::string buildNode(Node node) {
    if(node.value == Text) return node.contents;
    if(node.value == Empty) return "\n";
    std::string innerHTML = "";
    for(auto child : node.children) innerHTML += buildNode(child);

    switch(node.value) {
        case BoldItalic:
            return "<em><strong>" + innerHTML + "</strong></em>";
            break;
        case Bold:
            return "<strong>" + innerHTML + "</strong>";
            break;
        case Italic:
            return "<em>" + innerHTML + "</em>";
            break;  
        case Paragraph:
            return "<p>" + innerHTML + "</p>";
            break;  
        case Header:
            return "<h" + node.contents + ">" + innerHTML + "</h" + node.contents + ">";
            break;
        case List:
            return "<ul>" + innerHTML + "</ul>";
            break;
        case ListItem:
            return "<li>" + innerHTML + "</li>";
            break; 
        case Link:
            return "<a href=\"" + node.contents + "\">" + innerHTML + "</a>";
            break;
    } 
}